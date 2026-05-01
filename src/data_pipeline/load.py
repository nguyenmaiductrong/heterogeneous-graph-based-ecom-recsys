"""Ghi artefact pipeline (numpy, parquet, pickle, JSON) và chạy kiểm tra sanity."""
from __future__ import annotations

import json
import logging
import os
import pickle
import shutil

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from .sanity import (
    sanity_check_eval_mask,
    sanity_check_ground_truth,
    sanity_check_heterodata,
    sanity_check_temporal_artifacts,
)
from .splitter import SplitResult

logger = logging.getLogger(__name__)


def _save_ei_npy(
    src: np.ndarray,
    dst: np.ndarray,
    data_dir: str,
    prefix: str,
    ts: np.ndarray | None = None,
) -> None:
    src_arr = np.ascontiguousarray(src, dtype=np.int64)
    dst_arr = np.ascontiguousarray(dst, dtype=np.int64)
    np.save(os.path.join(data_dir, f"{prefix}_src.npy"), src_arr)
    np.save(os.path.join(data_dir, f"{prefix}_dst.npy"), dst_arr)
    if ts is not None:
        ts_arr = np.ascontiguousarray(ts, dtype=np.int64)
        if not (len(src_arr) == len(dst_arr) == len(ts_arr)):
            raise RuntimeError(
                f"độ dài không khớp tại {prefix}: src={len(src_arr)} dst={len(dst_arr)} ts={len(ts_arr)}"
            )
        np.save(os.path.join(data_dir, f"{prefix}_ts.npy"), ts_arr)
    logger.info("đã lưu %-40s %10d cạnh có_ts=%s", prefix, len(src_arr), ts is not None)


def _spark_ei_to_npy(
    spark_df: DataFrame,
    src_col: str,
    dst_col: str,
    data_dir: str,
    prefix: str,
    ts_col: str | None = "timestamp",
) -> None:
    tmp_path = os.path.join(data_dir, f"_tmp_{prefix}")
    try:
        select_cols = [
            F.col(src_col).cast("long").alias("src"),
            F.col(dst_col).cast("long").alias("dst"),
        ]
        read_cols = ["src", "dst"]
        if ts_col is not None:
            select_cols.append(F.col(ts_col).cast("long").alias("ts"))
            read_cols.append("ts")

        (
            spark_df
            .select(*select_cols)
            .write.mode("overwrite").parquet(tmp_path)
        )

        table = pq.read_table(tmp_path, columns=read_cols)
        src_arr = np.ascontiguousarray(
            table.column("src").to_numpy(zero_copy_only=False), dtype=np.int64,
        )
        dst_arr = np.ascontiguousarray(
            table.column("dst").to_numpy(zero_copy_only=False), dtype=np.int64,
        )
        ts_arr = None
        if ts_col is not None:
            ts_arr = np.ascontiguousarray(
                table.column("ts").to_numpy(zero_copy_only=False), dtype=np.int64,
            )
        del table

        np.save(os.path.join(data_dir, f"{prefix}_src.npy"), src_arr)
        np.save(os.path.join(data_dir, f"{prefix}_dst.npy"), dst_arr)
        if ts_arr is not None:
            if not (len(src_arr) == len(dst_arr) == len(ts_arr)):
                raise RuntimeError(
                    f"độ dài không khớp tại {prefix}: src={len(src_arr)} dst={len(dst_arr)} ts={len(ts_arr)}"
                )
            np.save(os.path.join(data_dir, f"{prefix}_ts.npy"), ts_arr)
        logger.info("đã lưu %-40s %10d cạnh có_ts=%s", prefix, len(src_arr), ts_arr is not None)

    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def _build_ground_truth(df: pd.DataFrame) -> dict[int, list[int]]:
    if df.empty:
        return {}
    grouped = (
        df.groupby("user_idx")["item_idx"]
        .apply(lambda s: sorted({int(x) for x in s}))
    )
    return {int(u): list(items) for u, items in grouped.items()}


def save_eval_split(
    df: pd.DataFrame,
    data_dir: str,
    graph_dir: str,
    split_name: str,
) -> dict[int, list[int]]:
    """Lưu artefact val/test: mảng cạnh, ground truth đa-positive, parquet.

    Các nhãn dương eval được khử trùng theo (user_idx, item_idx); bản ghi trùng
    chỉ giữ timestamp sớm nhất để mọi file (.npy, .pkl, .parquet) cùng tập cặp duy nhất.
    """
    raw_eval_events = len(df)

    parquet_cols = ["user_idx", "item_idx", "timestamp"]
    unique_df = (
        df[parquet_cols]
        .astype({"user_idx": "int64", "item_idx": "int64", "timestamp": "int64"})
        .groupby(["user_idx", "item_idx"], as_index=False, sort=True)["timestamp"]
        .min()
        .reset_index(drop=True)
    )
    unique_eval_positives = len(unique_df)

    user_arr = np.ascontiguousarray(unique_df["user_idx"].to_numpy(), dtype=np.int64)
    item_arr = np.ascontiguousarray(unique_df["item_idx"].to_numpy(), dtype=np.int64)
    ts_arr = np.ascontiguousarray(unique_df["timestamp"].to_numpy(), dtype=np.int64)

    np.save(os.path.join(data_dir, f"{split_name}_user_idx.npy"), user_arr)
    np.save(os.path.join(data_dir, f"{split_name}_product_idx.npy"), item_arr)
    np.save(os.path.join(data_dir, f"{split_name}_timestamp.npy"), ts_arr)

    gt = _build_ground_truth(unique_df)
    pkl_path = os.path.join(data_dir, f"{split_name}_ground_truth.pkl")
    with open(pkl_path, "wb") as fh:
        pickle.dump(gt, fh, protocol=pickle.HIGHEST_PROTOCOL)

    unique_df.to_parquet(
        os.path.join(graph_dir, f"{split_name}_ground_truth.parquet"), index=False,
    )

    logger.info(
        "đã lưu %s: raw_eval_events=%d unique_eval_positives=%d trong %d user",
        split_name, raw_eval_events, unique_eval_positives, len(gt),
    )
    return gt


def save_artifacts(
    split: SplitResult,
    aux_spark: DataFrame,
    train_mask_primary: dict,
    train_mask_seen_all: dict,
    prod_cat_df: pd.DataFrame,
    prod_brand_df: pd.DataFrame,
    category2idx: dict,
    brand2idx: dict,
    behavior2idx: dict,
    item_metadata_df: pd.DataFrame,
    data_dir: str,
    struct_dir: str,
    graph_dir: str,
    train_events_spark: DataFrame | None = None,
) -> dict[str, dict]:
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(struct_dir, exist_ok=True)
    os.makedirs(graph_dir, exist_ok=True)

    logger.info("Đang lưu mảng cạnh vào %s ...", data_dir)

    _save_ei_npy(
        split.train["user_idx"].to_numpy(),
        split.train["item_idx"].to_numpy(),
        data_dir, "purchase_train",
        ts=split.train["timestamp"].to_numpy(),
    )

    for beh in ("view", "cart"):
        _spark_ei_to_npy(
            aux_spark.filter(F.col("event_type") == beh),
            src_col="user_idx",
            dst_col="item_idx",
            data_dir=data_dir,
            prefix=f"{beh}_train",
            ts_col="timestamp",
        )

    val_gt = save_eval_split(split.val, data_dir, graph_dir, "val")
    test_gt = save_eval_split(split.test, data_dir, graph_dir, "test")

    primary_path = os.path.join(data_dir, "train_mask_purchase_only.pkl")
    seen_all_path = os.path.join(data_dir, "train_mask_seen_all.pkl")
    with open(primary_path, "wb") as fh:
        pickle.dump(train_mask_primary, fh, protocol=pickle.HIGHEST_PROTOCOL)
    with open(seen_all_path, "wb") as fh:
        pickle.dump(train_mask_seen_all, fh, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(data_dir, "train_mask.pkl"), "wb") as fh:
        pickle.dump(train_mask_primary, fh, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(
        "đã lưu mask chính (chỉ purchase) %d user, mask seen-all %d user",
        len(train_mask_primary), len(train_mask_seen_all),
    )

    logger.info("Đang lưu parquet cấu trúc vào %s ...", struct_dir)
    prod_cat_df.to_parquet(os.path.join(struct_dir, "product_category.parquet"), index=False)
    prod_brand_df.to_parquet(os.path.join(struct_dir, "product_brand.parquet"), index=False)
    item_metadata_df.to_parquet(os.path.join(graph_dir, "item_metadata.parquet"), index=False)
    logger.info(
        "product_category.parquet: %d dòng product_brand.parquet: %d dòng "
        "item_metadata.parquet: %d dòng",
        len(prod_cat_df), len(prod_brand_df), len(item_metadata_df),
    )

    if train_events_spark is not None:
        train_events_path = os.path.join(graph_dir, "train_events.parquet")
        # Ghi thẳng thư mục parquet (phân partition); không kéo toàn bộ lên driver
        # qua pyarrow — log sự kiện train có thể hàng chục triệu dòng và làm OOM driver.
        train_events_spark.write.mode("overwrite").parquet(train_events_path)
        logger.info("đã ghi train_events.parquet tại %s", train_events_path)

    if split.candidate_item_idx is not None:
        candidate_arr = np.ascontiguousarray(split.candidate_item_idx, dtype=np.int64)
        np.save(os.path.join(data_dir, "candidate_item_idx.npy"), candidate_arr)
        logger.info("candidate_item_idx.npy: %d item", len(candidate_arr))

    logger.info("Đang lưu ánh xạ từ vựng vào %s ...", struct_dir)
    for name, mapping in (
        ("user2idx", split.user2idx),
        ("item2idx", split.item2idx),
        ("category2idx", category2idx),
        ("brand2idx", brand2idx),
        ("behavior2idx", behavior2idx),
    ):
        path = os.path.join(struct_dir, f"{name}.json")
        with open(path, "w") as fh:
            json.dump({str(k): int(v) for k, v in mapping.items()}, fh)
        logger.info("%s: %d entry", name, len(mapping))

    return {"val_ground_truth": val_gt, "test_ground_truth": test_gt}


def save_node_counts(node_counts: dict[str, int], data_dir: str) -> None:
    path = os.path.join(data_dir, "node_counts.json")
    with open(path, "w") as fh:
        json.dump(node_counts, fh, indent=2)
    logger.info("đã ghi node_counts.json tại %s: %s", path, node_counts)


def verify_artifacts(
    data_dir: str,
    struct_dir: str,
    graph_dir: str,
    node_counts: dict[str, int],
    *,
    train_cutoff_ts: int,
    val_cutoff_ts: int,
) -> None:
    import torch
    from torch_geometric.data import HeteroData

    logger.info("Đang chạy sanity check trước huấn luyện ...")

    def _npy(prefix: str, suffix: str) -> np.ndarray:
        return np.load(os.path.join(data_dir, f"{prefix}_{suffix}.npy"))

    behavior_artifacts = {}
    for beh in ("view", "cart", "purchase"):
        behavior_artifacts[beh] = {
            "src": _npy(f"{beh}_train", "src"),
            "dst": _npy(f"{beh}_train", "dst"),
            "ts": _npy(f"{beh}_train", "ts"),
        }

    val_user = _npy("val", "user_idx")
    val_item = _npy("val", "product_idx")
    val_ts = _npy("val", "timestamp")
    test_user = _npy("test", "user_idx")
    test_item = _npy("test", "product_idx")
    test_ts = _npy("test", "timestamp")

    sanity_check_temporal_artifacts(
        behavior_artifacts=behavior_artifacts,
        val_ts=val_ts, test_ts=test_ts,
        train_cutoff_ts=train_cutoff_ts,
        val_cutoff_ts=val_cutoff_ts,
    )

    with open(os.path.join(data_dir, "val_ground_truth.pkl"), "rb") as fh:
        val_gt = pickle.load(fh)
    with open(os.path.join(data_dir, "test_ground_truth.pkl"), "rb") as fh:
        test_gt = pickle.load(fh)

    val_parquet = pd.read_parquet(os.path.join(graph_dir, "val_ground_truth.parquet"))
    test_parquet = pd.read_parquet(os.path.join(graph_dir, "test_ground_truth.parquet"))

    sanity_check_ground_truth(val_gt, val_parquet, split_name="val")
    sanity_check_ground_truth(test_gt, test_parquet, split_name="test")

    with open(os.path.join(data_dir, "train_mask_purchase_only.pkl"), "rb") as fh:
        primary_mask = pickle.load(fh)
    sanity_check_eval_mask(primary_mask, val_gt, split_name="val")
    sanity_check_eval_mask(primary_mask, test_gt, split_name="test")

    pc = pd.read_parquet(os.path.join(struct_dir, "product_category.parquet"))
    pb = pd.read_parquet(os.path.join(struct_dir, "product_brand.parquet"))
    category_ei = torch.from_numpy(pc[["product_idx", "category_idx"]].values.T.copy()).long()
    brand_ei = torch.from_numpy(pb[["product_idx", "brand_idx"]].values.T.copy()).long()

    view_ei = torch.from_numpy(np.stack([
        behavior_artifacts["view"]["src"],
        behavior_artifacts["view"]["dst"],
    ])).long().contiguous()
    cart_ei = torch.from_numpy(np.stack([
        behavior_artifacts["cart"]["src"],
        behavior_artifacts["cart"]["dst"],
    ])).long().contiguous()
    purchase_ei = torch.from_numpy(np.stack([
        behavior_artifacts["purchase"]["src"],
        behavior_artifacts["purchase"]["dst"],
    ])).long().contiguous()

    hetero = HeteroData()
    for ntype, n in node_counts.items():
        hetero[ntype].x = torch.arange(n, dtype=torch.long)
        hetero[ntype].num_nodes = n

    hetero[("user", "view", "product")].edge_index = view_ei
    hetero[("user", "cart", "product")].edge_index = cart_ei
    hetero[("user", "purchase", "product")].edge_index = purchase_ei
    hetero[("user", "view", "product")].edge_time = torch.from_numpy(
        behavior_artifacts["view"]["ts"],
    ).long()
    hetero[("user", "cart", "product")].edge_time = torch.from_numpy(
        behavior_artifacts["cart"]["ts"],
    ).long()
    hetero[("user", "purchase", "product")].edge_time = torch.from_numpy(
        behavior_artifacts["purchase"]["ts"],
    ).long()

    hetero[("product", "rev_view", "user")].edge_index = view_ei.flip(0).contiguous()
    hetero[("product", "rev_cart", "user")].edge_index = cart_ei.flip(0).contiguous()
    hetero[("product", "rev_purchase", "user")].edge_index = purchase_ei.flip(0).contiguous()
    hetero[("product", "rev_view", "user")].edge_time = hetero[("user", "view", "product")].edge_time
    hetero[("product", "rev_cart", "user")].edge_time = hetero[("user", "cart", "product")].edge_time
    hetero[("product", "rev_purchase", "user")].edge_time = (
        hetero[("user", "purchase", "product")].edge_time
    )

    hetero[("product", "belongs_to", "category")].edge_index = category_ei.contiguous()
    hetero[("category", "contains", "product")].edge_index = category_ei.flip(0).contiguous()
    hetero[("product", "producedBy", "brand")].edge_index = brand_ei.contiguous()
    hetero[("brand", "brands", "product")].edge_index = brand_ei.flip(0).contiguous()

    train_triplets = torch.stack([
        purchase_ei[0],
        purchase_ei[1],
        torch.full((purchase_ei.size(1),), 2, dtype=torch.long),
    ], dim=1)

    eval_pairs = pd.DataFrame({
        "user_idx": np.concatenate([val_user, test_user]),
        "item_idx": np.concatenate([val_item, test_item]),
    })

    sanity_check_heterodata(
        hetero,
        train_triplets,
        eval_pairs,
        num_nodes_dict=node_counts,
        check_leakage=True,
        verbose=True,
    )

    candidate_path = os.path.join(data_dir, "candidate_item_idx.npy")
    assert os.path.exists(candidate_path), (
        f"Thiếu candidate_item_idx.npy tại {candidate_path}"
    )
    candidate_arr = np.load(candidate_path)
    assert candidate_arr.dtype == np.int64, (
        f"candidate_item_idx.npy có dtype {candidate_arr.dtype}, cần int64"
    )
    num_items = int(node_counts.get("product", 0))
    if candidate_arr.size > 0:
        assert int(candidate_arr.min()) >= 0, (
            f"candidate_item_idx.npy min {int(candidate_arr.min())} < 0"
        )
        assert int(candidate_arr.max()) < num_items, (
            f"candidate_item_idx.npy max {int(candidate_arr.max())} >= num_items={num_items}"
        )
    # Giao thức chính dùng tập ứng viên warm_train_items = np.arange(num_items)
    expected = np.arange(num_items, dtype=np.int64)
    assert np.array_equal(candidate_arr, expected), (
        "candidate_item_idx.npy phải bằng np.arange(num_items) với giao thức warm_train_items"
    )
    logger.info(
        "candidate_item_idx.npy OK: %d item (warm_train_items)",
        len(candidate_arr),
    )

    logger.info("Sanity check PASSED — pipeline sẵn sàng huấn luyện.")
