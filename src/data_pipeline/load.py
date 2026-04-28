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
                f"length mismatch in {prefix}: src={len(src_arr)} dst={len(dst_arr)} ts={len(ts_arr)}"
            )
        np.save(os.path.join(data_dir, f"{prefix}_ts.npy"), ts_arr)
    logger.info("  saved %-40s %10d edges  ts=%s", prefix, len(src_arr), ts is not None)


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
                    f"length mismatch in {prefix}: src={len(src_arr)} dst={len(dst_arr)} ts={len(ts_arr)}"
                )
            np.save(os.path.join(data_dir, f"{prefix}_ts.npy"), ts_arr)
        logger.info("  saved %-40s %10d edges  ts=%s", prefix, len(src_arr), ts_arr is not None)

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
    """Save val/test artifacts: edge arrays, multi-positive ground truth, parquet."""
    user_arr = np.ascontiguousarray(df["user_idx"].to_numpy(), dtype=np.int64)
    item_arr = np.ascontiguousarray(df["item_idx"].to_numpy(), dtype=np.int64)
    ts_arr   = np.ascontiguousarray(df["timestamp"].to_numpy(), dtype=np.int64)

    np.save(os.path.join(data_dir, f"{split_name}_user_idx.npy"),    user_arr)
    np.save(os.path.join(data_dir, f"{split_name}_product_idx.npy"), item_arr)
    np.save(os.path.join(data_dir, f"{split_name}_timestamp.npy"),   ts_arr)

    gt = _build_ground_truth(df)
    pkl_path = os.path.join(data_dir, f"{split_name}_ground_truth.pkl")
    with open(pkl_path, "wb") as fh:
        pickle.dump(gt, fh, protocol=pickle.HIGHEST_PROTOCOL)

    parquet_cols = ["user_idx", "item_idx", "timestamp"]
    parquet_df = df[parquet_cols].astype({
        "user_idx": "int64", "item_idx": "int64", "timestamp": "int64"
    }).reset_index(drop=True)
    parquet_df.to_parquet(
        os.path.join(graph_dir, f"{split_name}_ground_truth.parquet"), index=False,
    )

    logger.info(
        "  saved %s: %d positives across %d users",
        split_name, len(df), len(gt),
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
    os.makedirs(data_dir,   exist_ok=True)
    os.makedirs(struct_dir, exist_ok=True)
    os.makedirs(graph_dir,  exist_ok=True)

    logger.info("Saving edge arrays to %s ...", data_dir)

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

    val_gt  = save_eval_split(split.val,  data_dir, graph_dir, "val")
    test_gt = save_eval_split(split.test, data_dir, graph_dir, "test")

    primary_path  = os.path.join(data_dir, "train_mask_purchase_only.pkl")
    seen_all_path = os.path.join(data_dir, "train_mask_seen_all.pkl")
    with open(primary_path, "wb") as fh:
        pickle.dump(train_mask_primary, fh, protocol=pickle.HIGHEST_PROTOCOL)
    with open(seen_all_path, "wb") as fh:
        pickle.dump(train_mask_seen_all, fh, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(data_dir, "train_mask.pkl"), "wb") as fh:
        pickle.dump(train_mask_primary, fh, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(
        "  saved primary mask (purchase-only) %d users, seen-all mask %d users",
        len(train_mask_primary), len(train_mask_seen_all),
    )

    logger.info("Saving structural parquets to %s ...", struct_dir)
    prod_cat_df.to_parquet(os.path.join(struct_dir, "product_category.parquet"), index=False)
    prod_brand_df.to_parquet(os.path.join(struct_dir, "product_brand.parquet"), index=False)
    item_metadata_df.to_parquet(os.path.join(graph_dir, "item_metadata.parquet"), index=False)
    logger.info(
        "  product_category.parquet: %d rows  product_brand.parquet: %d rows  item_metadata.parquet: %d rows",
        len(prod_cat_df), len(prod_brand_df), len(item_metadata_df),
    )

    if train_events_spark is not None:
        train_events_path = os.path.join(graph_dir, "train_events.parquet")
        tmp = os.path.join(graph_dir, "_tmp_train_events")
        try:
            train_events_spark.write.mode("overwrite").parquet(tmp)
            table = pq.read_table(tmp)
            pq.write_table(table, train_events_path)
            logger.info("  train_events.parquet rows=%d", table.num_rows)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    logger.info("Saving vocabulary mappings to %s ...", struct_dir)
    for name, mapping in (
        ("user2idx",     split.user2idx),
        ("item2idx",     split.item2idx),
        ("category2idx", category2idx),
        ("brand2idx",    brand2idx),
        ("behavior2idx", behavior2idx),
    ):
        path = os.path.join(struct_dir, f"{name}.json")
        with open(path, "w") as fh:
            json.dump({str(k): int(v) for k, v in mapping.items()}, fh)
        logger.info("  %s: %d entries", name, len(mapping))

    compute_and_save_svd_factors(
        data_dir=data_dir,
        num_users=split.num_users,
        num_items=split.num_items,
        rank=256,
        n_iter=4,
    )

    return {"val_ground_truth": val_gt, "test_ground_truth": test_gt}


def save_node_counts(node_counts: dict[str, int], data_dir: str) -> None:
    path = os.path.join(data_dir, "node_counts.json")
    with open(path, "w") as fh:
        json.dump(node_counts, fh, indent=2)
    logger.info("node_counts.json written to %s: %s", path, node_counts)


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

    logger.info("Running pre-training sanity check ...")

    def _npy(prefix: str, suffix: str) -> np.ndarray:
        return np.load(os.path.join(data_dir, f"{prefix}_{suffix}.npy"))

    behavior_artifacts = {}
    for beh in ("view", "cart", "purchase"):
        behavior_artifacts[beh] = {
            "src": _npy(f"{beh}_train", "src"),
            "dst": _npy(f"{beh}_train", "dst"),
            "ts":  _npy(f"{beh}_train", "ts"),
        }

    val_user = _npy("val", "user_idx")
    val_item = _npy("val", "product_idx")
    val_ts   = _npy("val", "timestamp")
    test_user = _npy("test", "user_idx")
    test_item = _npy("test", "product_idx")
    test_ts   = _npy("test", "timestamp")

    sanity_check_temporal_artifacts(
        behavior_artifacts=behavior_artifacts,
        val_ts=val_ts, test_ts=test_ts,
        train_cutoff_ts=train_cutoff_ts,
        val_cutoff_ts=val_cutoff_ts,
    )

    with open(os.path.join(data_dir, "val_ground_truth.pkl"),  "rb") as fh:
        val_gt = pickle.load(fh)
    with open(os.path.join(data_dir, "test_ground_truth.pkl"), "rb") as fh:
        test_gt = pickle.load(fh)

    val_parquet  = pd.read_parquet(os.path.join(graph_dir, "val_ground_truth.parquet"))
    test_parquet = pd.read_parquet(os.path.join(graph_dir, "test_ground_truth.parquet"))

    sanity_check_ground_truth(val_gt,  val_parquet,  split_name="val")
    sanity_check_ground_truth(test_gt, test_parquet, split_name="test")

    with open(os.path.join(data_dir, "train_mask_purchase_only.pkl"), "rb") as fh:
        primary_mask = pickle.load(fh)
    sanity_check_eval_mask(primary_mask, val_gt,  split_name="val")
    sanity_check_eval_mask(primary_mask, test_gt, split_name="test")

    pc = pd.read_parquet(os.path.join(struct_dir, "product_category.parquet"))
    pb = pd.read_parquet(os.path.join(struct_dir, "product_brand.parquet"))
    category_ei = torch.from_numpy(pc[["product_idx", "category_idx"]].values.T.copy()).long()
    brand_ei    = torch.from_numpy(pb[["product_idx", "brand_idx"]].values.T.copy()).long()

    view_ei     = torch.from_numpy(np.stack([behavior_artifacts["view"]["src"],     behavior_artifacts["view"]["dst"]])).long().contiguous()
    cart_ei     = torch.from_numpy(np.stack([behavior_artifacts["cart"]["src"],     behavior_artifacts["cart"]["dst"]])).long().contiguous()
    purchase_ei = torch.from_numpy(np.stack([behavior_artifacts["purchase"]["src"], behavior_artifacts["purchase"]["dst"]])).long().contiguous()

    hetero = HeteroData()
    for ntype, n in node_counts.items():
        hetero[ntype].x         = torch.arange(n, dtype=torch.long)
        hetero[ntype].num_nodes = n

    hetero[("user",     "view",         "product")].edge_index = view_ei
    hetero[("user",     "cart",         "product")].edge_index = cart_ei
    hetero[("user",     "purchase",     "product")].edge_index = purchase_ei
    hetero[("user",     "view",         "product")].edge_time  = torch.from_numpy(behavior_artifacts["view"]["ts"]).long()
    hetero[("user",     "cart",         "product")].edge_time  = torch.from_numpy(behavior_artifacts["cart"]["ts"]).long()
    hetero[("user",     "purchase",     "product")].edge_time  = torch.from_numpy(behavior_artifacts["purchase"]["ts"]).long()

    hetero[("product",  "rev_view",     "user")].edge_index    = view_ei.flip(0).contiguous()
    hetero[("product",  "rev_cart",     "user")].edge_index    = cart_ei.flip(0).contiguous()
    hetero[("product",  "rev_purchase", "user")].edge_index    = purchase_ei.flip(0).contiguous()
    hetero[("product",  "rev_view",     "user")].edge_time     = hetero[("user", "view",     "product")].edge_time
    hetero[("product",  "rev_cart",     "user")].edge_time     = hetero[("user", "cart",     "product")].edge_time
    hetero[("product",  "rev_purchase", "user")].edge_time     = hetero[("user", "purchase", "product")].edge_time

    hetero[("product",  "belongs_to",   "category")].edge_index = category_ei.contiguous()
    hetero[("category", "contains",     "product")].edge_index  = category_ei.flip(0).contiguous()
    hetero[("product",  "producedBy",   "brand")].edge_index    = brand_ei.contiguous()
    hetero[("brand",    "brands",       "product")].edge_index  = brand_ei.flip(0).contiguous()

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
        check_leakage=False,
        verbose=True,
    )
    logger.info("Sanity check PASSED — pipeline is ready for training.")


def compute_and_save_svd_factors(
    data_dir: str,
    num_users: int,
    num_items: int,
    *,
    rank: int = 256,
    n_iter: int = 4,
    behaviors: tuple[str, ...] = ("view", "cart", "purchase"),
    seed: int = 42,
) -> None:
    import scipy.sparse as sp
    import torch
    from sklearn.utils.extmath import randomized_svd
    from src.core.contracts import SVDFactors

    logger.info(
        "Computing SVD factors  rank=%d  n_iter=%d  behaviors=%s",
        rank, n_iter, behaviors,
    )

    US_dict: dict[str, torch.Tensor] = {}
    VS_dict: dict[str, torch.Tensor] = {}

    for beh in behaviors:
        src_path = os.path.join(data_dir, f"{beh}_train_src.npy")
        dst_path = os.path.join(data_dir, f"{beh}_train_dst.npy")
        if not (os.path.exists(src_path) and os.path.exists(dst_path)):
            raise FileNotFoundError(
                f"Missing edge files for behavior {beh!r}: {src_path} / {dst_path}. "
                "Call save_artifacts() first."
            )

        src = np.load(src_path)
        dst = np.load(dst_path)

        if len(src) == 0:
            US_dict[beh] = torch.zeros((num_users, rank), dtype=torch.float32)
            VS_dict[beh] = torch.zeros((num_items, rank), dtype=torch.float32)
            continue

        A = sp.csr_matrix(
            (np.ones(len(src), dtype=np.float32), (src, dst)),
            shape=(num_users, num_items),
            dtype=np.float32,
        )
        A.sum_duplicates()
        A.data = np.minimum(A.data, 1.0)

        deg_u = np.asarray(A.sum(axis=1)).ravel().astype(np.float32)
        deg_v = np.asarray(A.sum(axis=0)).ravel().astype(np.float32)
        d_u_inv = np.where(deg_u > 0, 1.0 / np.sqrt(deg_u), 0.0).astype(np.float32)
        d_v_inv = np.where(deg_v > 0, 1.0 / np.sqrt(deg_v), 0.0).astype(np.float32)
        A_norm = (sp.diags(d_u_inv) @ A @ sp.diags(d_v_inv)).tocsr()

        logger.info(
            "  [%s] adj shape=%s nnz=%d  -> randomized_svd",
            beh, A_norm.shape, A_norm.nnz,
        )

        effective_rank = min(rank, min(A_norm.shape) - 1)
        if effective_rank < 1:
            US_dict[beh] = torch.zeros((num_users, rank), dtype=torch.float32)
            VS_dict[beh] = torch.zeros((num_items, rank), dtype=torch.float32)
            continue

        U, S, Vt = randomized_svd(
            A_norm, n_components=effective_rank, n_iter=n_iter, random_state=seed,
        )
        sqrt_S = np.sqrt(S).astype(np.float32)
        US = (U * sqrt_S).astype(np.float32)
        VS = (Vt.T * sqrt_S).astype(np.float32)

        if effective_rank < rank:
            US = np.pad(US, ((0, 0), (0, rank - effective_rank)))
            VS = np.pad(VS, ((0, 0), (0, rank - effective_rank)))

        US_dict[beh] = torch.from_numpy(US)
        VS_dict[beh] = torch.from_numpy(VS)

        del A, A_norm, U, S, Vt, US, VS, sqrt_S, deg_u, deg_v, d_u_inv, d_v_inv

    svd = SVDFactors(US=US_dict, VS=VS_dict)
    svd.validate()

    out_path = os.path.join(data_dir, "svd_factors.pt")
    torch.save(svd, out_path)
    size_mb = os.path.getsize(out_path) / (1024 ** 2)
    logger.info("SVD factors saved: %s  (%.1f MB)", out_path, size_mb)
