from __future__ import annotations

import json
import logging
import os
import pickle
import shutil

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from torch_geometric.data import HeteroData

from .sanity import sanity_check_heterodata
from .splitter import SplitResult

logger = logging.getLogger(__name__)


def _save_ei_npy(src: np.ndarray, dst: np.ndarray, data_dir: str, prefix: str) -> None:
    np.save(os.path.join(data_dir, f"{prefix}_src.npy"), np.ascontiguousarray(src, dtype=np.int64))
    np.save(os.path.join(data_dir, f"{prefix}_dst.npy"), np.ascontiguousarray(dst, dtype=np.int64))
    logger.info("  saved %-40s %10d edges", prefix, len(src))


def _spark_ei_to_npy(
    spark_df: DataFrame,
    src_col: str,
    dst_col: str,
    data_dir: str,
    prefix: str,
) -> None:
    """Write a large Spark edge DataFrame to NumPy without .collect().

    Workers write Parquet in parallel; PyArrow reads columnar on the driver.
    Avoids serialising 100M+ rows through the Python heap (OOM with toPandas).
    """
    tmp_path = os.path.join(data_dir, f"_tmp_{prefix}")
    try:
        (
            spark_df
            .select(
                F.col(src_col).cast("long").alias("src"),
                F.col(dst_col).cast("long").alias("dst"),
            )
            .write.mode("overwrite").parquet(tmp_path)
        )

        table = pq.read_table(tmp_path, columns=["src", "dst"])
        src_arr = np.ascontiguousarray(
            table.column("src").to_numpy(zero_copy_only=False),
            dtype=np.int64,
        )
        dst_arr = np.ascontiguousarray(
            table.column("dst").to_numpy(zero_copy_only=False),
            dtype=np.int64,
        )
        del table

        np.save(os.path.join(data_dir, f"{prefix}_src.npy"), src_arr)
        np.save(os.path.join(data_dir, f"{prefix}_dst.npy"), dst_arr)
        logger.info("  saved %-40s %10d edges", prefix, len(src_arr))

    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def save_artifacts(
    split: SplitResult,
    aux_spark: DataFrame,
    train_mask: dict,
    prod_cat_df: pd.DataFrame,
    prod_brand_df: pd.DataFrame,
    category2idx: dict,
    brand2idx: dict,
    data_dir: str,
    struct_dir: str,
) -> None:
    os.makedirs(data_dir,   exist_ok=True)
    os.makedirs(struct_dir, exist_ok=True)

    logger.info("Saving edge arrays to %s ...", data_dir)

    _save_ei_npy(
        split.train["user_idx"].to_numpy(),
        split.train["item_idx"].to_numpy(),
        data_dir, "purchase_train",
    )

    for beh in ("view", "cart"):
        _spark_ei_to_npy(
            aux_spark.filter(F.col("event_type") == beh),
            src_col="user_idx",
            dst_col="item_idx",
            data_dir=data_dir,
            prefix=f"{beh}_train",
        )

    np.save(
        os.path.join(data_dir, "test_user_idx.npy"),
        np.ascontiguousarray(split.test["user_idx"].to_numpy(), dtype=np.int64),
    )
    np.save(
        os.path.join(data_dir, "test_product_idx.npy"),
        np.ascontiguousarray(split.test["item_idx"].to_numpy(), dtype=np.int64),
    )
    logger.info("  saved test pairs: %d", len(split.test))

    np.save(
        os.path.join(data_dir, "val_user_idx.npy"),
        np.ascontiguousarray(split.val["user_idx"].to_numpy(), dtype=np.int64),
    )
    np.save(
        os.path.join(data_dir, "val_product_idx.npy"),
        np.ascontiguousarray(split.val["item_idx"].to_numpy(), dtype=np.int64),
    )
    logger.info("  saved val pairs:  %d", len(split.val))

    mask_path = os.path.join(data_dir, "train_mask.pkl")
    with open(mask_path, "wb") as fh:
        pickle.dump(train_mask, fh, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("  saved train_mask.pkl  (%d entries)", len(train_mask))

    logger.info("Saving structural parquets to %s ...", struct_dir)

    prod_cat_df.to_parquet(os.path.join(struct_dir, "product_category.parquet"), index=False)
    prod_brand_df.to_parquet(os.path.join(struct_dir, "product_brand.parquet"), index=False)
    logger.info(
        "  product_category.parquet: %d rows  product_brand.parquet: %d rows",
        len(prod_cat_df), len(prod_brand_df),
    )

    logger.info("Saving vocabulary mappings to %s ...", struct_dir)

    for name, mapping in (
        ("user2idx",     split.user2idx),
        ("item2idx",     split.item2idx),
        ("category2idx", category2idx),
        ("brand2idx",    brand2idx),
    ):
        path = os.path.join(struct_dir, f"{name}.json")
        with open(path, "w") as fh:
            json.dump({str(k): v for k, v in mapping.items()}, fh)
        logger.info("  %s: %d entries", name, len(mapping))

    compute_and_save_svd_factors(
        data_dir=data_dir,
        num_users=split.num_users,
        num_items=split.num_items,
        rank=256,
        n_iter=4,
    )


def save_node_counts(node_counts: dict[str, int], data_dir: str) -> None:
    path = os.path.join(data_dir, "node_counts.json")
    with open(path, "w") as fh:
        json.dump(node_counts, fh, indent=2)
    logger.info("node_counts.json written to %s: %s", path, node_counts)


def verify_artifacts(
    data_dir: str,
    struct_dir: str,
    node_counts: dict[str, int],
) -> None:
    logger.info("Running pre-training sanity check ...")

    def _npy_ei(prefix: str) -> torch.Tensor:
        src = np.load(os.path.join(data_dir, f"{prefix}_src.npy"))
        dst = np.load(os.path.join(data_dir, f"{prefix}_dst.npy"))
        return torch.from_numpy(np.stack([src, dst])).long().contiguous()

    view_ei     = _npy_ei("view_train")
    cart_ei     = _npy_ei("cart_train")
    purchase_ei = _npy_ei("purchase_train")

    pc = pd.read_parquet(os.path.join(struct_dir, "product_category.parquet"))
    pb = pd.read_parquet(os.path.join(struct_dir, "product_brand.parquet"))
    category_ei = torch.from_numpy(pc[["product_idx", "category_idx"]].values.T.copy()).long()
    brand_ei    = torch.from_numpy(pb[["product_idx", "brand_idx"]].values.T.copy()).long()

    hetero = HeteroData()
    for ntype, n in node_counts.items():
        hetero[ntype].x         = torch.arange(n, dtype=torch.long)
        hetero[ntype].num_nodes = n

    hetero[("user",     "view",         "product")].edge_index = view_ei
    hetero[("user",     "cart",         "product")].edge_index = cart_ei
    hetero[("user",     "purchase",     "product")].edge_index = purchase_ei
    hetero[("product",  "rev_view",     "user")].edge_index    = view_ei.flip(0).contiguous()
    hetero[("product",  "rev_cart",     "user")].edge_index    = cart_ei.flip(0).contiguous()
    hetero[("product",  "rev_purchase", "user")].edge_index    = purchase_ei.flip(0).contiguous()
    hetero[("product",  "belongs_to",   "category")].edge_index = category_ei.contiguous()
    hetero[("category", "contains",     "product")].edge_index  = category_ei.flip(0).contiguous()
    hetero[("product",  "producedBy",   "brand")].edge_index    = brand_ei.contiguous()
    hetero[("brand",    "brands",       "product")].edge_index  = brand_ei.flip(0).contiguous()

    train_triplets = torch.stack([
        purchase_ei[0],
        purchase_ei[1],
        torch.full((purchase_ei.size(1),), 2, dtype=torch.long),
    ], dim=1)

    test_users = np.load(os.path.join(data_dir, "test_user_idx.npy"))
    test_items = np.load(os.path.join(data_dir, "test_product_idx.npy"))
    ground_truth = {int(u): int(i) for u, i in zip(test_users, test_items)}

    sanity_check_heterodata(
        hetero,
        train_triplets,
        ground_truth,
        num_nodes_dict=node_counts,
        check_leakage=True,
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
    """Per-behavior randomized SVD of the GCN-normalized binary user×item
    adjacency. Writes data_dir/svd_factors.pt.

    Convention (matches src/core/contracts.SVDFactors):
        A_norm_k  ≈  US_k @ VS_k^T
        US_k = U_k * sqrt(S_k)   (num_users, rank)
        VS_k = V_k * sqrt(S_k)   (num_items, rank)

    Reads {beh}_train_{src,dst}.npy already on disk. Uses sklearn's
    randomized_svd — never densifies A.
    """
    import scipy.sparse as sp
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
            "  [%s] adj shape=%s nnz=%d  → randomized_svd",
            beh, A_norm.shape, A_norm.nnz,
        )

        U, S, Vt = randomized_svd(
            A_norm, n_components=rank, n_iter=n_iter, random_state=seed,
        )
        sqrt_S = np.sqrt(S).astype(np.float32)
        US = (U * sqrt_S).astype(np.float32)
        VS = (Vt.T * sqrt_S).astype(np.float32)

        assert US.shape == (num_users, rank), (beh, US.shape)
        assert VS.shape == (num_items, rank), (beh, VS.shape)

        US_dict[beh] = torch.from_numpy(US)
        VS_dict[beh] = torch.from_numpy(VS)

        del A, A_norm, U, S, Vt, US, VS, sqrt_S, deg_u, deg_v, d_u_inv, d_v_inv

    svd = SVDFactors(US=US_dict, VS=VS_dict)
    svd.validate()

    out_path = os.path.join(data_dir, "svd_factors.pt")
    torch.save(svd, out_path)
    size_mb = os.path.getsize(out_path) / (1024 ** 2)
    logger.info("SVD factors saved: %s  (%.1f MB)", out_path, size_mb)
