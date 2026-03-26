import os
import gc
import json
import shutil
import numpy as np
import scipy.sparse as sp
import pyarrow.parquet as pq
from pyspark.sql import DataFrame

from .spark_utils import log_step, get_dir_size_gb

# PARQUET
@log_step("LOAD: Save cleaned Parquet")
def save_cleaned_parquet(df: DataFrame, output_dir: str,
                         num_partitions: int = 64) -> str:
    path = os.path.join(output_dir, "cleaned.parquet")
    df.repartition(num_partitions).write.mode("overwrite").parquet(path)
    size_gb = get_dir_size_gb(path)
    print(f"Path: {path}")
    print(f"Size: {size_gb:.2f} GB")
    return path

def save_node_mapping(mapping_df: DataFrame, mappings_dir: str,
                      name: str, save_csv: bool = True):
    parquet_path = os.path.join(mappings_dir, f"{name}.parquet")
    mapping_df.write.mode("overwrite").parquet(parquet_path)

    if save_csv:
        n = mapping_df.count()
        if n < 1000000:
            csv_path = os.path.join(mappings_dir, f"{name}_csv")
            mapping_df.coalesce(1).write.mode("overwrite").option("header", "true").csv(csv_path)

    print(f"Saved {name} -> {parquet_path}")

@log_step("LOAD: Save node summary")
def save_node_summary(stats_dir: str, **counts) -> dict:
    counts["total_nodes"] = (
        counts.get("num_users", 0) + counts.get("num_products", 0) +
        counts.get("num_categories", 0) + counts.get("num_brands", 0)
    )
    path = os.path.join(stats_dir, "node_summary.json")
    with open(path, "w") as f:
        json.dump(counts, f, indent=2)

    print(f"Node summary:")
    for k, v in counts.items():
        print(f"{k:>35s}: {v:>12,}")
    return counts


def _parquet_to_numpy(parquet_path: str, columns: list) -> dict:
    table = pq.read_table(parquet_path, columns=columns)
    result = {}
    for col in columns:
        result[col] = table.column(col).to_numpy().astype(np.int64)
    del table
    gc.collect()
    return result


def spark_df_to_numpy_via_parquet(spark_df: DataFrame, tmp_dir: str,
                                  prefix: str, columns: list) -> dict:
    tmp_path = os.path.join(tmp_dir, f"_tmp_{prefix}")
    spark_df.select(*columns).write.mode("overwrite").parquet(tmp_path)

    arrays = _parquet_to_numpy(tmp_path, columns)

    shutil.rmtree(tmp_path, ignore_errors=True)
    return arrays


def edges_to_numpy_safe(spark_df: DataFrame, tmp_dir: str, prefix: str):
    columns = ["user_idx", "product_idx", "unix_ts"]
    arrays = spark_df_to_numpy_via_parquet(spark_df, tmp_dir, prefix, columns)
    return arrays["user_idx"], arrays["product_idx"], arrays["unix_ts"]


def save_edge_arrays(src: np.ndarray, dst: np.ndarray,
                     ts: np.ndarray, edge_dir: str, prefix: str):
    np.save(os.path.join(edge_dir, f"{prefix}_src.npy"), src)
    np.save(os.path.join(edge_dir, f"{prefix}_dst.npy"), dst)
    np.save(os.path.join(edge_dir, f"{prefix}_ts.npy"), ts)


def save_structural_edge_arrays(spark_df: DataFrame, edge_dir: str,
                                relation: str):
    pdf = spark_df.toPandas()
    src_col = pdf.columns[0]
    dst_col = pdf.columns[1]

    np.save(os.path.join(edge_dir, f"{relation}_src.npy"),
            pdf[src_col].values.astype(np.int64))
    np.save(os.path.join(edge_dir, f"{relation}_dst.npy"),
            pdf[dst_col].values.astype(np.int64))
    print(f"  Saved {relation}: {len(pdf):,} edges")

def build_and_save_sparse_adj(src: np.ndarray, dst: np.ndarray,
                              num_src: int, num_dst: int,
                              graph_dir: str, name: str) -> sp.csr_matrix:
    data = np.ones(len(src), dtype=np.float32)
    adj = sp.csr_matrix((data, (src, dst)), shape=(num_src, num_dst))
    adj.data = np.ones_like(adj.data)
    adj.eliminate_zeros()

    sp.save_npz(os.path.join(graph_dir, f"{name}.npz"), adj)
    print(f"  {name}: shape={adj.shape}, nnz={adj.nnz:,}")
    return adj

@log_step("LOAD: Save graph metadata")
def save_graph_meta(graph_dir: str, node_summary: dict,
                    edge_counts: dict) -> dict:
    total_edges = sum(edge_counts.values())
    meta = {
        "node_counts": {
            "user": node_summary.get("num_users", 0),
            "product": node_summary.get("num_products", 0),
            "category": node_summary.get("num_categories", 0),
            "brand": node_summary.get("num_brands", 0),
            "total": node_summary.get("total_nodes", 0),
        },
        "edge_counts": edge_counts,
        "total_edges": total_edges,
    }

    path = os.path.join(graph_dir, "graph_meta.json")
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Total nodes: {meta['node_counts']['total']:,}")
    print(f"Total edges: {total_edges:,}")
    for etype, cnt in edge_counts.items():
        print(f"    {etype:>20s}: {cnt:>12,}")

    return meta

@log_step("VALIDATE: Check graph integrity")
def validate_graph(edge_dir: str, graph_dir: str,
                   node_summary: dict, cfg: dict):
    n_u = node_summary["num_users"]
    n_p = node_summary["num_products"]
    n_c = node_summary["num_categories"]
    n_b = node_summary["num_brands"]

    errors = []

    for beh in cfg["graph"]["behavior_edge_types"]:
        for split in ["train", "val", "test"]:
            prefix = f"{beh}_{split}"
            src_path = os.path.join(edge_dir, f"{prefix}_src.npy")
            if not os.path.exists(src_path):
                errors.append(f"MISSING: {prefix}_src.npy")
                continue
            src = np.load(src_path)
            dst = np.load(os.path.join(edge_dir, f"{prefix}_dst.npy"))

            if len(src) != len(dst):
                errors.append(f"{prefix}: src/dst length mismatch")
            if len(src) > 0:
                if src.min() < 0:
                    errors.append(f"{prefix}: negative src index")
                if src.max() >= n_u:
                    errors.append(f"{prefix}: src index >= num_users")
                if dst.min() < 0:
                    errors.append(f"{prefix}: negative dst index")
                if dst.max() >= n_p:
                    errors.append(f"{prefix}: dst index >= num_products")

            print(f"  {prefix:>20s}: {len(src):>10,} edges — OK")

    for rel, max_s, max_d in [
        ("belongsTo", n_p, n_c),
        ("producedBy", n_p, n_b),
    ]:
        src = np.load(os.path.join(edge_dir, f"{rel}_src.npy"))
        dst = np.load(os.path.join(edge_dir, f"{rel}_dst.npy"))
        if src.max() >= max_s:
            errors.append(f"{rel}: src exceeds limit")
        if dst.max() >= max_d:
            errors.append(f"{rel}: dst exceeds limit")
        print(f"{rel:>20s}: {len(src):>10,} edges - OK")

    if errors:
        print(f"ERRORS FOUND:")
        for e in errors:
            print(f"{e}")
        raise ValueError(f"Graph validation failed with {len(errors)} errors")
    else:
        print(f"All checks PASSED")