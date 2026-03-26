set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

if [ -f ~/rees46_env/bin/activate ]; then
    source ~/rees46_env/bin/activate
fi

echo "# REES46 HETEROGENEOUS GRAPH DATA PIPELINE #"
echo "# Phase 1: Raw CSV → Graph (NumPy + Sparse) #"
echo "Project root: $PROJECT_ROOT"
echo "Start time: $(date)"
echo "Python: $(python3 --version)"

python3 -u << 'PIPELINE_EOF'
import sys, os, time, gc, shutil
import numpy as np
import json
import pyarrow.parquet as pq
from pyspark.sql import functions as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(".")), "src"))
sys.path.insert(0, "src")

from data_pipeline.spark_utils import load_config, create_spark_session, ensure_dirs
from data_pipeline.extract import load_raw_csv, load_cleaned_parquet, load_node_mapping
from data_pipeline.transform import (
    clean_raw_data, filter_cold_start, print_data_profile,
    build_node_mapping, build_product_category, build_product_brand,
    build_behavior_edges, temporal_split, leave_one_out_split,
    select_seed_users, expand_subgraph,
)
from data_pipeline.load import (
    save_cleaned_parquet, save_node_mapping, save_node_summary,
    edges_to_numpy_safe, save_edge_arrays, save_structural_edge_arrays,
    build_and_save_sparse_adj, save_graph_meta, validate_graph,
)

pipeline_start = time.time()

# ---- Load config ----
cfg = load_config()
paths = cfg["paths"]
ensure_dirs(cfg)

tmp_dir = os.path.join(paths["output_dir"], "_tmp")
os.makedirs(tmp_dir, exist_ok=True)

# ---- Create Spark ----
spark = create_spark_session(cfg, "FullPipeline")

try:
    # ================================================================
    # STEP 1: Load & Clean
    # ================================================================
    print("# STEP 1: Load & Clean Raw Data")
    df_raw = load_raw_csv(spark, paths["raw_csv_pattern"])
    df_clean = clean_raw_data(df_raw, cfg)
    df_filtered = filter_cold_start(df_clean, cfg)
    print_data_profile(df_filtered)
    save_cleaned_parquet(df_filtered, paths["output_dir"],
                         cfg["spark"]["shuffle_partitions"])

    del df_raw, df_clean, df_filtered
    spark.catalog.clearCache()
    gc.collect()

    # ================================================================
    # STEP 2: Build Node Mappings
    # ================================================================
    print("#  STEP 2: Build Node ID Mappings")

    df = load_cleaned_parquet(spark, paths["output_dir"])
    df.cache()

    user_map = build_node_mapping(df, "user_id", "user2idx")
    save_node_mapping(user_map, paths["node_mappings_dir"], "user2idx")

    product_map = build_node_mapping(df, "product_id", "product2idx")
    save_node_mapping(product_map, paths["node_mappings_dir"], "product2idx")

    category_map = build_node_mapping(df, "category", "category2idx")
    save_node_mapping(category_map, paths["node_mappings_dir"], "category2idx")

    brand_map = build_node_mapping(df, "brand_clean", "brand2idx")
    save_node_mapping(brand_map, paths["node_mappings_dir"], "brand2idx")

    pc_df = build_product_category(df, product_map, category_map)
    save_structural_edge_arrays(pc_df, paths["edge_lists_dir"], "belongsTo")
    pc_df.write.mode("overwrite").parquet(
        os.path.join(paths["node_mappings_dir"], "product_category.parquet"))

    pb_df = build_product_brand(df, product_map, brand_map)
    save_structural_edge_arrays(pb_df, paths["edge_lists_dir"], "producedBy")
    pb_df.write.mode("overwrite").parquet(
        os.path.join(paths["node_mappings_dir"], "product_brand.parquet"))

    node_summary = save_node_summary(
        paths["stats_dir"],
        num_users=user_map.count(),
        num_products=product_map.count(),
        num_categories=category_map.count(),
        num_brands=brand_map.count(),
        num_product_category_links=pc_df.count(),
        num_product_brand_links=pb_df.count(),
    )

    del pc_df, pb_df
    gc.collect()

    # ================================================================
    # STEP 3: Build Edge Lists + Temporal Split
    # ================================================================
    print("#  STEP 3: Edge Lists + Temporal Split + Sparse Adjacency")

    edge_counts = {}

    for behavior in cfg["graph"]["behavior_edge_types"]:
        edges_df = build_behavior_edges(df, behavior, user_map, product_map)
        edges_df.cache()

        train_df, val_df, test_df = temporal_split(edges_df, behavior, cfg)

        for split_name, split_df in [("train", train_df),
                                      ("val", val_df),
                                      ("test", test_df)]:
            prefix = f"{behavior}_{split_name}"
            src, dst, ts = edges_to_numpy_safe(
                split_df, tmp_dir, prefix)
            save_edge_arrays(src, dst, ts, paths["edge_lists_dir"], prefix)
            edge_counts[prefix] = int(len(src))
            print(f"    Saved {prefix}: {len(src):,}")

            if split_name == "train":
                build_and_save_sparse_adj(
                    src, dst,
                    node_summary["num_users"],
                    node_summary["num_products"],
                    paths["graph_dir"],
                    f"adj_{behavior}_train"
                )

            del src, dst, ts
            gc.collect()

        edges_df.unpersist()

    for rel in cfg["graph"]["structural_edge_types"]:
        src = np.load(os.path.join(paths["edge_lists_dir"], f"{rel}_src.npy"))
        edge_counts[rel] = int(len(src))
        del src

    save_graph_meta(paths["graph_dir"], node_summary, edge_counts)

    # ================================================================
    # CLEANUP between temporal and LOO
    # ================================================================
    print("#  Cleaning Spark caches before LOO split...")
    df.unpersist()
    spark.catalog.clearCache()
    gc.collect()

    # ================================================================
    # STEP 3b: Leave-One-Out Split (Protocol A)
    # ================================================================
    print("#  STEP 3b: Leave-One-Out Split (Protocol A)")

    loo_dir = os.path.join(paths["output_dir"], "loo")
    os.makedirs(loo_dir, exist_ok=True)

    df = load_cleaned_parquet(spark, paths["output_dir"])
    user_map = load_node_mapping(spark, paths["node_mappings_dir"], "user2idx")
    product_map = load_node_mapping(spark, paths["node_mappings_dir"], "product2idx")

    loo = leave_one_out_split(df, cfg, user_map, product_map, loo_dir)

    spark.catalog.clearCache()
    gc.collect()

    # --- Save val/test pairs via pyarrow ---
    val_table = pq.read_table(loo["val_parquet_path"],
                              columns=["user_idx", "product_idx"])
    np.save(os.path.join(loo_dir, "val_user_idx.npy"),
            val_table.column("user_idx").to_numpy().astype(np.int64))
    np.save(os.path.join(loo_dir, "val_product_idx.npy"),
            val_table.column("product_idx").to_numpy().astype(np.int64))
    n_val = len(val_table)
    print(f"    Saved LOO val pairs: {n_val:,}")
    del val_table
    gc.collect()

    test_table = pq.read_table(loo["test_parquet_path"],
                               columns=["user_idx", "product_idx"])
    np.save(os.path.join(loo_dir, "test_user_idx.npy"),
            test_table.column("user_idx").to_numpy().astype(np.int64))
    np.save(os.path.join(loo_dir, "test_product_idx.npy"),
            test_table.column("product_idx").to_numpy().astype(np.int64))
    n_test = len(test_table)
    print(f"    Saved LOO test pairs: {n_test:,}")
    del test_table
    gc.collect()

    # --- Save LOO train adjacency per behavior ---
    train_parquet_path = loo["train_parquet_path"]

    for beh in cfg["graph"]["behavior_edge_types"]:
        beh_df = (spark.read.parquet(train_parquet_path)
                  .filter(F.col("event_type") == beh)
                  .select("user_idx", "product_idx", "unix_ts"))

        prefix = f"loo_{beh}_train"
        src, dst, ts = edges_to_numpy_safe(beh_df, tmp_dir, prefix)

        np.save(os.path.join(loo_dir, f"{prefix}_src.npy"), src)
        np.save(os.path.join(loo_dir, f"{prefix}_dst.npy"), dst)
        np.save(os.path.join(loo_dir, f"{prefix}_ts.npy"), ts)

        build_and_save_sparse_adj(
            src, dst,
            node_summary["num_users"],
            node_summary["num_products"],
            loo_dir,
            f"adj_loo_{beh}_train"
        )
        print(f"    Saved {prefix}: {len(src):,} edges")

        del src, dst, ts
        gc.collect()

    # --- Save LOO metadata ---
    loo_meta = {
        "protocol": "leave_one_out",
        "description": "Last purchase=test, second-last=val, rest=train. "
                       "Auxiliary behaviors (view,cart) fully in train.",
        "eval_users": loo["eval_user_count"],
        "val_pairs": n_val,
        "test_pairs": n_test,
        "num_negative_samples": cfg.get("evaluation", {}).get(
            "num_negative_samples", 99),
    }
    with open(os.path.join(loo_dir, "loo_meta.json"), "w") as f:
        json.dump(loo_meta, f, indent=2)
    print(f"    LOO metadata saved to {loo_dir}/loo_meta.json")

    # --- Cleanup temp parquets ---
    for p in [loo["val_parquet_path"], loo["test_parquet_path"],
              loo["train_parquet_path"]]:
        shutil.rmtree(p, ignore_errors=True)
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # ================================================================
    # STEP 4: Validate
    # ================================================================
    print("#  STEP 4: Validate Graph Integrity")

    validate_graph(paths["edge_lists_dir"], paths["graph_dir"],
                   node_summary, cfg)

    # ================================================================
    # STEP 5: Create REES46-Small (optional)
    # ================================================================
    if cfg["sampling"]["create_small_version"]:
        print("# STEP 5: Create REES46-Small Subgraph")

        spark.catalog.clearCache()
        gc.collect()

        df = load_cleaned_parquet(spark, paths["output_dir"])

        seed_users = select_seed_users(
            df, cfg["sampling"]["small_num_users"])
        sub_df = expand_subgraph(df, seed_users)
        print_data_profile(sub_df)

        small_dir = paths["small_dir"]
        sub_df.write.mode("overwrite").parquet(
            os.path.join(small_dir, "cleaned.parquet"))
        print(f"Saved REES46-Small -> {small_dir}/cleaned.parquet")

    # DONE
    total_time = time.time() - pipeline_start
    print("\n" + "#"*60)
    print(f"#  PIPELINE COMPLETE — Total: {total_time/60:.1f} minutes")
    print("#" + "="*59)
    print(f"#  Output: {paths['output_dir']}")
    print(f"#    graph/          — sparse adjacency (.npz)")
    print(f"#    edge_lists/     — edge arrays (.npy)")
    print(f"#    node_mappings/  — ID mappings (.parquet)")
    print(f"#    statistics/     — summaries (.json)")
    print(f"#    loo/            — leave-one-out split (.npy)")
    print(f"#  Next: Phase 2 — Implement baselines (CRGCN)")
    print("#" + "="*59)

finally:
    spark.stop()

PIPELINE_EOF

echo "Pipeline finished at: $(date)"