from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys

import pandas as pd
from pyspark.sql import functions as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_pipeline.spark_utils import create_spark_session, load_config
from src.data_pipeline.extract import load_raw_csvs, clean
from src.data_pipeline.transform import (
    filter_by_train_only_counts,
    temporal_split_purchases,
    map_auxiliary,
    map_all_train_events,
    build_structural_edges,
    build_train_mask,
)
from src.data_pipeline.load import save_artifacts, save_node_counts, verify_artifacts
from src.data_pipeline.manifest import (
    write_split_manifest,
    write_artifacts_manifest,
    write_data_card,
    write_baseline_contract,
)

logger = logging.getLogger(__name__)

_VALID_BEHAVIORS = ("view", "cart", "purchase")


def _parse_args() -> argparse.Namespace:
    _config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config", "spark_config.yaml",
    )
    _cfg     = load_config(_config_path)
    _filt    = _cfg.get("filter",    {})
    _split   = _cfg.get("split",     {})
    _proto   = _cfg.get("protocol",  {})

    p = argparse.ArgumentParser(
        description="Prepare REES46 data for BPATMP++ (warm-new-purchase full-ranking).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--csv-glob",      required=True)
    p.add_argument("--spark-config",  default=_config_path)
    p.add_argument("--data-dir",      default="data/processed/temporal")
    p.add_argument("--struct-dir",    default="data/processed/temporal/node_mappings")
    p.add_argument("--graph-dir",     default="data/processed/temporal/graph")
    p.add_argument("--stats-dir",     default="data/processed/temporal/statistics")
    p.add_argument("--target-behavior", default=str(_proto.get("target_behavior", "purchase")), choices=list(_VALID_BEHAVIORS))
    p.add_argument("--min-user-purchases", type=int,
                   default=int(_filt.get("min_train_user_purchases", _filt.get("min_user_interactions", 5))))
    p.add_argument("--min-item-purchases", type=int,
                   default=int(_filt.get("min_train_item_purchases", _filt.get("min_item_interactions", 5))))
    p.add_argument("--filter-rounds", type=int, default=int(_filt.get("iterative_filter_rounds", 3)))
    p.add_argument("--train-end",     default=str(_split.get("train_end", "2020-02-29")))
    p.add_argument("--val-end",       default=str(_split.get("val_end",   "2020-03-31")))
    p.add_argument("--log-level",     default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def _behavior_counts(spark_df, label: str) -> dict[str, int]:
    rows = spark_df.groupBy("event_type").count().collect()
    return {label: 0, **{r["event_type"]: int(r["count"]) for r in rows}}


def main() -> None:
    args = _parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.info("=" * 60)
    logger.info("REES46 BPATMP++ Data Preparation")
    logger.info("=" * 60)

    cfg = load_config(args.spark_config)
    proto_cfg = cfg.get("protocol", {})
    eval_cfg  = cfg.get("evaluation", {})
    artifacts_cfg = cfg.get("artifacts", {})
    filter_cfg = cfg.get("filter", {})
    behavior_cfg = cfg.get("behavior", {})

    transductive_item_vocab = bool(proto_cfg.get("transductive_item_vocab", False))
    transductive_metadata   = bool(proto_cfg.get("allow_transductive_item_metadata", False))
    metadata_source = "all_rows" if transductive_metadata else "train_only"
    drop_repeated   = bool(proto_cfg.get("drop_repeated_train_purchases_from_eval", True))
    protocol_name   = str(proto_cfg.get("name", "warm_new_purchase_full_ranking"))
    primary_mask_behaviors = tuple(eval_cfg.get("mask_behaviors_primary", ["purchase"]))
    seen_all_mask_behaviors = tuple(eval_cfg.get("mask_behaviors_seen_all", ["view", "cart", "purchase"]))
    behavior_ids = {b: int(i) for b, i in behavior_cfg.get("ids", {"view": 0, "cart": 1, "purchase": 2}).items()}
    unknown_brand    = str(filter_cfg.get("unknown_brand", "__UNKNOWN_BRAND__"))
    unknown_category = str(filter_cfg.get("unknown_category", "__UNKNOWN_CATEGORY__"))

    train_cutoff_ts = int(
        (pd.Timestamp(args.train_end, tz="UTC") + pd.Timedelta(days=1)).timestamp()
    )
    val_cutoff_ts = int(
        (pd.Timestamp(args.val_end, tz="UTC") + pd.Timedelta(days=1)).timestamp()
    )

    spark = create_spark_session(cfg)
    node_counts: dict[str, int] = {}
    artifacts_summary = {}

    try:
        clean_spark = clean(load_raw_csvs(spark, args.csv_glob), cfg)

        if args.min_user_purchases > 1 or args.min_item_purchases > 1:
            clean_spark = filter_by_train_only_counts(
                clean_spark,
                target_behavior=args.target_behavior,
                train_cutoff_ts=train_cutoff_ts,
                min_user_purchases=args.min_user_purchases,
                min_item_purchases=args.min_item_purchases,
                rounds=args.filter_rounds,
            )

        clean_spark = clean_spark.cache()

        split = temporal_split_purchases(
            clean_spark,
            args.target_behavior,
            args.train_end,
            args.val_end,
            transductive_item_vocab=transductive_item_vocab,
            drop_repeated_train_purchases_from_eval=drop_repeated,
            protocol_name=protocol_name,
        )

        aux_spark = map_auxiliary(
            clean_spark, spark, split, args.target_behavior,
            global_cutoff=split.train_end_ts,
        ).cache()

        train_events_spark = map_all_train_events(
            clean_spark, spark, split,
            global_cutoff=split.train_end_ts,
            behavior_ids=behavior_ids,
        )

        prod_cat_df, prod_brand_df, category2idx, brand2idx, meta_stats = build_structural_edges(
            clean_spark, spark, split.item2idx,
            train_cutoff_ts=split.train_end_ts,
            metadata_source=metadata_source,
            unknown_category=unknown_category,
            unknown_brand=unknown_brand,
        )
        item_metadata_df = meta_stats.pop("item_metadata")
        clean_spark.unpersist()

        eval_user_ids = sorted({
            *split.val["user_idx"].astype(int).tolist(),
            *split.test["user_idx"].astype(int).tolist(),
        })

        train_mask_primary = build_train_mask(
            split.train, aux_spark, eval_user_ids, spark,
            mask_behaviors=primary_mask_behaviors,
        )
        train_mask_seen_all = build_train_mask(
            split.train, aux_spark, eval_user_ids, spark,
            mask_behaviors=seen_all_mask_behaviors,
        )

        artifacts_summary = save_artifacts(
            split, aux_spark,
            train_mask_primary, train_mask_seen_all,
            prod_cat_df, prod_brand_df,
            category2idx, brand2idx,
            behavior_ids, item_metadata_df,
            args.data_dir, args.struct_dir, args.graph_dir,
            train_events_spark=train_events_spark,
        )

        node_counts = {
            "user":     split.num_users,
            "product":  split.num_items,
            "category": len(category2idx),
            "brand":    len(brand2idx),
        }
        save_node_counts(node_counts, args.data_dir)

        os.makedirs(args.stats_dir, exist_ok=True)
        if artifacts_cfg.get("save_manifest", True):
            write_split_manifest(
                os.path.join(args.stats_dir, "split_manifest.json"),
                behaviors=list(_VALID_BEHAVIORS),
                target_behavior=args.target_behavior,
                protocol_name=protocol_name,
                timezone_str=str(proto_cfg.get("split_timezone", "UTC")),
                train_end=args.train_end,
                val_end=args.val_end,
                train_cutoff_ts=split.train_end_ts,
                val_cutoff_ts=split.val_end_ts,
                candidate_set=str(proto_cfg.get("candidate_set", "warm_train_items")),
                ground_truth=str(proto_cfg.get("ground_truth", "all_purchases")),
                primary_mask_behaviors=list(primary_mask_behaviors),
                transductive_item_vocab=transductive_item_vocab,
                allow_transductive_item_metadata=transductive_metadata,
                extra={"stats": split.stats},
            )

        if artifacts_cfg.get("save_data_card", True):
            train_behavior_counts = {
                "view":     int(aux_spark.filter(F.col("event_type") == "view").count()),
                "cart":     int(aux_spark.filter(F.col("event_type") == "cart").count()),
                "purchase": int(len(split.train)),
            }
            val_positives  = len(split.val)
            test_positives = len(split.test)

            write_data_card(
                os.path.join(args.stats_dir, "data_card.json"),
                os.path.join(args.stats_dir, "data_card.md"),
                raw_rows=None,
                cleaned_rows=None,
                behavior_counts_overall=train_behavior_counts,
                behavior_counts_by_split={
                    "train": train_behavior_counts,
                    "val":   {"purchase": val_positives},
                    "test":  {"purchase": test_positives},
                },
                num_users=split.num_users,
                num_items=split.num_items,
                num_categories=len(category2idx),
                num_brands=len(brand2idx),
                train_purchase_pairs=len(split.train),
                val_positives=val_positives,
                test_positives=test_positives,
                val_users=int(split.val["user_idx"].nunique()) if not split.val.empty else 0,
                test_users=int(split.test["user_idx"].nunique()) if not split.test.empty else 0,
                cold_users_dropped={
                    "val":  int(split.stats.get("val_cold_users_dropped", 0)),
                    "test": int(split.stats.get("test_cold_users_dropped", 0)),
                },
                cold_items_dropped={
                    "val":  int(split.stats.get("val_cold_items_dropped", 0)),
                    "test": int(split.stats.get("test_cold_items_dropped", 0)),
                },
                repeated_purchase_dropped={
                    "val":  int(split.stats.get("val_repeated_train_purchase_dropped", 0)),
                    "test": int(split.stats.get("test_repeated_train_purchase_dropped", 0)),
                },
                metadata_missingness=meta_stats,
                edge_artifact_counts=train_behavior_counts,
                mask_stats={
                    "primary":  {
                        "users": len(train_mask_primary),
                        "avg_seen": float(sum(len(v) for v in train_mask_primary.values()) / max(len(train_mask_primary), 1)),
                    },
                    "seen_all": {
                        "users": len(train_mask_seen_all),
                        "avg_seen": float(sum(len(v) for v in train_mask_seen_all.values()) / max(len(train_mask_seen_all), 1)),
                    },
                },
                leakage_check={"status": "see verify_artifacts() output"},
                config={
                    "protocol":    proto_cfg,
                    "filter":      filter_cfg,
                    "evaluation":  eval_cfg,
                    "split":       cfg.get("split", {}),
                    "behavior":    behavior_cfg,
                },
                train_cutoff_ts=split.train_end_ts,
                val_cutoff_ts=split.val_end_ts,
                train_end=args.train_end,
                val_end=args.val_end,
                protocol_name=protocol_name,
            )

        if eval_cfg.get("save_baseline_contract", True):
            write_baseline_contract(
                os.path.join(args.stats_dir, "baseline_contract.md"),
                protocol_name=protocol_name,
                primary_metric=str(eval_cfg.get("primary_metric", "NDCG@20")),
                metrics=list(eval_cfg.get("metrics", [])),
                primary_mask_behaviors=list(primary_mask_behaviors),
                candidate_set=str(proto_cfg.get("candidate_set", "warm_train_items")),
                ground_truth=str(proto_cfg.get("ground_truth", "all_purchases")),
            )

        aux_spark.unpersist()

    finally:
        spark.stop()
        logger.info("SparkSession stopped.")
        checkpoint_dir = cfg["spark"].get("checkpoint_dir", "/tmp/spark_checkpoints")
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)
            logger.info("Cleaned up checkpoint dir: %s", checkpoint_dir)

    verify_artifacts(
        args.data_dir, args.struct_dir, args.graph_dir, node_counts,
        train_cutoff_ts=train_cutoff_ts,
        val_cutoff_ts=val_cutoff_ts,
    )

    if artifacts_cfg.get("save_manifest", True):
        write_artifacts_manifest(
            os.path.join(args.stats_dir, "artifacts_manifest.json"),
            roots={
                "edge_lists":    args.data_dir,
                "graph":         args.graph_dir,
                "node_mappings": args.struct_dir,
            },
        )

    logger.info("=" * 60)
    logger.info("Data preparation complete.")
    logger.info("Artifacts: data_dir=%s graph_dir=%s struct_dir=%s stats_dir=%s",
                args.data_dir, args.graph_dir, args.struct_dir, args.stats_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
