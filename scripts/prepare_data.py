from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_pipeline.spark_utils import create_spark_session, load_config
from src.data_pipeline.extract import load_raw_csvs, clean
from src.data_pipeline.transform import (
    filter_cold_items,
    temporal_split_purchases,
    map_auxiliary,
    build_structural_edges,
    build_train_mask,
)
from src.data_pipeline.load import save_artifacts, save_node_counts, verify_artifacts

logger = logging.getLogger(__name__)

_VALID_BEHAVIORS = ("view", "cart", "purchase")


def _parse_args() -> argparse.Namespace:
    _config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config", "spark_config.yaml",
    )
    _cfg   = load_config(_config_path)
    _filt  = _cfg.get("filter", {})
    _split = _cfg.get("split",  {})

    _default_min_user      = int(_filt.get("min_user_interactions",   5))
    _default_min_item      = int(_filt.get("min_item_interactions",   5))
    _default_filter_rounds = int(_filt.get("iterative_filter_rounds", 3))
    _default_train_end     = str(_split.get("train_end", "2020-02-29"))
    _default_val_end       = str(_split.get("val_end",   "2020-03-31"))

    p = argparse.ArgumentParser(
        description="Prepare REES46 data for BAGNN training (Global Temporal Split).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--csv-glob",
        required=True,
        help='Glob pattern for raw REES46 CSV files, e.g. "data/raw/*.csv".',
    )
    p.add_argument(
        "--spark-config",
        default=_config_path,
        help="Path to spark_config.yaml (Spark tuning + filter/split defaults).",
    )
    p.add_argument(
        "--data-dir",
        default="data/processed/temporal",
        help=(
            "Output directory for edge arrays, test/val pairs, train_mask, and "
            "node_counts."
        ),
    )
    p.add_argument(
        "--struct-dir",
        default="data/processed/temporal/node_mappings",
        help=(
            "Output directory for structural parquets and vocabulary JSONs."
        ),
    )
    p.add_argument(
        "--target-behavior",
        default="purchase",
        choices=list(_VALID_BEHAVIORS),
        help="Behaviour used for the evaluation split.",
    )
    p.add_argument(
        "--min-interactions",
        type=int,
        default=_default_min_user,
        help=(
            "Minimum number of target-behaviour interactions per user. "
            f"Loaded from spark_config.yaml (currently {_default_min_user})."
        ),
    )
    p.add_argument(
        "--min-item-interactions",
        type=int,
        default=_default_min_item,
        help=(
            "Minimum number of target-behaviour interactions per item. "
            f"Loaded from spark_config.yaml (currently {_default_min_item})."
        ),
    )
    p.add_argument(
        "--filter-rounds",
        type=int,
        default=_default_filter_rounds,
        help=(
            "Iterative cold-start filter rounds. "
            f"Loaded from spark_config.yaml (currently {_default_filter_rounds})."
        ),
    )
    p.add_argument(
        "--train-end",
        default=_default_train_end,
        help=(
            "Last calendar day of training window (inclusive UTC). "
            f"Loaded from spark_config.yaml (currently {_default_train_end!r})."
        ),
    )
    p.add_argument(
        "--val-end",
        default=_default_val_end,
        help=(
            "Last calendar day of validation window (inclusive UTC). "
            f"Loaded from spark_config.yaml (currently {_default_val_end!r})."
        ),
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.info("=" * 60)
    logger.info("REES46 BAGNN Data Preparation (Global Temporal Split)")
    logger.info("=" * 60)
    logger.info("spark-config         : %s", args.spark_config)
    logger.info("csv-glob             : %s", args.csv_glob)
    logger.info("data-dir             : %s", args.data_dir)
    logger.info("struct-dir           : %s", args.struct_dir)
    logger.info("target-behavior      : %s", args.target_behavior)
    logger.info("min-interactions     : %d", args.min_interactions)
    logger.info("min-item-interactions: %d", args.min_item_interactions)
    logger.info("filter-rounds        : %d", args.filter_rounds)
    logger.info("train-end            : %s", args.train_end)
    logger.info("val-end              : %s", args.val_end)

    cfg = load_config(args.spark_config)
    spark = create_spark_session(cfg)

    try:
        clean_spark = clean(load_raw_csvs(spark, args.csv_glob), cfg)

        if args.min_item_interactions > 1:
            clean_spark = filter_cold_items(
                clean_spark,
                args.target_behavior,
                args.min_item_interactions,
                rounds=args.filter_rounds,
            )

        clean_spark = clean_spark.cache()

        split = temporal_split_purchases(
            clean_spark,
            args.target_behavior,
            args.min_interactions,
            args.train_end,
            args.val_end,
        )
        aux_spark = map_auxiliary(
            clean_spark, spark, split, args.target_behavior,
            global_cutoff=split.train_end_ts,
        )

        prod_cat_df, prod_brand_df, category2idx, brand2idx = build_structural_edges(
            clean_spark, spark, split.item2idx,
        )
        clean_spark.unpersist()

        train_mask = build_train_mask(split.train, aux_spark, split.test, spark)

        save_artifacts(
            split, aux_spark, train_mask,
            prod_cat_df, prod_brand_df, category2idx, brand2idx,
            args.data_dir, args.struct_dir,
        )

        node_counts = {
            "user":     split.num_users,
            "product":  split.num_items,
            "category": len(category2idx),
            "brand":    len(brand2idx),
        }
        save_node_counts(node_counts, args.data_dir)

    finally:
        spark.stop()
        logger.info("SparkSession stopped.")
        checkpoint_dir = cfg["spark"].get("checkpoint_dir", "/tmp/spark_checkpoints")
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)
            logger.info("Cleaned up checkpoint dir: %s", checkpoint_dir)

    verify_artifacts(args.data_dir, args.struct_dir, node_counts)

    logger.info("=" * 60)
    logger.info("Data preparation complete.")
    logger.info("Artifacts written to: %s", args.data_dir)
    logger.info(
        "Run:\n  python -m src.training.trainer --config config/training.yaml"
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
