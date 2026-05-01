from __future__ import annotations

"""Điều phối chuẩn bị dữ liệu theo thời gian.

Luồng (sau khi Spark khởi động):
1. load_raw_csvs -> clean
2. filter_by_train_only_counts — ngưỡng số lần mua user/item chỉ trong cửa sổ train
3. temporal_split_purchases -> SplitResult (vocab, purchase train/val/test)
4. map_auxiliary, map_all_train_events, build_structural_edges; bỏ cache DataFrame đã làm sạch
5. build_train_mask (chế độ chính + seen-all) cho user được đánh giá
6. save_artifacts, save_node_counts
Dừng SparkSession, sau đó:
7. verify_artifacts (kiểm tra .npy / parquet, không dùng Spark)
"""

import argparse
import logging
import os
import shutil
import sys

import pandas as pd

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

logger = logging.getLogger(__name__)

_VALID_BEHAVIORS = ("view", "cart", "purchase")


def _log_phase(step: int, total: int, title: str) -> None:
    logger.info("%s Giai đoạn %d/%d — %s %s", "=" * 12, step, total, title, "=" * 12)


def _parse_args() -> argparse.Namespace:
    _config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config", "spark_config.yaml",
    )
    _cfg = load_config(_config_path)
    _filt = _cfg.get("filter", {})
    _split = _cfg.get("split", {})
    _proto = _cfg.get("protocol", {})

    p = argparse.ArgumentParser(
        description="Chuẩn bị dữ liệu REES46 cho BPATMP (warm-start, dự đoán mua mới, xếp hạng đầy đủ).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--csv-glob", required=True)
    p.add_argument("--spark-config", default=_config_path)
    p.add_argument("--data-dir", default="data/processed/temporal")
    p.add_argument("--struct-dir", default="data/processed/temporal/node_mappings")
    p.add_argument("--graph-dir", default="data/processed/temporal/graph")
    p.add_argument(
        "--target-behavior",
        default=str(_proto.get("target_behavior", "purchase")),
        choices=list(_VALID_BEHAVIORS),
    )
    p.add_argument(
        "--min-user-purchases",
        type=int,
        default=int(
            _filt.get("min_train_user_purchases", _filt.get("min_user_interactions", 5)),
        ),
    )
    p.add_argument(
        "--min-item-purchases",
        type=int,
        default=int(
            _filt.get("min_train_item_purchases", _filt.get("min_item_interactions", 5)),
        ),
    )
    p.add_argument("--filter-rounds", type=int, default=int(_filt.get("iterative_filter_rounds", 3)))
    p.add_argument("--train-end", default=str(_split.get("train_end", "2020-02-29")))
    p.add_argument("--val-end", default=str(_split.get("val_end", "2020-03-31")))
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.info("=" * 60)
    logger.info("REES46 BPATMP — Chuẩn bị dữ liệu")
    logger.info("=" * 60)

    cfg = load_config(args.spark_config)
    proto_cfg = cfg.get("protocol", {})
    eval_cfg = cfg.get("evaluation", {})
    filter_cfg = cfg.get("filter", {})
    behavior_cfg = cfg.get("behavior", {})

    transductive_item_vocab = bool(proto_cfg.get("transductive_item_vocab", False))
    transductive_metadata = bool(proto_cfg.get("allow_transductive_item_metadata", False))
    metadata_source = "all_rows" if transductive_metadata else "train_only"
    drop_repeated = bool(proto_cfg.get("drop_repeated_train_purchases_from_eval", True))
    protocol_name = str(proto_cfg.get("name", "warm_new_purchase_full_ranking"))
    primary_mask_behaviors = tuple(eval_cfg.get("mask_behaviors_primary", ["purchase"]))
    seen_all_mask_behaviors = tuple(eval_cfg.get("mask_behaviors_seen_all", ["view", "cart", "purchase"]))
    behavior_ids = {b: int(i) for b, i in behavior_cfg.get("ids", {"view": 0, "cart": 1, "purchase": 2}).items()}
    unknown_brand = str(filter_cfg.get("unknown_brand", "__UNKNOWN_BRAND__"))
    unknown_category = str(filter_cfg.get("unknown_category", "__UNKNOWN_CATEGORY__"))

    train_cutoff_ts = int(
        (pd.Timestamp(args.train_end, tz="UTC") + pd.Timedelta(days=1)).timestamp()
    )
    val_cutoff_ts = int(
        (pd.Timestamp(args.val_end, tz="UTC") + pd.Timedelta(days=1)).timestamp()
    )

    spark = create_spark_session(cfg)
    node_counts: dict[str, int] = {}

    _total_phases = 7
    try:
        _log_phase(1, _total_phases, "Đọc & làm sạch CSV thô (Spark)")
        clean_spark = clean(load_raw_csvs(spark, args.csv_glob), cfg)

        if args.min_user_purchases > 1 or args.min_item_purchases > 1:
            _log_phase(2, _total_phases, "Lọc bậc theo cửa sổ train (user & item)")
            clean_spark = filter_by_train_only_counts(
                clean_spark,
                target_behavior=args.target_behavior,
                train_cutoff_ts=train_cutoff_ts,
                min_user_purchases=args.min_user_purchases,
                min_item_purchases=args.min_item_purchases,
                rounds=args.filter_rounds,
            )
        else:
            logger.info(
                "Giai đoạn 2/%d — bỏ qua (min user/item purchase ≤ 1).", _total_phases,
            )

        clean_spark = clean_spark.cache()

        _log_phase(3, _total_phases, "Chia tập theo thời gian (hành vi mục tiêu) -> vocab & các splits")
        split = temporal_split_purchases(
            clean_spark,
            args.target_behavior,
            args.train_end,
            args.val_end,
            transductive_item_vocab=transductive_item_vocab,
            drop_repeated_train_purchases_from_eval=drop_repeated,
            protocol_name=protocol_name,
        )

        _log_phase(4, _total_phases, "Cạnh phụ, nhật ký sự kiện train, metadata cấu trúc")
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
        del meta_stats
        clean_spark.unpersist()

        eval_user_ids = sorted({
            *split.val["user_idx"].astype(int).tolist(),
            *split.test["user_idx"].astype(int).tolist(),
        })

        _log_phase(5, _total_phases, "Mask xếp hạng đầy đủ (chính & seen-all)")
        train_mask_primary = build_train_mask(
            split.train, aux_spark, eval_user_ids, spark,
            mask_behaviors=primary_mask_behaviors,
        )
        train_mask_seen_all = build_train_mask(
            split.train, aux_spark, eval_user_ids, spark,
            mask_behaviors=seen_all_mask_behaviors,
        )

        _log_phase(6, _total_phases, "Lưu tensor, parquet, ánh xạ")
        save_artifacts(
            split, aux_spark,
            train_mask_primary, train_mask_seen_all,
            prod_cat_df, prod_brand_df,
            category2idx, brand2idx,
            behavior_ids, item_metadata_df,
            args.data_dir, args.struct_dir, args.graph_dir,
            train_events_spark=train_events_spark,
        )

        node_counts = {
            "user": split.num_users,
            "product": split.num_items,
            "category": len(category2idx),
            "brand": len(brand2idx),
        }
        save_node_counts(node_counts, args.data_dir)

        aux_spark.unpersist()

    finally:
        spark.stop()
        logger.info("Đã dừng SparkSession.")
        checkpoint_dir = cfg["spark"].get("checkpoint_dir", "/tmp/spark_checkpoints")
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)
            logger.info("Đã dọn thư mục checkpoint: %s", checkpoint_dir)

    _log_phase(7, _total_phases, "Kiểm tra artefact (sanity, không dùng Spark)")
    verify_artifacts(
        args.data_dir, args.struct_dir, args.graph_dir, node_counts,
        train_cutoff_ts=train_cutoff_ts,
        val_cutoff_ts=val_cutoff_ts,
    )

    logger.info("=" * 60)
    logger.info("Hoàn tất chuẩn bị dữ liệu.")
    logger.info(
        "Artefact: data_dir=%s graph_dir=%s struct_dir=%s",
        args.data_dir,
        args.graph_dir,
        args.struct_dir,
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
