from __future__ import annotations

import logging
from typing import Iterable, Literal

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from .splitter import DataSplitter, SplitResult

logger = logging.getLogger(__name__)

_VALID_BEHAVIORS = ("view", "cart", "purchase")
_DEFAULT_BEHAVIOR_IDS = {"view": 0, "cart": 1, "purchase": 2}


def _train_window(df: DataFrame, train_cutoff_ts: int) -> DataFrame:
    return df.filter(F.col("timestamp") < train_cutoff_ts)


def filter_by_train_only_counts(
    df: DataFrame,
    target_behavior: str,
    train_cutoff_ts: int,
    *,
    min_user_purchases: int,
    min_item_purchases: int,
    rounds: int = 3,
) -> DataFrame:
    """Iteratively keep users and items whose train-window target-behavior count
    meets the thresholds. Filtering decisions use only the train window so no
    val/test signal can change the train vocabulary.
    """
    if min_user_purchases <= 1 and min_item_purchases <= 1:
        return df

    for r in range(max(rounds, 1)):
        train_target = (
            df.filter(F.col("timestamp") < train_cutoff_ts)
              .filter(F.col("event_type") == target_behavior)
        )

        if min_item_purchases > 1:
            valid_items = (
                train_target.groupBy("product_id").count()
                .filter(F.col("count") >= min_item_purchases)
                .select("product_id")
            )
            df = df.join(F.broadcast(valid_items), on="product_id", how="inner")

        if min_user_purchases > 1:
            train_target = (
                df.filter(F.col("timestamp") < train_cutoff_ts)
                  .filter(F.col("event_type") == target_behavior)
            )
            valid_users = (
                train_target.groupBy("user_id").count()
                .filter(F.col("count") >= min_user_purchases)
                .select("user_id")
            )
            df = df.join(F.broadcast(valid_users), on="user_id", how="inner")

        df = df.checkpoint(eager=False)
        logger.info("filter_by_train_only_counts: round %d/%d done.", r + 1, rounds)

    return df


def temporal_split_purchases(
    df: DataFrame,
    target_behavior: str,
    train_end: str,
    val_end: str,
    *,
    transductive_item_vocab: bool = False,
    drop_repeated_train_purchases_from_eval: bool = True,
    protocol_name: str = "warm_new_purchase_full_ranking",
) -> SplitResult:
    """Pull the target-behavior rows to Pandas and run the date-based split.

    Train-only filtering must already be applied to ``df``.
    """
    purchase_spark = df.filter(F.col("event_type") == target_behavior).select(
        F.col("user_id"),
        F.col("product_id").alias("item_id"),
        F.col("timestamp"),
        F.col("event_type"),
    ).cache()
    purchase_df = purchase_spark.toPandas()
    purchase_spark.unpersist()

    logger.info(
        "Temporal — %s interactions: %d  (unique users: %d, items: %d)",
        target_behavior, len(purchase_df),
        purchase_df["user_id"].nunique(), purchase_df["item_id"].nunique(),
    )

    split = DataSplitter(purchase_df).temporal_split_by_dates(
        train_end=train_end, val_end=val_end,
        transductive_item_vocab=transductive_item_vocab,
        drop_repeated_train_purchases_from_eval=drop_repeated_train_purchases_from_eval,
        protocol_name=protocol_name,
    )
    logger.info("\n%s", split.summary())
    return split


def _user_item_maps(spark: SparkSession, split: SplitResult) -> tuple[DataFrame, DataFrame]:
    user_map = spark.createDataFrame(
        [(int(k), int(v)) for k, v in split.user2idx.items()],
        schema="user_id LONG, user_idx LONG",
    )
    item_map = spark.createDataFrame(
        [(int(k), int(v)) for k, v in split.item2idx.items()],
        schema="product_id LONG, item_idx LONG",
    )
    return user_map, item_map


def map_auxiliary(
    df: DataFrame,
    spark: SparkSession,
    split: SplitResult,
    target_behavior: str,
    *,
    global_cutoff: int,
) -> DataFrame:
    """Map non-target events through the train vocab. Drops rows whose
    timestamp is outside the train window or whose user/item is not in vocab.
    """
    user_map, item_map = _user_item_maps(spark, split)

    aux = (
        df.filter(F.col("event_type") != target_behavior)
        .join(F.broadcast(user_map), on="user_id",    how="inner")
        .join(F.broadcast(item_map), on="product_id", how="inner")
        .filter(F.col("timestamp") < global_cutoff)
    )

    cols = [
        F.col("user_idx").cast("long").alias("user_idx"),
        F.col("item_idx").cast("long").alias("item_idx"),
        F.col("timestamp").cast("long").alias("timestamp"),
        F.col("event_type"),
    ]
    if "user_session" in aux.columns:
        cols.append(F.col("user_session"))
    return aux.select(*cols)


def map_all_train_events(
    df: DataFrame,
    spark: SparkSession,
    split: SplitResult,
    *,
    global_cutoff: int,
    behavior_ids: dict[str, int] | None = None,
) -> DataFrame:
    """Build the full train-window event log (view + cart + purchase) mapped to
    train indices. Output schema is the contract for ``train_events.parquet``.
    """
    if behavior_ids is None:
        behavior_ids = _DEFAULT_BEHAVIOR_IDS

    user_map, item_map = _user_item_maps(spark, split)

    rows = (
        df.filter(F.col("event_type").isin(list(_VALID_BEHAVIORS)))
        .join(F.broadcast(user_map), on="user_id",    how="inner")
        .join(F.broadcast(item_map), on="product_id", how="inner")
        .filter(F.col("timestamp") < global_cutoff)
    )

    behavior_id_col = F.coalesce(*[
        F.when(F.col("event_type") == b, F.lit(int(behavior_ids[b])))
        for b in _VALID_BEHAVIORS
    ]).cast("int")

    cols = [
        F.col("user_idx").cast("long").alias("user_idx"),
        F.col("item_idx").cast("long").alias("item_idx"),
        F.col("event_type").alias("behavior"),
        behavior_id_col.alias("behavior_id"),
        F.col("timestamp").cast("long").alias("timestamp"),
    ]
    if "user_session" in rows.columns:
        cols.append(F.col("user_session"))

    return rows.select(*cols)


def build_structural_edges(
    df: DataFrame,
    spark: SparkSession,
    item2idx: dict,
    *,
    train_cutoff_ts: int | None = None,
    metadata_source: Literal["train_only", "all_rows"] = "train_only",
    unknown_category: str = "__UNKNOWN_CATEGORY__",
    unknown_brand: str = "__UNKNOWN_BRAND__",
) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict, dict]:
    """Compute mode category/brand per product. Primary protocol restricts the
    source rows to the train window so no future metadata leaks in.

    Returns: prod_cat_df, prod_brand_df, category2idx, brand2idx, meta_stats.
    """
    if metadata_source == "train_only":
        if train_cutoff_ts is None:
            raise ValueError("train_cutoff_ts is required when metadata_source='train_only'")
        df = df.filter(F.col("timestamp") < train_cutoff_ts)
    elif metadata_source != "all_rows":
        raise ValueError(f"Unknown metadata_source: {metadata_source!r}")

    item_map = spark.createDataFrame(
        [(int(k), int(v)) for k, v in item2idx.items()],
        schema="product_id LONG, product_idx LONG",
    )

    item_rows = (
        df.join(F.broadcast(item_map), on="product_id", how="inner")
        .select("product_id", "product_idx", "category", "brand")
        .cache()
    )

    w_mode = Window.partitionBy("product_id").orderBy(F.desc("cnt"))

    cat_mode = (
        item_rows
        .groupBy("product_id", "product_idx", "category")
        .count().withColumnRenamed("count", "cnt")
        .withColumn("rn", F.row_number().over(w_mode))
        .filter(F.col("rn") == 1)
        .select("product_idx", "category")
    )

    brand_mode = (
        item_rows
        .groupBy("product_id", "product_idx", "brand")
        .count().withColumnRenamed("count", "cnt")
        .withColumn("rn", F.row_number().over(w_mode))
        .filter(F.col("rn") == 1)
        .select("product_idx", "brand")
    )

    cat_pdf   = cat_mode.toPandas()
    brand_pdf = brand_mode.toPandas()
    item_rows.unpersist()

    seen_products = set(map(int, cat_pdf["product_idx"].tolist()) if not cat_pdf.empty else [])
    all_products  = set(int(v) for v in item2idx.values())
    missing_meta_products = all_products - seen_products

    if missing_meta_products:
        logger.info(
            "build_structural_edges: %d products lack metadata in source rows -> "
            "assigning unknown category/brand.",
            len(missing_meta_products),
        )
        if cat_pdf.empty:
            cat_pdf = pd.DataFrame(columns=["product_idx", "category"])
        if brand_pdf.empty:
            brand_pdf = pd.DataFrame(columns=["product_idx", "brand"])
        miss_rows_cat = pd.DataFrame({
            "product_idx": sorted(missing_meta_products),
            "category":    [unknown_category] * len(missing_meta_products),
        })
        miss_rows_brand = pd.DataFrame({
            "product_idx": sorted(missing_meta_products),
            "brand":       [unknown_brand] * len(missing_meta_products),
        })
        cat_pdf   = pd.concat([cat_pdf, miss_rows_cat],     ignore_index=True)
        brand_pdf = pd.concat([brand_pdf, miss_rows_brand], ignore_index=True)

    unique_cats   = sorted(cat_pdf["category"].dropna().unique().tolist())
    unique_brands = sorted(brand_pdf["brand"].dropna().unique().tolist())
    category2idx  = {c: i for i, c in enumerate(unique_cats)}
    brand2idx     = {b: i for i, b in enumerate(unique_brands)}

    cat_pdf["category_idx"] = cat_pdf["category"].map(category2idx).astype(np.int64)
    brand_pdf["brand_idx"]  = brand_pdf["brand"].map(brand2idx).astype(np.int64)

    prod_cat_df = cat_pdf[["product_idx", "category_idx"]].copy()
    prod_cat_df["product_idx"]  = prod_cat_df["product_idx"].astype(np.int64)
    prod_brand_df = brand_pdf[["product_idx", "brand_idx"]].copy()
    prod_brand_df["product_idx"] = prod_brand_df["product_idx"].astype(np.int64)

    n_unknown_cat   = int((cat_pdf["category"]   == unknown_category).sum())
    n_unknown_brand = int((brand_pdf["brand"]    == unknown_brand).sum())
    meta_stats = {
        "metadata_source":              metadata_source,
        "num_products":                 len(item2idx),
        "num_categories":               len(category2idx),
        "num_brands":                   len(brand2idx),
        "products_unknown_category":    n_unknown_cat,
        "products_unknown_brand":       n_unknown_brand,
        "products_missing_metadata":    len(missing_meta_products),
    }

    item_metadata = pd.DataFrame({
        "product_idx":     prod_cat_df["product_idx"].values,
    }).merge(
        cat_pdf[["product_idx", "category", "category_idx"]],   on="product_idx", how="left",
    ).merge(
        brand_pdf[["product_idx", "brand", "brand_idx"]],       on="product_idx", how="left",
    )
    item_metadata["metadata_source"] = metadata_source

    logger.info(
        "Structural edges: %d products -> %d categories, %d brands. unknown_cat=%d unknown_brand=%d",
        len(item2idx), len(category2idx), len(brand2idx),
        n_unknown_cat, n_unknown_brand,
    )

    meta_stats["item_metadata"] = item_metadata
    return prod_cat_df, prod_brand_df, category2idx, brand2idx, meta_stats


def build_train_mask(
    purchase_train: pd.DataFrame,
    aux_spark: DataFrame | None,
    eval_user_ids: Iterable[int],
    spark: SparkSession,
    *,
    mask_behaviors: tuple[str, ...] = ("purchase",),
) -> dict[int, list[int]]:
    """Build per-user seen-item sets for full-ranking eval masking.

    Primary call uses ``mask_behaviors=("purchase",)``. The seen-all diagnostic
    mask uses ``("view", "cart", "purchase")``. The auxiliary DataFrame may be
    None when ``mask_behaviors`` does not include any auxiliary behavior.
    """
    eval_users = sorted({int(u) for u in eval_user_ids})
    eval_users_df = spark.createDataFrame(
        [(u,) for u in eval_users], schema="user_idx LONG"
    )

    parts = []
    if "purchase" in mask_behaviors and not purchase_train.empty:
        purchase_spark = spark.createDataFrame(
            purchase_train[["user_idx", "item_idx"]]
            .astype({"user_idx": "int64", "item_idx": "int64"})
        )
        parts.append(
            purchase_spark.select(
                F.col("user_idx").cast("long"),
                F.col("item_idx").cast("long"),
            )
        )

    aux_behaviors = tuple(b for b in mask_behaviors if b in ("view", "cart"))
    if aux_behaviors:
        if aux_spark is None:
            raise ValueError(
                f"aux_spark is required when mask_behaviors includes {aux_behaviors}"
            )
        parts.append(
            aux_spark
            .filter(F.col("event_type").isin(list(aux_behaviors)))
            .select(
                F.col("user_idx").cast("long"),
                F.col("item_idx").cast("long"),
            )
        )

    if not parts:
        return {u: [] for u in eval_users}

    union_df = parts[0]
    for p in parts[1:]:
        union_df = union_df.union(p)

    mask_rows = (
        union_df
        .join(F.broadcast(eval_users_df), on="user_idx", how="inner")
        .distinct()
        .groupBy("user_idx")
        .agg(F.collect_list("item_idx").alias("seen_items"))
        .collect()
    )

    mask = {int(r["user_idx"]): [int(x) for x in r["seen_items"]] for r in mask_rows}
    for u in eval_users:
        mask.setdefault(u, [])

    avg = float(np.mean([len(v) for v in mask.values()])) if mask else 0.0
    logger.info(
        "Train mask (mask_behaviors=%s): %d users, avg %.1f seen items per user.",
        mask_behaviors, len(mask), avg,
    )
    return mask
