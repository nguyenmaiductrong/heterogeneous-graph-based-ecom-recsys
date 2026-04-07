from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from .splitter import DataSplitter, SplitResult

logger = logging.getLogger(__name__)


def filter_cold_items(
    df: DataFrame,
    target_behavior: str,
    min_item_interactions: int,
    rounds: int = 3,
) -> DataFrame:
    """Iterative cold-start filter.

    Each round removes items with fewer than min_item_interactions target-behavior
    interactions, then drops ALL event types for those items.

    checkpoint(eager=False) breaks the growing lineage chain — without it, the DAG
    grows O(rounds) deep and causes stack overflow + planning-time blowup.
    valid_items is small (<10MB) so Spark auto-broadcasts it on every join.
    """
    if min_item_interactions <= 1:
        return df

    for r in range(rounds):
        valid_items = (
            df.filter(F.col("event_type") == target_behavior)
            .groupBy("product_id")
            .count()
            .filter(F.col("count") >= min_item_interactions)
            .select("product_id")
        )
        df = df.join(F.broadcast(valid_items), on="product_id", how="inner")
        df = df.checkpoint(eager=False)
        logger.info("filter_cold_items: round %d/%d — checkpoint queued.", r + 1, rounds)

    return df


def temporal_split_purchases(
    df: DataFrame,
    target_behavior: str,
    min_interactions: int,
    train_end: str,
    val_end: str,
) -> SplitResult:
    """Date-based global temporal split via DataSplitter (Pandas).

    The min_interactions user filter runs in Spark before toPandas() so the
    driver never sees cold-start users at all. Cold-start items (only purchased
    after the training cutoff) are retained in the item vocabulary by
    DataSplitter so that BAGNN can embed them via structural graph edges.
    """
    purchase_spark = df.filter(F.col("event_type") == target_behavior).select(
        F.col("user_id"),
        F.col("product_id").alias("item_id"),
        F.col("timestamp"),
        F.col("event_type"),
    )

    if min_interactions > 0:
        user_counts = (
            purchase_spark
            .groupBy("user_id")
            .count()
        ).cache()

        stats = user_counts.agg(
            F.count("*").alias("n_before"),
            F.sum((F.col("count") >= min_interactions).cast("long")).alias("n_after"),
        ).collect()[0]
        logger.info(
            "Temporal min_interactions filter: kept %d / %d users (min=%d)",
            stats["n_after"], stats["n_before"], min_interactions,
        )

        valid_users = (
            user_counts
            .filter(F.col("count") >= min_interactions)
            .select("user_id")
        )
        purchase_spark = (
            purchase_spark
            .join(F.broadcast(valid_users), on="user_id", how="inner")
            .cache()
        )
        purchase_df = purchase_spark.toPandas()
        purchase_spark.unpersist()
        user_counts.unpersist()
    else:
        purchase_spark = purchase_spark.cache()
        purchase_df = purchase_spark.toPandas()
        purchase_spark.unpersist()

    logger.info(
        "Temporal — %s interactions: %d  (unique users: %d, items: %d)",
        target_behavior, len(purchase_df),
        purchase_df["user_id"].nunique(), purchase_df["item_id"].nunique(),
    )

    split = DataSplitter(purchase_df).temporal_split_by_dates(
        train_end=train_end, val_end=val_end,
    )
    logger.info("\n%s", split.summary())
    return split


def map_auxiliary(
    df: DataFrame,
    spark: SparkSession,
    split: SplitResult,
    target_behavior: str,
    *,
    global_cutoff: int,
) -> DataFrame:
    """Map non-target interactions through the purchase vocabulary via broadcast join.

    Returns a Spark DataFrame — do NOT call .toPandas() on it.
    Auxiliary events can be 100M+ rows; use _spark_ei_to_npy() to persist to disk.

    user2idx (~5MB) and item2idx (~1MB) are well below the 256MB broadcast threshold.
    Temporal leakage is prevented via global_cutoff: only events strictly before
    the training cutoff are included.

    item2idx now covers ALL items (including cold-start ones), so auxiliary view /
    cart events for items that are only purchased after the cutoff are correctly
    captured and provide training signal to BAGNN.
    """
    user_map = spark.createDataFrame(
        [(int(k), int(v)) for k, v in split.user2idx.items()],
        schema="user_id LONG, user_idx LONG",
    )
    item_map = spark.createDataFrame(
        [(int(k), int(v)) for k, v in split.item2idx.items()],
        schema="product_id LONG, item_idx LONG",
    )

    aux = (
        df.filter(F.col("event_type") != target_behavior)
        .join(F.broadcast(user_map), on="user_id",    how="inner")
        .join(F.broadcast(item_map), on="product_id", how="inner")
        .filter(F.col("timestamp") < global_cutoff)
    )

    logger.info("Auxiliary [temporal]: global_cutoff=%d applied.", global_cutoff)

    return aux.select(
        F.col("user_idx").cast("long"),
        F.col("item_idx").cast("long"),
        F.col("timestamp").cast("long"),
        F.col("event_type"),
    )


def build_structural_edges(
    df: DataFrame,
    spark: SparkSession,
    item2idx: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """Build product→category and product→brand edges via Spark Window functions.

    Mode (most frequent value) per product is computed with row_number() over a
    count-based window — pure Spark SQL, no Python UDFs, no collect() on large data.

    Returns
    -------
    prod_cat_df   : Pandas DataFrame (product_idx, category_idx) — int64 columns
    prod_brand_df : Pandas DataFrame (product_idx, brand_idx)    — int64 columns
    category2idx  : dict
    brand2idx     : dict
    """
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

    unique_cats   = sorted(r[0] for r in cat_mode.select("category").distinct().collect())
    unique_brands = sorted(r[0] for r in brand_mode.select("brand").distinct().collect())

    category2idx = {c: i for i, c in enumerate(unique_cats)}
    brand2idx    = {b: i for i, b in enumerate(unique_brands)}

    cat_vocab = spark.createDataFrame(
        list(category2idx.items()), schema="category STRING, category_idx LONG"
    )
    brand_vocab = spark.createDataFrame(
        list(brand2idx.items()), schema="brand STRING, brand_idx LONG"
    )

    prod_cat_df = (
        cat_mode
        .join(F.broadcast(cat_vocab), on="category", how="inner")
        .select(
            F.col("product_idx").cast("long"),
            F.col("category_idx").cast("long"),
        )
        .toPandas()
    )
    prod_brand_df = (
        brand_mode
        .join(F.broadcast(brand_vocab), on="brand", how="inner")
        .select(
            F.col("product_idx").cast("long"),
            F.col("brand_idx").cast("long"),
        )
        .toPandas()
    )
    item_rows.unpersist()

    logger.info(
        "Structural edges: %d products -> %d categories, %d brands.",
        len(item2idx), len(category2idx), len(brand2idx),
    )
    return prod_cat_df, prod_brand_df, category2idx, brand2idx


def build_train_mask(
    purchase_train: pd.DataFrame,
    aux_spark: DataFrame,
    test_df: pd.DataFrame,
    spark: SparkSession,
) -> dict[int, list[int]]:
    """Build per-user seen-item sets for full-ranking eval masking.

    Operates in Spark so that the 100M+ auxiliary rows never land on the driver.
    The final collect() is safe: it works on the per-user aggregation (one Row per
    eval user) rather than the raw edge data.
    """
    eval_users = set(test_df["user_idx"].astype(int).unique())

    eval_users_df = spark.createDataFrame(
        [(int(u),) for u in eval_users], schema="user_idx LONG"
    )

    purchase_spark = spark.createDataFrame(
        purchase_train[["user_idx", "item_idx"]]
        .astype({"user_idx": "int64", "item_idx": "int64"})
    )

    mask_rows = (
        purchase_spark
        .select(F.col("user_idx").cast("long"), F.col("item_idx").cast("long"))
        .union(
            aux_spark.select(F.col("user_idx").cast("long"), F.col("item_idx").cast("long"))
        )
        .join(F.broadcast(eval_users_df), on="user_idx", how="inner")
        .distinct()
        .groupBy("user_idx")
        .agg(F.collect_list("item_idx").alias("seen_items"))
        .collect()
    )

    mask = {int(r["user_idx"]): [int(x) for x in r["seen_items"]] for r in mask_rows}

    logger.info(
        "Train mask: %d eval users, avg %.1f seen items per user.",
        len(mask),
        np.mean([len(v) for v in mask.values()]) if mask else 0.0,
    )
    return mask
