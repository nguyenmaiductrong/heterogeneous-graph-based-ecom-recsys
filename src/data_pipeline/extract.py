from __future__ import annotations

import logging

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from .spark_utils import get_rees46_schema, load_config

logger = logging.getLogger(__name__)

_VALID_BEHAVIORS = ("view", "cart", "purchase")


def load_raw_csvs(spark: SparkSession, csv_glob: str) -> DataFrame:
    df = (
        spark.read
        .option("header", "true")
        .option("mode", "DROPMALFORMED")
        .option("timestampFormat", "yyyy-MM-dd HH:mm:ss z")
        .schema(get_rees46_schema())
        .csv(csv_glob)
    )
    logger.info("Queued lazy CSV read: %s", csv_glob)
    return df


def clean(df: DataFrame, cfg: dict | None = None) -> DataFrame:
    if cfg is None:
        cfg = load_config()

    fc = cfg["filter"]
    unknown_brand = fc.get("unknown_brand", "unknown")
    unknown_cat   = fc.get("unknown_category", "unknown")
    level         = fc.get("category_level", "top")

    df = df.dropna(subset=["user_id", "product_id", "event_type", "event_time"])
    df = df.filter(F.col("event_type").isin(list(_VALID_BEHAVIORS)))
    df = df.dropDuplicates(["user_id", "product_id", "event_type", "event_time"])
    df = df.withColumn("timestamp", F.col("event_time").cast("long"))

    brand_clean = F.lower(F.trim(F.col("brand")))
    df = df.withColumn(
        "brand",
        F.when(F.col("brand").isNotNull() & (brand_clean != ""), brand_clean)
         .otherwise(F.lit(unknown_brand)),
    )

    if level == "top":
        cat_expr = F.when(
            F.col("category_code").isNotNull() & (F.col("category_code") != ""),
            F.split(F.col("category_code"), r"\.").getItem(0),
        ).otherwise(F.lit(unknown_cat))
    elif level == "second":
        parts = F.split(F.col("category_code"), r"\.")
        cat_expr = F.when(
            F.col("category_code").isNotNull() & (F.col("category_code") != ""),
            F.concat_ws(".", parts.getItem(0), F.coalesce(parts.getItem(1), F.lit("general"))),
        ).otherwise(F.lit(unknown_cat))
    else:
        cat_expr = F.when(
            F.col("category_code").isNotNull() & (F.col("category_code") != ""),
            F.col("category_code"),
        ).otherwise(F.lit(unknown_cat))

    df = df.withColumn("category", cat_expr)

    return df.select("user_id", "product_id", "event_type", "event_time", "timestamp", "category", "brand", "price")
