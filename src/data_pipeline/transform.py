from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from .spark_utils import log_step, count_and_log
from pyspark.sql.types import StructType, StructField, LongType

# STEP 1: CLEANING
@log_step("TRANSFORM: Clean data")
def clean_raw_data(df: DataFrame, cfg: dict) -> DataFrame:
    fc = cfg["filter"]
    behaviors = cfg["graph"]["behavior_edge_types"]

    # 1. Drop nulls
    df = df.dropna(subset=["user_id", "product_id", "event_type", "event_time"])

    # 2. Filter behaviors
    df = df.filter(F.col("event_type").isin(behaviors))

    # 2b. Validate price — loại null, âm, 0, và outlier cực đoan
    price_lower = fc.get("price_min", 0.01)
    price_upper = fc.get("price_max", 50000.0)
    df = df.filter(
        F.col("price").isNotNull()
        & (F.col("price") >= price_lower)
        & (F.col("price") <= price_upper)
    )

    # 2c. Validate timestamp — loại records ngoài khoảng hợp lệ
    #     (parse lỗi -> epoch 1970, hoặc timestamp tương lai)
    date_min = fc.get("date_min", "2019-10-01")
    date_max = fc.get("date_max", "2020-04-30")
    df = df.filter(
        (F.col("event_time") >= F.lit(date_min).cast("timestamp"))
        & (F.col("event_time") <= F.lit(date_max).cast("timestamp"))
    )

    # 2d. Lọc bot/spam users — users có quá nhiều interactions/ngày
    #     là bot hoặc crawler, sẽ dominate graph và làm sai embedding
    max_daily = fc.get("max_daily_interactions", 500)
    daily_counts = (df
        .withColumn("_date", F.to_date(F.col("event_time")))
        .groupBy("user_id", "_date")
        .count()
    )
    bot_users = (daily_counts
        .filter(F.col("count") > max_daily)
        .select("user_id")
        .distinct()
    )
    n_bots = bot_users.count()
    print(f"  Bot/spam users detected (>{max_daily} interactions/day): {n_bots:,}")
    df = df.join(bot_users, on="user_id", how="left_anti")

    # 3. Category extraction
    level = fc["category_level"]
    unknown_cat = fc["unknown_category"]

    if level == "top":
        df = df.withColumn(
            "category",
            F.when(
                F.col("category_code").isNotNull() & (F.col("category_code") != ""),
                F.split(F.col("category_code"), "\\.").getItem(0)
            ).otherwise(F.lit(unknown_cat))
        )
    elif level == "second":
        df = df.withColumn(
            "category",
            F.when(
                F.col("category_code").isNotNull() & (F.col("category_code") != ""),
                F.concat_ws(".",
                    F.split(F.col("category_code"), "\\.").getItem(0),
                    F.coalesce(
                        F.split(F.col("category_code"), "\\.").getItem(1),
                        F.lit("general")
                    )
                )
            ).otherwise(F.lit(unknown_cat))
        )
    else:  # full
        df = df.withColumn(
            "category",
            F.when(
                F.col("category_code").isNotNull() & (F.col("category_code") != ""),
                F.col("category_code")
            ).otherwise(F.lit(unknown_cat))
        )

    # 4. Brand
    unknown_brand = fc["unknown_brand"]
    if fc["fill_missing_brand"]:
        df = df.withColumn(
            "brand_clean",
            F.when(
                F.col("brand").isNotNull() & (F.col("brand") != ""),
                F.lower(F.trim(F.col("brand")))
            ).otherwise(F.lit(unknown_brand))
        )
    else:
        df = (df
            .filter(F.col("brand").isNotNull() & (F.col("brand") != ""))
            .withColumn("brand_clean", F.lower(F.trim(F.col("brand"))))
        )

    # 5. Deduplicate
    df = df.dropDuplicates(["user_id", "product_id", "event_type", "event_time"])

    # 6. Date column
    df = df.withColumn("event_date", F.to_date(F.col("event_time")))

    # Select final columns
    df = df.select(
        "user_id", "product_id", "event_type",
        "event_time", "event_date",
        "category", "brand_clean", "price"
    )

    count_and_log(df, "After cleaning")
    return df

# STEP 1b: COLD-START FILTERING
@log_step("TRANSFORM: Filter cold-start")
def filter_cold_start(df: DataFrame, cfg: dict) -> DataFrame:
    fc = cfg["filter"]
    target = cfg["graph"]["target_behavior"]
    rounds = fc["iterative_filter_rounds"]

    prev_count = df.count()
    print(f"  Before filtering: {prev_count:,}")

    for i in range(rounds):
        # --- Filter users ---
        user_counts = df.groupBy("user_id").count()
        if fc["require_purchase"]:
            purchase_users = (df
                .filter(F.col("event_type") == target)
                .select("user_id").distinct()
            )
            valid_users = (user_counts
                .filter(F.col("count") >= fc["min_user_interactions"])
                .join(purchase_users, on="user_id", how="inner")
                .select("user_id")
            )
        else:
            valid_users = (user_counts
                .filter(F.col("count") >= fc["min_user_interactions"])
                .select("user_id")
            )
        df = df.join(F.broadcast(valid_users), on="user_id", how="inner")

        # --- Filter items ---
        item_counts = df.groupBy("product_id").count()
        valid_items = (item_counts
            .filter(F.col("count") >= fc["min_item_interactions"])
            .select("product_id")
        )
        df = df.join(F.broadcast(valid_items), on="product_id", how="inner")
        df = df.checkpoint()
        curr_count = df.count()
        removed = prev_count - curr_count
        print(f"  Round {i+1}: {curr_count:,} (removed {removed:,})")
        if removed < prev_count * 0.001:
            break
        prev_count = curr_count

    return df

# STEP 2: NODE ID MAPPINGS
@log_step("TRANSFORM: Build node mappings")
def build_node_mapping(df: DataFrame, col_name: str, mapping_name: str) -> DataFrame:
    distinct_rdd = df.select(col_name).distinct().rdd.map(lambda r: r[0])
    indexed_rdd  = distinct_rdd.zipWithIndex()

    schema = StructType([
        StructField(col_name, df.schema[col_name].dataType, False),
        StructField("idx", LongType(), False),
    ])
    mapping_df = df.sparkSession.createDataFrame(indexed_rdd, schema)
    n = mapping_df.count()
    print(f"  {mapping_name}: {n:,} unique entries")
    return mapping_df

@log_step("TRANSFORM: Build Product -> Category lookup")
def build_product_category(df: DataFrame, product_map: DataFrame,
                           category_map: DataFrame) -> DataFrame:
    w = Window.partitionBy("product_id").orderBy(F.desc("count"))
    product_cat = (df
        .groupBy("product_id", "category").count()
        .withColumn("rank", F.row_number().over(w))
        .filter(F.col("rank") == 1)
        .select("product_id", "category")
    )

    result = (product_cat
        .join(product_map.select(
            F.col("product_id"), F.col("idx").alias("product_idx")),
            on="product_id", how="inner")
        .join(category_map.select(
            F.col("category"), F.col("idx").alias("category_idx")),
            on="category", how="inner")
        .select("product_idx", "category_idx")
    )
    count_and_log(result, "Product -> Category links")
    return result

@log_step("TRANSFORM: Build Product -> Brand lookup")
def build_product_brand(df: DataFrame, product_map: DataFrame,
                        brand_map: DataFrame) -> DataFrame:
    w = Window.partitionBy("product_id").orderBy(F.desc("count"))
    product_brand = (df
        .groupBy("product_id", "brand_clean").count()
        .withColumn("rank", F.row_number().over(w))
        .filter(F.col("rank") == 1)
        .select("product_id", "brand_clean")
    )

    result = (product_brand
        .join(product_map.select(
            F.col("product_id"), F.col("idx").alias("product_idx")),
            on="product_id", how="inner")
        .join(brand_map.select(
            F.col("brand_clean"), F.col("idx").alias("brand_idx")),
            on="brand_clean", how="inner")
        .select("product_idx", "brand_idx")
    )
    count_and_log(result, "Product -> Brand links")
    return result

# STEP 3: BEHAVIOR EDGE LISTS
def build_behavior_edges(df: DataFrame, behavior: str,
                         user_map: DataFrame,
                         product_map: DataFrame) -> DataFrame:
    edges = (df
        .filter(F.col("event_type") == behavior)
        .join(user_map.select(
            F.col("user_id"), F.col("idx").alias("user_idx")),
            on="user_id", how="inner")
        .join(product_map.select(
            F.col("product_id"), F.col("idx").alias("product_idx")),
            on="product_id", how="inner")
        .withColumn("unix_ts", F.col("event_time").cast("long"))
        .select("user_idx", "product_idx", "unix_ts", "event_date")
    )
    n = edges.count()
    print(f"  {behavior:>10s}: {n:>12,} edges")
    return edges

def temporal_split(edges_df: DataFrame, behavior: str, cfg: dict) -> tuple:
    sc = cfg["split"]
    
    labeled = edges_df.withColumn(
        "split",
        F.when(F.col("event_date") <= sc["train_end"], F.lit("train"))
         .when(F.col("event_date") <= sc["val_end"],   F.lit("val"))
         .otherwise(F.lit("test"))
    )
    labeled.cache()

    train_df = labeled.filter(F.col("split") == "train").drop("split")
    val_df   = labeled.filter(F.col("split") == "val").drop("split")
    test_df  = labeled.filter(F.col("split") == "test").drop("split")

    counts = (labeled.groupBy("split").count()
                     .collect())
    count_map = {r["split"]: r["count"] for r in counts}
    total = sum(count_map.values())
    for s in ["train", "val", "test"]:
        n = count_map.get(s, 0)
        print(f"{behavior} {s}: {n:,} ({n/total*100:.1f}%)")

    labeled.unpersist()
    return train_df, val_df, test_df

# STEP 3b: LEAVE-ONE-OUT SPLIT (Protocol A — chuẩn baselines cho A* paper)

@log_step("TRANSFORM: Leave-one-out split (Protocol A)")
def leave_one_out_split(df: DataFrame, cfg: dict,
                        user_map: DataFrame,
                        product_map: DataFrame,
                        loo_dir: str) -> dict:
    target = cfg["graph"]["target_behavior"]
    spark = df.sparkSession

    w = Window.partitionBy("user_id").orderBy(F.desc("event_time"))

    purchases = (df
        .filter(F.col("event_type") == target)
        .withColumn("loo_rank", F.row_number().over(w))
    )

    purchases = purchases.checkpoint()

    user_max_rank = (purchases
        .groupBy("user_id")
        .agg(F.max("loo_rank").alias("max_rank"))
    )
    valid_eval_users = (user_max_rank
        .filter(F.col("max_rank") >= 2)
        .select("user_id")
    )
    n_eval_users = valid_eval_users.count()
    n_total_users = user_max_rank.count()
    n_dropped = n_total_users - n_eval_users
    print(f"  Total users with purchases:    {n_total_users:,}")
    print(f"  Users with >= 2 purchases:     {n_eval_users:,}")
    print(f"  Users dropped (< 2 purchases): {n_dropped:,}")

    test_purchases = (purchases
        .join(valid_eval_users, on="user_id", how="inner")
        .filter(F.col("loo_rank") == 1)
    )

    val_purchases = (purchases
        .join(valid_eval_users, on="user_id", how="inner")
        .filter(F.col("loo_rank") == 2)
    )

    test_val_keys = (test_purchases
        .select("user_id", "product_id", "event_time")
        .unionByName(
            val_purchases.select("user_id", "product_id", "event_time")
        )
    )
    test_val_keys = test_val_keys.checkpoint()

    train_purchases = (df
        .filter(F.col("event_type") == target)
        .join(test_val_keys,
              on=["user_id", "product_id", "event_time"],
              how="left_anti")
    )

    auxiliary = df.filter(F.col("event_type") != target)
    cols = ["user_id", "product_id", "event_type",
            "event_time", "event_date", "category", "brand_clean", "price"]
    train_all = (train_purchases.select(cols)
                 .unionByName(auxiliary.select(cols)))

    def to_indexed(src_df):
        return (src_df
            .join(user_map.select(
                F.col("user_id"), F.col("idx").alias("user_idx")),
                on="user_id", how="inner")
            .join(product_map.select(
                F.col("product_id"), F.col("idx").alias("product_idx")),
                on="product_id", how="inner")
        )

    val_pairs = (to_indexed(val_purchases)
        .select("user_idx", "product_idx")
    )
    val_pairs_path = os.path.join(loo_dir, "_val_pairs_parquet")
    val_pairs.write.mode("overwrite").parquet(val_pairs_path)

    test_pairs = (to_indexed(test_purchases)
        .select("user_idx", "product_idx")
    )
    test_pairs_path = os.path.join(loo_dir, "_test_pairs_parquet")
    test_pairs.write.mode("overwrite").parquet(test_pairs_path)

    train_all_indexed = (to_indexed(train_all)
        .withColumn("unix_ts", F.col("event_time").cast("long"))
        .select("user_idx", "product_idx", "event_type", "unix_ts")
    )
    train_path = os.path.join(loo_dir, "_train_all_parquet")
    train_all_indexed.write.mode("overwrite").parquet(train_path)

    train_reload = spark.read.parquet(train_path)
    train_stats = (train_reload
        .groupBy("event_type").count()
        .collect()
    )
    train_stats_map = {r["event_type"]: r["count"] for r in train_stats}
    n_train_all = sum(train_stats_map.values())
    n_train_purchase = train_stats_map.get(target, 0)
    n_train_view = train_stats_map.get("view", 0)
    n_train_cart = train_stats_map.get("cart", 0)

    n_val = spark.read.parquet(val_pairs_path).count()
    n_test = spark.read.parquet(test_pairs_path).count()

    print(f"LOO Split Summary:")
    print(f"Train total: {n_train_all:>12,}")
    print(f"- view: {n_train_view:>12,}  (all)")
    print(f"- cart: {n_train_cart:>12,}  (all)")
    print(f"- purchase: {n_train_purchase:>12,}  (excl. val+test)")
    print(f"Val pairs: {n_val:>12,}  (1 per user)")
    print(f"Test pairs: {n_test:>12,}  (1 per user)")
    print(f"Eval users: {n_eval_users:>12,}")

    return {
        "train_parquet_path": train_path,
        "val_parquet_path": val_pairs_path,
        "test_parquet_path": test_pairs_path,
        "eval_user_count": n_eval_users,
        "val_count": n_val,
        "test_count": n_test,
    }


import os


# STEP 4: SMALL VERSION SEED SELECTION
@log_step("TRANSFORM: Select seed users for REES46-Small")
def select_seed_users(df: DataFrame, num_users: int) -> DataFrame:
    user_stats = (df
        .groupBy("user_id")
        .agg(
            F.countDistinct("event_type").alias("num_behaviors"),
            F.count("*").alias("total_interactions"),
            F.sum(F.when(F.col("event_type") == "purchase", 1)
                  .otherwise(0)).alias("num_purchases"),
        )
    )

    seed_users = (user_stats
        .orderBy(
            F.desc("num_behaviors"),
            F.desc("num_purchases"),
            F.desc("total_interactions"),
        )
        .limit(num_users)
        .select("user_id")
    )

    count_and_log(seed_users, "Seed users selected")
    return seed_users


@log_step("TRANSFORM: Expand subgraph from seeds")
def expand_subgraph(df: DataFrame, seed_users: DataFrame) -> DataFrame:
    sub_df = df.join(F.broadcast(seed_users), on="user_id", how="inner")
    count_and_log(sub_df, "Subgraph interactions")
    return sub_df

# DATA PROFILING
@log_step("PROFILE: Data statistics")
def print_data_profile(df: DataFrame):
    total = df.count()
    n_users = df.select("user_id").distinct().count()
    n_items = df.select("product_id").distinct().count()
    n_cats = df.select("category").distinct().count()
    n_brands = df.select("brand_clean").distinct().count()

    print(f"Total interactions: {total:,}")
    print(f"Unique users: {n_users:,}")
    print(f"Unique products: {n_items:,}")
    print(f"Unique categories: {n_cats:,}")
    print(f"Unique brands: {n_brands:,}")
    print(f"Density: {total / max(n_users * n_items, 1) * 100:.6f}%")

    # Behavior distribution
    print(f"Behavior distribution:")
    for row in df.groupBy("event_type").count().orderBy(F.desc("count")).collect():
        pct = row["count"] / total * 100
        print(f"{row['event_type']:>10s}: {row['count']:>12,} ({pct:.1f}%)")

    # Date range
    dr = df.agg(
        F.min("event_date").alias("min"), F.max("event_date").alias("max")
    ).collect()[0]
    print(f"Date range: {dr['min']} -> {dr['max']}")