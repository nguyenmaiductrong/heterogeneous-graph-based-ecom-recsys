import os
import yaml
import time
from functools import wraps
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, LongType, TimestampType
)

_CONFIG_CACHE = None
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

def get_project_root() -> str:
    return _PROJECT_ROOT


def load_config(config_path: str | None = None) -> dict:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE

    if config_path is None:
        config_path = os.path.join(_PROJECT_ROOT, "config", "spark_config.yaml")

    with open(config_path, "r") as f:
        _CONFIG_CACHE = yaml.safe_load(f)

    return _CONFIG_CACHE


def ensure_dirs(cfg: dict) -> None:
    paths = cfg["paths"]
    for key in ["output_dir", "node_mappings_dir", "edge_lists_dir",
                "splits_dir", "stats_dir", "graph_dir", "small_dir"]:
        os.makedirs(paths[key], exist_ok=True)


def create_spark_session(
    cfg: dict | None = None, app_name_suffix: str = ""
) -> SparkSession:
    if cfg is None:
        cfg = load_config()

    sc = cfg["spark"]
    app_name = sc["app_name"]
    if app_name_suffix:
        app_name = f"{app_name}_{app_name_suffix}"

    spark = (SparkSession.builder
        .appName(app_name)
        .master(sc["master"])
        .config("spark.driver.memory", sc["driver_memory"])
        .config("spark.executor.memory", sc["executor_memory"])
        .config("spark.local.dir", sc["local_dir"])
        .config("spark.sql.shuffle.partitions", sc["shuffle_partitions"])
        .config("spark.default.parallelism", sc["default_parallelism"])
        .config("spark.sql.adaptive.enabled", str(sc["aqe_enabled"]).lower())
        .config("spark.sql.autoBroadcastJoinThreshold", sc["broadcast_threshold"])
        .config("spark.sql.parquet.compression.codec", sc["parquet_compression"])
        .config("spark.sql.files.maxPartitionBytes", sc["max_partition_bytes"])
        .config("spark.memory.fraction", sc["memory_fraction"])
        .config("spark.memory.storageFraction", sc["storage_fraction"])
        .config("spark.sql.shuffle.spill.enabled", "true")
        .config("spark.driver.maxResultSize", sc["driver_max_result_size"])
        .config("spark.sql.execution.arrow.pyspark.enabled",
                str(sc.get("arrow_enabled", True)).lower())
        .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "true")
        .config("spark.executor.heartbeatInterval",
                sc.get("heartbeat_interval", "120s"))
        .config("spark.network.timeout",
                sc.get("network_timeout", "600s"))
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    checkpoint_dir = sc.get("checkpoint_dir", "/tmp/spark_checkpoints")
    spark.sparkContext.setCheckpointDir(checkpoint_dir)
    return spark


def get_rees46_schema() -> StructType:
    return StructType([
        StructField("event_time", TimestampType(), True),
        StructField("event_type", StringType(), True),
        StructField("product_id", LongType(), True),
        StructField("category_id", LongType(), True),
        StructField("category_code", StringType(), True),
        StructField("brand", StringType(), True),
        StructField("price", DoubleType(), True),
        StructField("user_id", LongType(), True),
        StructField("user_session", StringType(), True),
    ])


def log_step(step_name: str):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            print(f"{step_name}")
            t0 = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - t0
            print(f"[{step_name}] Done in {elapsed:.1f}s")
            return result
        return wrapper
    return decorator


def count_and_log(df: DataFrame, label: str) -> int:
    n = df.count()
    print(f"{label}: {n:,}")
    return n


def get_dir_size_gb(path: str) -> float:
    total = 0
    for dp, dn, filenames in os.walk(path):
        for f in filenames:
            total += os.path.getsize(os.path.join(dp, f))
    return total / (1024 ** 3)