"""Khởi tạo SparkSession, schema REES46 và nạp YAML cấu hình pipeline."""
from __future__ import annotations
import logging
import os
import shutil
import time
from functools import wraps

import yaml
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import (
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

logger = logging.getLogger(__name__)

_CONFIG_CACHE: dict | None = None
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

    with open(config_path) as f:
        _CONFIG_CACHE = yaml.safe_load(f)

    return _CONFIG_CACHE


def ensure_dirs(cfg: dict) -> None:
    paths = cfg["paths"]
    for key in [
        "output_dir", "node_mappings_dir", "edge_lists_dir",
        "splits_dir", "stats_dir", "graph_dir", "small_dir",
    ]:
        os.makedirs(paths[key], exist_ok=True)


def create_spark_session(
    cfg: dict | None = None,
    app_name_suffix: str = "",
) -> SparkSession:
    if cfg is None:
        cfg = load_config()

    sc = cfg["spark"]
    app_name = sc["app_name"] + (f"_{app_name_suffix}" if app_name_suffix else "")

    spark = (
        SparkSession.builder
        .appName(app_name)
        .master(sc["master"])
        # Bộ nhớ
        .config("spark.driver.memory", sc["driver_memory"])
        .config("spark.executor.memory", sc["executor_memory"])
        .config("spark.driver.maxResultSize", sc["driver_max_result_size"])
        # Thư mục tạm / checkpoint cục bộ
        .config("spark.local.dir", sc["local_dir"])
        # Song song
        .config("spark.sql.shuffle.partitions", sc["shuffle_partitions"])
        .config("spark.default.parallelism", sc["default_parallelism"])
        # Adaptive Query Execution (AQE)
        .config("spark.sql.adaptive.enabled", str(sc["aqe_enabled"]).lower())
        .config(
            "spark.sql.adaptive.coalescePartitions.enabled",
            str(sc.get("adaptive_coalesce_enabled", True)).lower(),
        )
        .config(
            "spark.sql.adaptive.skewJoin.enabled",
            str(sc.get("adaptive_skew_join_enabled", True)).lower(),
        )
        .config("spark.sql.autoBroadcastJoinThreshold", sc["broadcast_threshold"])
        # I/O
        .config("spark.sql.parquet.compression.codec", sc["parquet_compression"])
        .config("spark.sql.files.maxPartitionBytes", sc["max_partition_bytes"])
        # Tràn shuffle ra đĩa (giảm OOM khi group-by lớn)
        .config("spark.sql.shuffle.spill.enabled", "true")
        # Quản lý bộ nhớ
        .config("spark.memory.fraction", sc["memory_fraction"])
        .config("spark.memory.storageFraction", sc["storage_fraction"])
        .config(
            "spark.sql.execution.arrow.pyspark.enabled",
            str(sc.get("arrow_enabled", True)).lower(),
        )
        .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "true")
        .config("spark.executor.heartbeatInterval", sc.get("heartbeat_interval", "120s"))
        .config("spark.network.timeout", sc.get("network_timeout", "600s"))
        # Cố định múi giờ session SQL để cast/parse timestamp lặp được
        .config("spark.sql.session.timeZone", sc.get("session_timezone", "UTC"))
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    checkpoint_dir = sc.get("checkpoint_dir", "/tmp/spark_checkpoints")
    # Xóa checkpoint cũ từ các lần chạy trước khi bật phiên Spark mới.
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
        logger.info("Đã dọn checkpoint cũ: %s", checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    spark.sparkContext.setCheckpointDir(checkpoint_dir)

    logger.info(
        "SparkSession sẵn sàng — master=%s driver_mem=%s shuffle_parts=%s AQE=bật",
        sc["master"], sc["driver_memory"], sc["shuffle_partitions"],
    )
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
    """Decorator: ghi log lúc vào/ra và thời gian thực thi của một bước pipeline."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.info(">>> %s", step_name)
            t0 = time.time()
            result = func(*args, **kwargs)
            logger.info("<<< %s hoàn thành sau %.1f giây", step_name, time.time() - t0)
            return result
        return wrapper
    return decorator


def count_and_log(df: DataFrame, label: str) -> int:
    """Kích hoạt action count và in kết quả (giữ print() để test tương thích)."""
    n = df.count()
    print(f"{label}: {n:,}")
    return n


def get_dir_size_gb(path: str) -> float:
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for fname in filenames:
            total += os.path.getsize(os.path.join(dirpath, fname))
    return total / (1024 ** 3)
