import os
import json
import numpy as np
import scipy.sparse as sp
from pyspark.sql import SparkSession, DataFrame
from .spark_utils import get_rees46_schema, log_step, count_and_log

# Raw data
@log_step("EXTRACT: Load raw CSV")
def load_raw_csv(spark: SparkSession, csv_pattern: str) -> DataFrame:
    df = (spark.read
        .option("header", "true")
        .option("mode", "DROPMALFORMED")
        .option("timestampFormat", "yyyy-MM-dd HH:mm:ss z")
        .schema(get_rees46_schema())
        .csv(csv_pattern)
    )
    count_and_log(df, "Raw records loaded")
    return df

# Processed data
@log_step("EXTRACT: Load cleaned Parquet")
def load_cleaned_parquet(spark: SparkSession, output_dir: str) -> DataFrame:
    path = os.path.join(output_dir, "cleaned.parquet")
    df = spark.read.parquet(path)
    count_and_log(df, "Cleaned records loaded")
    return df

# Node mappings 
def load_node_mapping(spark: SparkSession, mappings_dir: str,
                      name: str) -> DataFrame:
    path = os.path.join(mappings_dir, f"{name}.parquet")
    return spark.read.parquet(path)

def load_all_node_mappings(spark: SparkSession, mappings_dir: str) -> dict:
    mappings = {}
    for name in ["user2idx", "product2idx", "category2idx", "brand2idx"]:
        mappings[name] = load_node_mapping(spark, mappings_dir, name)
    return mappings

# Graph metadata
def load_node_summary(stats_dir: str) -> dict:
    path = os.path.join(stats_dir, "node_summary.json")
    with open(path, "r") as f:
        return json.load(f)

def load_graph_meta(graph_dir: str) -> dict:
    path = os.path.join(graph_dir, "graph_meta.json")
    with open(path, "r") as f:
        return json.load(f)

# Numpy edge arrays
def load_edge_arrays(edge_dir: str, prefix: str):
    src = np.load(os.path.join(edge_dir, f"{prefix}_src.npy"))
    dst = np.load(os.path.join(edge_dir, f"{prefix}_dst.npy"))
    ts_path = os.path.join(edge_dir, f"{prefix}_ts.npy")
    ts = np.load(ts_path) if os.path.exists(ts_path) else None
    return src, dst, ts