import os
import shutil

import pytest


@pytest.fixture(scope="session")
def spark():
    if not (shutil.which("java") or os.environ.get("JAVA_HOME")):
        pytest.skip("JAVA is not installed; Spark-dependent test skipped.")
    from pyspark.sql import SparkSession
    try:
        spark_session = (
            SparkSession.builder
            .appName("pytest-pyspark-local")
            .master("local[2]")
            .config("spark.driver.memory", "2g")
            .config("spark.sql.shuffle.partitions", "2")
            .getOrCreate()
        )
    except Exception as e:
        pytest.skip(f"Cannot start SparkSession in this environment: {e}")
    yield spark_session
    spark_session.stop()