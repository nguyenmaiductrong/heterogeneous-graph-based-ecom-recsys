import pytest
from pyspark.sql import SparkSession

@pytest.fixture(scope="session")
def spark():
    spark_session = SparkSession.builder \
        .appName("pytest-pyspark-local") \
        .master("local[2]") \
        .config("spark.driver.memory", "2g") \
        .config("spark.sql.shuffle.partitions", "2") \
        .getOrCreate()
    yield spark_session
    spark_session.stop()