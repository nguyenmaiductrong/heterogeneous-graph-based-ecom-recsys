import os
import pytest
from pyspark.sql import SparkSession
from src.data_pipeline.spark_utils import log_step, count_and_log, get_dir_size_gb

def test_log_step():
    @log_step(step_name="Test Decorator")
    def dummy_addition(a, b):
        return a + b

    result = dummy_addition(3, 5)
    assert result == 8

def test_count_and_log(spark, capsys):
    data = [("User1", "ProductA"), ("User2", "ProductB"), ("User3", "ProductC")]
    df = spark.createDataFrame(data, ["user_id", "product_id"])
    
    count = count_and_log(df, "Total Interactions")
    
    assert count == 3
    
    captured = capsys.readouterr()
    assert "Total Interactions: 3" in captured.out

def test_get_dir_size_gb(tmp_path):
    file1 = tmp_path / "dummy1.txt"
    file1.write_bytes(b"0" * 1024 * 1024) 
    
    file2 = tmp_path / "dummy2.txt"
    file2.write_bytes(b"0" * 1024 * 1024)
    
    size_in_gb = get_dir_size_gb(str(tmp_path))
    
    expected_gb = (2 * 1024 * 1024) / (1024 ** 3)
    
    assert size_in_gb == expected_gb