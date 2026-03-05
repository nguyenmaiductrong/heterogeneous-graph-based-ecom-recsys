from pyspark.sql import SparkSession
from pyspark.sql.functions import col, collect_set, count

def main():
    spark = SparkSession.builder \
        .appName("REES46_Data_Processing") \
        .getOrCreate()

    print("Đã khởi tạo Spark Session thành công!")

    gcs_input_path = "gs://ten-bucket/data/rees46_ecommerce_data.csv"
    gcs_output_path = "gs://ten-bucket/output/rees46_baskets"

    print(f"Đang đọc tập dữ liệu REES46 từ: {gcs_input_path} ...")

    df = spark.read.csv(gcs_input_path, header=True, inferSchema=True)

    print("--- Cấu trúc dữ liệu REES46 gốc ---")
    df.printSchema()

    purchases_df = df.filter(
        (col("event_type") == "purchase") & 
        col("user_session").isNotNull() &
        col("product_id").isNotNull()
    )

    print("Đang tạo danh sách giỏ hàng theo từng phiên giao dịch...")
    baskets_df = purchases_df.groupBy("user_session").agg(
        collect_set("product_id").alias("items")
    )

    from pyspark.sql.functions import size
    valid_baskets_df = baskets_df.filter(size(col("items")) > 1)

    print("--- 5 Giỏ hàng đầu tiên (đã nhóm) ---")
    valid_baskets_df.show(5, truncate=False)
    
    total_baskets = valid_baskets_df.count()
    print(f"Tổng số giỏ hàng có từ 2 sản phẩm trở lên: {total_baskets}")

    print(f"Đang ghi dữ liệu giỏ hàng ra: {gcs_output_path} ...")
    valid_baskets_df.write.mode("overwrite").parquet(gcs_output_path)

    print("Hoàn tất tiền xử lý tập dữ liệu REES46!")
    spark.stop()

if __name__ == "__main__":
    main()