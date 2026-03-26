import pytest
from src.data_pipeline.extract import load_raw_csv

# ==========================================
# TEST 1: DỮ LIỆU CHUẨN (HAPPY PATH)
# ==========================================
def test_load_raw_csv_happy_path(spark, tmp_path):
    """Kiểm tra hàm đọc đúng file CSV hợp lệ và parse đúng kiểu dữ liệu."""
    
    # 1. Tạo mock data chuẩn format REES46
    valid_csv_content = """event_time,event_type,product_id,category_id,category_code,brand,price,user_id,user_session
2019-10-01 00:00:00 UTC,view,44600062,2103807459595387724,,shiseido,35.79,541312140,72d76fde-8bb3-4e00-8c23-a032dfed738c
2019-10-01 00:00:05 UTC,cart,3900821,2053013552326770905,appliances.environment.water_heater,aqua,33.20,554748717,9333dfbd-b87a-4708-9857-6336556b0fcc
"""
    file_path = tmp_path / "valid_data.csv"
    file_path.write_text(valid_csv_content)
    
    # 2. Thực thi hàm
    df = load_raw_csv(spark, str(file_path))
    
    # 3. Kiểm tra (Assertions)
    # Phải có đúng 2 dòng data
    assert df.count() == 2
    
    # Kéo dòng đầu tiên về driver để kiểm tra chi tiết giá trị và kiểu dữ liệu
    first_row = df.collect()[0]
    
    # Kiểm tra kiểu Integer/Double/String đã được ép đúng chưa
    assert first_row["event_type"] == "view"
    assert first_row["product_id"] == 44600062
    assert first_row["price"] == 35.79
    
    # Kiểm tra cột event_time xem có parse timestamp thành công không 
    # (Nếu lỗi nó sẽ trả về None/Null)
    assert first_row["event_time"] is not None

# ==========================================
# TEST 2: KIỂM TRA CHẾ ĐỘ DROPMALFORMED CHÍNH XÁC
# ==========================================
def test_load_raw_csv_handles_invalid_types(spark, tmp_path):
    """Kiểm tra chế độ DROPMALFORMED có thực sự xóa các dòng bị sai kiểu dữ liệu không."""
    
    # 1. Tạo mock data (LƯU Ý: Phải căn lề trái tuyệt đối để không bị dư khoảng trắng)
    invalid_csv_content = """event_time,event_type,product_id,category_id,category_code,brand,price,user_id,user_session
2019-10-01 00:00:00 UTC,view,44600062,2103807459595387724,,shiseido,35.79,541312140,session_1
2019-10-01 00:00:05 UTC,view,CHUOI_LOI_ID,2053013552326770905,,aqua,33.20,554748717,session_2
"""
    file_path = tmp_path / "invalid_data.csv"
    # Dùng .strip() để loại bỏ các ký tự xuống dòng thừa ở đầu/cuối chuỗi
    file_path.write_text(invalid_csv_content.strip())
    
    # 2. Thực thi hàm
    df = load_raw_csv(spark, str(file_path))
    
    # 3. Ép Spark đọc thật bằng collect() thay vì count()
    actual_data = df.collect()
    
    # Dòng số 2 chứa CHUOI_LOI_ID sẽ bị xóa, nên danh sách chỉ còn lại 1 dòng duy nhất
    assert len(actual_data) == 1, "Spark đã không xóa dòng bị lỗi sai kiểu dữ liệu"
    
    # Kiểm tra xem dòng sống sót chính là dòng hợp lệ (product_id = 44600062)
    assert actual_data[0]["product_id"] == 44600062