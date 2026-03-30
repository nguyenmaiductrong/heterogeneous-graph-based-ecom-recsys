import os
import numpy as np
import pickle
import pytest
from src.training.lightgcn_baseline import train

def test_checkpoint_rolling_mechanism(tmp_path, monkeypatch):
    """
    Test đảm bảo cơ chế Rolling Checkpoint hoạt động đúng:
    Chỉ giữ lại đúng 1 file duy nhất để không làm tràn ổ cứng.
    """

    monkeypatch.chdir(tmp_path)

    # 2. TẠO DỮ LIỆU GIẢ LẬP (Mock Data) SIÊU NHỎ
    # Giả lập đồ thị mini chỉ có 4 users và 4 items để test 
    data_dir = tmp_path / "loo"
    data_dir.mkdir()
    
    train_u = np.array([0, 1, 2, 3])
    train_i = np.array([4, 5, 6, 7])
    np.save(data_dir / "loo_purchase_train_src.npy", train_u)
    np.save(data_dir / "loo_purchase_train_dst.npy", train_i)
    
    # Mock train_mask cho Negative Sampling
    train_mask = {0: {4}, 1: {5}, 2: {6}, 3: {7}}
    with open(data_dir / "train_mask.pkl", "wb") as f:
        pickle.dump(train_mask, f)
        
    # 3. THỰC THI TRAINING LOOP
    # Chạy 25 epochs. 
    # Dùng batch_size nhỏ cho data giả lập.
    train(data_dir=str(data_dir), epochs=25, batch_size=2, lr=0.01)
    
    # 4. NGHIỆM THU KẾT QUẢ (Assertions)
    ckpt_dir = tmp_path / "checkpoints"
    
    # Đảm bảo thư mục được tạo ra
    assert ckpt_dir.exists(), "Bug: Thư mục 'checkpoints' chưa được tạo ra."
    
    # Đếm số lượng file .pth trong thư mục
    saved_files = list(ckpt_dir.glob("*.pth"))
    
    # KIỂM TRA QUAN TRỌNG NHẤT: Bắt buộc chỉ có ĐÚNG 1 FILE tồn tại
    assert len(saved_files) == 1, f"Bug Tràn Ổ Cứng: Có {len(saved_files)} files được tạo ra thay vì chỉ 1 file duy nhất!"
    
    # Kiểm tra tên file đúng chuẩn
    assert saved_files[0].name == "lightgcn_purchase_latest.pth", "Bug: Sai tên file checkpoint."
    
    # Đảm bảo file không bị rỗng (có ghi dữ liệu vật lý xuống)
    assert saved_files[0].stat().st_size > 0, "Bug: File checkpoint được tạo nhưng trống rỗng (0 bytes)."