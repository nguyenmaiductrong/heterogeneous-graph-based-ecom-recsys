import os
import numpy as np
import pickle
import pytest
from src.training.lightgcn_baseline import train

def test_checkpoint_rolling_mechanism(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    train_u = np.array([0, 1, 2, 3], dtype=np.int64)
    train_i = np.array([0, 1, 2, 3], dtype=np.int64) # Lưu ý: num_items sẽ là max+1
    np.save(data_dir / "loo_purchase_train_src.npy", train_u)
    np.save(data_dir / "loo_purchase_train_dst.npy", train_i)

    checkpoint_root = tmp_path / "checkpoints"
    from src.training.lightgcn_baseline import LightGCNConfig, train
    
    cfg = LightGCNConfig(
        data_dir=str(data_dir),
        checkpoint_dir=str(checkpoint_root), 
        epochs=20,       # Chạy 20 epoch
        save_every=10,   # Lưu ở epoch 10 và 20
        keep_top_k=1,    # Ép chỉ giữ 1 file duy nhất để khớp với assert của bạn
        batch_size=2
    )

    train(cfg)

    # Kiểm tra
    assert checkpoint_root.exists(), "Thư mục checkpoint không tồn tại"
    
    saved_files = sorted(list(checkpoint_root.glob("lightgcn_epoch_*.pth")))
    
    assert len(saved_files) == 1, f"Tìm thấy {len(saved_files)} files, mong đợi 1."
    

    assert "lightgcn_epoch_0020.pth" in saved_files[0].name
    assert saved_files[0].stat().st_size > 0