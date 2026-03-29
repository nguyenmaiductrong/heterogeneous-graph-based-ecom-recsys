import os
import numpy as np
import pickle
import pytest
from scripts.verify_loo_split import verify_and_build_mask

def test_verify_and_build_mask_logic(tmp_path):
    """
    Test logic đọc file numpy, gộp các hành vi, và tạo train_mask.pkl.
    Đặc biệt test việc loại bỏ các user cold-start.
    """
    data_dir = tmp_path / "loo"
    data_dir.mkdir()

    val_users = np.array([1, 2])
    test_users = np.array([1, 2])
    np.save(data_dir / "val_user_idx.npy", val_users)
    np.save(data_dir / "test_user_idx.npy", test_users)

    np.save(data_dir / "loo_view_train_src.npy", np.array([1, 99]))
    np.save(data_dir / "loo_view_train_dst.npy", np.array([101, 901]))

    np.save(data_dir / "loo_cart_train_src.npy", np.array([1, 2]))
    np.save(data_dir / "loo_cart_train_dst.npy", np.array([102, 201]))

    np.save(data_dir / "loo_purchase_train_src.npy", np.array([2]))
    np.save(data_dir / "loo_purchase_train_dst.npy", np.array([202]))

    verify_and_build_mask(str(data_dir))

    mask_path = data_dir / "train_mask.pkl"
    assert mask_path.exists(), "Lỗi: Script không tạo ra được file train_mask.pkl"

    with open(mask_path, 'rb') as f:
        train_mask = pickle.load(f)

    assert len(train_mask) == 2, f"Mask chứa {len(train_mask)} users, mong đợi chỉ có 2 core users."
    assert 1 in train_mask
    assert train_mask[1] == {101, 102}, "Gộp hành vi cho User 1 bị sai"
    assert 2 in train_mask
    assert train_mask[2] == {201, 202}, "Gộp hành vi cho User 2 bị sai"
    assert 99 not in train_mask, "Lỗi nghiêm trọng: User cold-start chưa bị lọc bỏ khỏi train_mask!"