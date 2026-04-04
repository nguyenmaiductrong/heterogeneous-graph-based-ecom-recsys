import os
import pandas as pd
import pickle
import pytest
from scripts.verify_loo_split import verify_and_build_mask

def test_verify_and_build_mask_logic(tmp_path):

    data_dir = tmp_path / "loo"
    data_dir.mkdir()

    val_df = pd.DataFrame({
        'user_idx': [1, 2], 
        'product_idx': [10, 20]
    })
    val_df.to_parquet(data_dir / "_val_pairs_parquet")

    test_df = pd.DataFrame({
        'user_idx': [1, 2], 
        'product_idx': [11, 21]
    })
    test_df.to_parquet(data_dir / "_test_pairs_parquet")

    train_df = pd.DataFrame({
        'user_idx': [1, 1, 2, 2, 99],
        'product_idx': [101, 102, 201, 202, 901],
        'event_type': ['view', 'purchase', 'cart', 'purchase', 'view'],
        'unix_ts': [1000, 1001, 1002, 1003, 1004]
    })
    train_df.to_parquet(data_dir / "_train_all_parquet")

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