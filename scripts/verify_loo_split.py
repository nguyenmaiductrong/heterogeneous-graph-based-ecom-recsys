import os
import time
import argparse
import pandas as pd
import pickle

def verify_and_build_mask(data_dir: str):

    t0 = time.time()
    mask_path = os.path.join(data_dir, "train_mask.pkl")

    try:
        val_df = pd.read_parquet(os.path.join(data_dir, "_val_pairs_parquet"))
        test_df = pd.read_parquet(os.path.join(data_dir, "_test_pairs_parquet"))
    except Exception as e:
        print(f"Lỗi: Không thể đọc file Parquet. Chi tiết: {e}")
        return

    n_val = len(val_df)
    n_test = len(test_df)
    
    # Kiểm tra tính duy nhất của user (Mỗi user chỉ có đúng 1 pair trong LOO)
    is_val_unique = val_df['user_idx'].is_unique
    is_test_unique = test_df['user_idx'].is_unique

    print(f"  - Số lượng Val pairs:  {n_val:,}")
    print(f"  - Số lượng Test pairs: {n_test:,}")
    
    if is_val_unique and is_test_unique:
        print("Verify: Mỗi user có ĐÚNG 1 test pair và 1 val pair.")
    else:
        print(f"ANOMALY: Phát hiện user bị lặp trong tập Eval!")
        return 

    
    # Đọc tập train tổng hợp (bao gồm cả view, cart, purchase đã split)
    train_df = pd.read_parquet(os.path.join(data_dir, "_train_all_parquet"))
    print(f"Tổng số tương tác trong tập Train: {len(train_df):,}")

    # Chỉ giữ lại interactions của những user có trong tập Eval (1,052,774 users)
    eval_users = set(val_df['user_idx'].unique())
    
    print("  - Đang gom nhóm (groupby) và tạo dictionary mask...")
    # Lọc và groupby để lấy danh sách item đã tương tác
    train_mask = (train_df[train_df['user_idx'].isin(eval_users)]
                  .groupby('user_idx')['product_idx']
                  .apply(set)
                  .to_dict())
    
    print("Đang lưu file train_mask.pkl...")
    with open(mask_path, 'wb') as f:
        pickle.dump(train_mask, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    print(f"Đã lưu thành công tại: {mask_path}")
  
    actual_len = len(train_mask)
    expected_len = 1052774
    
    if actual_len == expected_len:
        print(f"len(train_mask) == {expected_len:,}: OK")
    else:
        print(f"len(train_mask): FAILED (Thực tế: {actual_len:,})")
            
    print(f"\nHOÀN TẤT (Tổng thời gian: {time.time() - t0:.2f}s)")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    args = parser.parse_args()
    
    verify_and_build_mask(args.data_dir)