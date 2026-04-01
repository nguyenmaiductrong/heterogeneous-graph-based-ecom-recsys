import os
import time
import argparse
import pickle
from collections import defaultdict

import numpy as np


def verify_and_build_mask(data_dir: str) -> None:
    t0 = time.time()
    mask_path = os.path.join(data_dir, "train_mask.pkl")

    print("Đang kiểm tra Validation và Test splits...")

    try:
        val_users = np.load(os.path.join(data_dir, "val_user_idx.npy"))
        test_users = np.load(os.path.join(data_dir, "test_user_idx.npy"))
    except FileNotFoundError as e:
        print(f"Lỗi: Không tìm thấy file numpy. Chi tiết: {e}")
        return

    val_unique_users = np.unique(val_users)
    test_unique_users = np.unique(test_users)

    val_anomaly = len(val_users) - len(val_unique_users)
    test_anomaly = len(test_users) - len(test_unique_users)

    print(f"  - Số lượng Val pairs:  {len(val_users):,}")
    print(f"  - Số lượng Test pairs: {len(test_users):,}")

    if val_anomaly == 0 and test_anomaly == 0:
        print("Verify: Mỗi user có đúng 1 test pair và 1 val pair.")
    else:
        print(
            f"ANOMALY DETECTED: {val_anomaly} users bị lặp trong val, "
            f"{test_anomaly} users bị lặp trong test."
        )
        return

    print("Đang build train_mask từ các file train_src và train_dst...")

    view_u = np.load(os.path.join(data_dir, "loo_view_train_src.npy"))
    view_i = np.load(os.path.join(data_dir, "loo_view_train_dst.npy"))

    cart_u = np.load(os.path.join(data_dir, "loo_cart_train_src.npy"))
    cart_i = np.load(os.path.join(data_dir, "loo_cart_train_dst.npy"))

    pur_u = np.load(os.path.join(data_dir, "loo_purchase_train_src.npy"))
    pur_i = np.load(os.path.join(data_dir, "loo_purchase_train_dst.npy"))

    full_train_mask: defaultdict[int, set[int]] = defaultdict(set)
    total_edges = 0
    for u_arr, i_arr in ((view_u, view_i), (cart_u, cart_i), (pur_u, pur_i)):
        total_edges += len(u_arr)
        for u, i in zip(u_arr, i_arr):
            full_train_mask[int(u)].add(int(i))

    print(f"- Tổng số lượng tương tác (edges) trong tập train: {total_edges:,}")
    print("- Đang gom nhóm các tương tác thành set (defaultdict)...")

    eval_users_set = set(val_users.tolist())
    train_mask = {
        u: items for u, items in full_train_mask.items() if u in eval_users_set
    }
    del full_train_mask

    with open(mask_path, "wb") as f:
        pickle.dump(train_mask, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Đã lưu tại: {mask_path}")

    try:
        with open(mask_path, "rb") as f:
            loaded_mask = pickle.load(f)
        print("train_mask.pkl load được: OK")

        loaded_len = len(loaded_mask)
        if loaded_len == 1_052_774:
            print("len(train_mask) == 1,052,774: OK")
        else:
            print(
                f"len(train_mask) == 1,052,774: FAILED (Thực tế: {loaded_len:,})"
            )

        print(f"\nStats printed, no anomaly. (Tổng thời gian: {time.time() - t0:.2f}s)")

    except Exception as e:
        print(f"Lỗi khi load file pkl: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify LOO Split and Build Train Mask for Numpy arrays"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Thư mục chứa các file .npy (ví dụ: data/mnt/data/rees46/processed/loo)",
    )
    args = parser.parse_args()

    verify_and_build_mask(args.data_dir)
