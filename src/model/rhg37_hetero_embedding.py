import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import math

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# 1. Generate Synthetic REES46-like Dataset (dùng test RHG-36 vì chưa truy cập dc vm)

NUM_USERS    = 5_000
NUM_PRODUCTS = 2_000
NUM_EVENTS   = 50_000
EVENT_TYPES  = ["view", "cart", "purchase"]
# REES46 thực tế: view ~90%, cart ~7%, purchase ~3%
EVENT_WEIGHTS = [0.90, 0.07, 0.03]

def generate_rees46_mini(num_users, num_products, num_events):
    base_time = datetime(2024, 1, 1)

    records = []
    for i in range(num_events):
        user_id    = random.randint(1, num_users)
        product_id = random.randint(1, num_products)
        event_type = random.choices(EVENT_TYPES, weights=EVENT_WEIGHTS)[0]
        event_time = base_time + timedelta(seconds=random.randint(0, 30*24*3600))

        records.append({
            "event_time"    : event_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "event_type"    : event_type,
            "product_id"    : product_id,
            "category_id"   : random.randint(1, 50),
            "category_code" : f"category.sub{random.randint(1,10)}",
            "brand"         : f"brand_{random.randint(1,30)}",
            "price"         : round(random.uniform(5.0, 500.0), 2),
            "user_id"       : user_id,
            "user_session"  : f"sess_{random.randint(1, 20000)}",
        })

    df = pd.DataFrame(records)
    df["event_time"] = pd.to_datetime(df["event_time"])
    df = df.sort_values("event_time").reset_index(drop=True)
    return df

# 2. Preprocess — tạo contiguous ID mapping

def preprocess(df):
    user2idx    = {uid: idx for idx, uid in enumerate(df["user_id"].unique())}
    product2idx = {pid: idx for idx, pid in enumerate(df["product_id"].unique())}

    df["user_idx"]    = df["user_id"].map(user2idx)
    df["product_idx"] = df["product_id"].map(product2idx)

    return df, user2idx, product2idx

def get_behavior_pairs(df, event_type):
    sub = df[df["event_type"] == event_type][["user_idx", "product_idx"]].values
    return torch.tensor(sub, dtype=torch.long)

# 3. HeteroEmbedding (RHG-37) 

class HeteroEmbedding(nn.Module):
    def __init__(self, num_users: int, num_products: int, dim: int = 128):
        super().__init__()
        self.dim = dim

        self.user_emb    = nn.Embedding(num_users,    dim)
        self.product_emb = nn.Embedding(num_products, dim)

        # Xavier uniform: fan_in = fan_out = dim
        a = math.sqrt(6.0 / (dim + dim))   # ≈ 0.1531 với dim=128
        nn.init.uniform_(self.user_emb.weight,    -a, a)
        nn.init.uniform_(self.product_emb.weight, -a, a)

    def forward(self, user_ids, product_ids):
        return self.user_emb(user_ids), self.product_emb(product_ids)

    def get_all_users(self):
        return self.user_emb(torch.arange(self.user_emb.num_embeddings))

    def get_all_products(self):
        return self.product_emb(torch.arange(self.product_emb.num_embeddings))


# 4. Verify Acceptance Criteria 

if __name__ == "__main__":
    df = generate_rees46_mini(NUM_USERS, NUM_PRODUCTS, NUM_EVENTS)
    print(f"Dataset shape : {df.shape}")
    print(f"Event dist    :\n{df['event_type'].value_counts(normalize=True).round(3)}")
    print(df.head(3))

    df, user2idx, product2idx = preprocess(df)

    NUM_USERS_ACTUAL    = len(user2idx)
    NUM_PRODUCTS_ACTUAL = len(product2idx)
    print(f"\nUnique users    : {NUM_USERS_ACTUAL}")
    print(f"Unique products : {NUM_PRODUCTS_ACTUAL}")

    pairs_view     = get_behavior_pairs(df, "view")
    pairs_cart     = get_behavior_pairs(df, "cart")
    pairs_purchase = get_behavior_pairs(df, "purchase")
    print(f"\nBehavior counts — view: {len(pairs_view)}, "
          f"cart: {len(pairs_cart)}, purchase: {len(pairs_purchase)}")

    print("\n" + "="*55)
    print("ACCEPTANCE CRITERIA CHECK")
    print("="*55)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    model = HeteroEmbedding(
        num_users=NUM_USERS_ACTUAL,
        num_products=NUM_PRODUCTS_ACTUAL,
        dim=128
    ).to(device)

    BATCH_SIZE = 256
    batch_idx  = torch.randint(0, len(pairs_view), (BATCH_SIZE,))
    batch      = pairs_view[batch_idx].to(device)
    user_ids   = batch[:, 0]
    prod_ids   = batch[:, 1]

    u_emb, p_emb = model(user_ids, prod_ids)

    #  Check 1: Shape
    assert u_emb.shape == (BATCH_SIZE, 128), f"user emb shape sai: {u_emb.shape}"
    assert p_emb.shape == (BATCH_SIZE, 128), f"product emb shape sai: {p_emb.shape}"
    print(f" Shape OK  — user: {u_emb.shape}, product: {p_emb.shape}")

    #  Check 2: Init stats hợp lý (Xavier)
    for name, weight in [("user",    model.user_emb.weight),
                         ("product", model.product_emb.weight)]:
        mean = weight.mean().item()
        std  = weight.std().item()
        a    = math.sqrt(6.0 / (128 + 128))
        assert abs(mean) < 0.01,                    f"{name} mean={mean:.4f} quá lớn"
        assert 0.07 < std < 0.11,                   f"{name} std={std:.4f} ngoài range"
        assert weight.min().item() >= -a - 1e-6,    f"{name} có giá trị < -a"
        assert weight.max().item() <=  a + 1e-6,    f"{name} có giá trị >  a"
        print(f" Init OK — {name:8s}: mean={mean:+.4f}, std={std:.4f}, "
              f"range=[{weight.min().item():.4f}, {weight.max().item():.4f}]")

    #  Check 3: Gradient flows
    loss = u_emb.sum() + p_emb.sum()
    loss.backward()
    assert model.user_emb.weight.grad    is not None, "Grad user bị None!"
    assert model.product_emb.weight.grad is not None, "Grad product bị None!"
    print(f" Gradient — user grad norm:    {model.user_emb.weight.grad.norm():.4f}")
    print(f" Gradient — product grad norm: {model.product_emb.weight.grad.norm():.4f}")

    # 5. Memory footprint
    user_mb    = NUM_USERS_ACTUAL    * 128 * 4 / 1024**2
    product_mb = NUM_PRODUCTS_ACTUAL * 128 * 4 / 1024**2
    print(f"\n📦 Memory estimate:")
    print(f"   user_emb    : {user_mb:.2f} MB")
    print(f"   product_emb : {product_mb:.2f} MB")
    print(f"   total       : {user_mb + product_mb:.2f} MB")

    print("\n" + "="*55)
    print("RHG-37 PASSED — Sẵn sàng cho RHG-36 (HierarchyGate)")
    print("="*55)
