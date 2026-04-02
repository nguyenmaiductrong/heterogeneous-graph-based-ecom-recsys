import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import math

from src.core.contracts import NODE_TYPES, EMBED_DIM

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# 1. Generate Synthetic REES46-like Dataset (dùng test vì chưa truy cập dc vm)

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

# 3. HeteroEmbedding 

class HeteroEmbedding(nn.Module):
    def __init__(
        self,
        num_nodes: dict[str, int],  # {"user": N_u, "product": N_p, "category": N_c, "brand": N_b}
        dim: int = EMBED_DIM,
    ):
        super().__init__()
        self.dim = dim

        # Per-type embedding — mỗi node type có embedding table riêng
        self.embeddings = nn.ModuleDict({
            ntype: nn.Embedding(num_nodes[ntype], dim)
            for ntype in NODE_TYPES
        })

        # Per-type Xavier uniform init — fan_in = fan_out = dim
        a = math.sqrt(6.0 / (dim + dim))   # ≈ 0.1531 với dim=128
        for ntype in NODE_TYPES:
            nn.init.uniform_(self.embeddings[ntype].weight, -a, a)

    def forward(self, node_ids: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Args:
            node_ids: {node_type: Tensor(batch_size,)}
        Returns:
            {node_type: Tensor(batch_size, dim)}
        """
        return {
            ntype: self.embeddings[ntype](ids)
            for ntype, ids in node_ids.items()
        }

    def get_all(self, ntype: str) -> torch.Tensor:
        """Lấy toàn bộ embedding của một node type — dùng lúc eval."""
        num = self.embeddings[ntype].num_embeddings
        return self.embeddings[ntype](
            torch.arange(num, device=self.embeddings[ntype].weight.device)
        )

    def embedding_l2_norm(self) -> torch.Tensor:
        """L2 norm của tất cả embedding weights — dùng cho regularization trong BPRTrainingStep."""
        return sum(
            emb.weight.pow(2).sum()
            for emb in self.embeddings.values()
        )


# 4. Verify Acceptance Criteria

if __name__ == "__main__":
    df = generate_rees46_mini(NUM_USERS, NUM_PRODUCTS, NUM_EVENTS)
    print(f"Dataset shape : {df.shape}")
    print(f"Event dist    :\n{df['event_type'].value_counts(normalize=True).round(3)}")

    df, user2idx, product2idx = preprocess(df)

    num_nodes = {
        "user":     len(user2idx),
        "product":  len(product2idx),
        "category": 50,
        "brand":    30,
    }
    print(f"\nnum_nodes: {num_nodes}")

    print("\n" + "="*55)
    print("ACCEPTANCE CRITERIA CHECK")
    print("="*55)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    model = HeteroEmbedding(num_nodes=num_nodes, dim=128).to(device)

    # Check 1: Shape cho từng node type
    for ntype, n in num_nodes.items():
        sample_ids = torch.randint(0, n, (256,)).to(device)
        out = model.forward({ntype: sample_ids})
        assert out[ntype].shape == (256, 128), f"{ntype} shape sai: {out[ntype].shape}"
        print(f" Shape OK  — {ntype:10s}: {out[ntype].shape}")

    # Check 2: Xavier init stats cho từng type
    a = math.sqrt(6.0 / (128 + 128))
    for ntype in NODE_TYPES:
        w = model.embeddings[ntype].weight
        assert abs(w.mean().item()) < 0.02,     f"{ntype} mean quá lớn"
        assert w.min().item() >= -a - 1e-6,     f"{ntype} có giá trị < -a"
        assert w.max().item() <=  a + 1e-6,     f"{ntype} có giá trị > +a"
        print(f" Init OK  — {ntype:10s}: std={w.std().item():.4f}, "
              f"range=[{w.min().item():.4f}, {w.max().item():.4f}]")

    # Check 3: Gradient flows
    sample_ids = torch.randint(0, num_nodes["user"], (256,)).to(device)
    out = model.forward({"user": sample_ids})
    loss = out["user"].sum()
    loss.backward()
    assert model.embeddings["user"].weight.grad is not None, "Grad user bị None!"
    print(f"\n Gradient — user grad norm: {model.embeddings['user'].weight.grad.norm():.4f}")

    # Check 4: embedding_l2_norm là scalar
    l2 = model.embedding_l2_norm()
    assert l2.dim() == 0, "l2_norm phải là scalar"
    print(f" L2 norm OK: {l2.item():.2f}")

    # Check 5: Memory footprint
    print(f"\n Memory estimate:")
    total_mb = 0.0
    for ntype, n in num_nodes.items():
        mb = n * 128 * 4 / 1024**2
        total_mb += mb
        print(f"   {ntype:10s}: {mb:.2f} MB")
    print(f"   {'total':10s}: {total_mb:.2f} MB")

    print("\n" + "="*55)
    print("Per-type embedding init (Xavier uniform) em đảm bảo giá trị nằm trong khoảng [-a, a] với a ≈ 0.1531, và mean gần 0.")
    print("="*55)
