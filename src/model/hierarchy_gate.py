import torch
import torch.nn as nn
import math

from .hetero_embedding import (
    HeteroEmbedding,
    generate_rees46_mini,
    preprocess,
    get_behavior_pairs,
    NUM_USERS,
    NUM_PRODUCTS,
    NUM_EVENTS,
)

torch.manual_seed(42)

#  HierarchyGate  

class HierarchyGate(nn.Module):
    """
    Soft gating module fuse 3 behavior embeddings theo thứ tự
    ưu tiên purchase > cart > view.

    Args:
        dim : embedding dimension (phải match HeteroEmbedding.dim)
    """
    def __init__(self, dim: int = 128):
        super().__init__()
        self.dim = dim

        # MLP nhận concat của 3 embedding → 3 gate scores
        # Input: [B, dim*3], Output: [B, 3]
        self.gate_mlp = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.ReLU(),
            nn.Linear(dim, 3),
        )

        # Init weights
        for layer in self.gate_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(
        self,
        emb_view:     torch.Tensor,   # [B, dim]
        emb_cart:     torch.Tensor,   # [B, dim]
        emb_purchase: torch.Tensor,   # [B, dim]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            fused_emb   : [B, dim]  — weighted combination của 3 emb
            gate_weights: [B, 3]    — (w_view, w_cart, w_purchase), sum=1
        """
        # 1. Concat 3 embedding làm input cho MLP
        concat = torch.cat([emb_view, emb_cart, emb_purchase], dim=-1)  # [B, dim*3]

        # 2. MLP → raw scores → softmax
        scores       = self.gate_mlp(concat)               # [B, 3]
        gate_weights = torch.softmax(scores, dim=-1)       # [B, 3], sum=1

        # 3. Weighted sum — purchase bias được học qua data
        w_v = gate_weights[:, 0:1]   # [B, 1]
        w_c = gate_weights[:, 1:2]   # [B, 1]
        w_p = gate_weights[:, 2:3]   # [B, 1]

        fused_emb = w_v * emb_view + w_c * emb_cart + w_p * emb_purchase  # [B, dim]

        return fused_emb, gate_weights


# Verify Acceptance Criteria 

if __name__ == "__main__":
    # Setup data từ RHG-37
    import numpy as np
    import random
    np.random.seed(42)
    random.seed(42)

    df = generate_rees46_mini(NUM_USERS, NUM_PRODUCTS, NUM_EVENTS)
    df, user2idx, product2idx = preprocess(df)

    NUM_USERS_ACTUAL    = len(user2idx)
    NUM_PRODUCTS_ACTUAL = len(product2idx)

    pairs_view     = get_behavior_pairs(df, "view")
    pairs_cart     = get_behavior_pairs(df, "cart")
    pairs_purchase = get_behavior_pairs(df, "purchase")

    print("=" * 55)
    print("ACCEPTANCE CRITERIA CHECK — HierarchyGate")
    print("=" * 55)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    emb_model  = HeteroEmbedding(
        num_nodes={"user": NUM_USERS_ACTUAL, "product": NUM_PRODUCTS_ACTUAL, "category": 50, "brand": 30},
        dim=128,
    ).to(device)
    gate_model = HierarchyGate(dim=128).to(device)

    BATCH = 256

    def sample_batch(pairs, batch_size, device):
        idx = torch.randint(0, len(pairs), (batch_size,))
        b   = pairs[idx].to(device)
        return b[:, 0], b[:, 1]

    u_view, p_view     = sample_batch(pairs_view,     BATCH, device)
    u_cart, p_cart     = sample_batch(pairs_cart,     BATCH, device)
    u_purch, p_purch   = sample_batch(pairs_purchase, BATCH, device)

    emb_view     = emb_model({"user": u_view} )["user"]
    emb_cart     = emb_model({"user": u_cart} )["user"]
    emb_purchase = emb_model({"user": u_purch})["user"]

    fused_emb, gate_weights = gate_model(emb_view, emb_cart, emb_purchase)

    #  Check 1: forward() → fused emb shape đúng
    assert fused_emb.shape == (BATCH, 128), f"fused shape sai: {fused_emb.shape}"
    print(f" forward()    — fused_emb shape: {fused_emb.shape}")

    #  Check 2: gate weights sum = 1
    weight_sums = gate_weights.sum(dim=-1)
    assert torch.allclose(weight_sums, torch.ones(BATCH, device=device), atol=1e-6), \
        f"gate sum sai: min={weight_sums.min():.6f}, max={weight_sums.max():.6f}"
    print(f" Gate sum=1   — min={weight_sums.min():.6f}, max={weight_sums.max():.6f}")

    mean_w = gate_weights.mean(dim=0)
    print(f"   Gate mean    — view={mean_w[0]:.4f}, cart={mean_w[1]:.4f}, purchase={mean_w[2]:.4f}")
    print(f"   (random init nên ≈ 0.333 mỗi loại, sau training purchase sẽ cao hơn)")

    #  Check 3: Gradient flows qua cả gate_model lẫn emb_model
    loss = fused_emb.sum()
    loss.backward()

    assert gate_model.gate_mlp[0].weight.grad is not None, "Grad MLP layer 0 bị None!"
    assert gate_model.gate_mlp[2].weight.grad is not None, "Grad MLP layer 2 bị None!"
    assert emb_model.embeddings["user"].weight.grad is not None, "Grad user_emb bị None!"
    print(f" Gradient    — MLP L0 grad norm: {gate_model.gate_mlp[0].weight.grad.norm():.4f}")
    print(f" Gradient    — MLP L2 grad norm: {gate_model.gate_mlp[2].weight.grad.norm():.4f}")
    print(f" Gradient    — user_emb grad norm: {emb_model.embeddings['user'].weight.grad.norm():.4f}")

    print("\n" + "=" * 55)
    print("HierarchyGate sẵn sàng, em đảm bảo forward đúng shape, gate weights sum=1, và gradient flow qua cả gate lẫn embedding")
    print("=" * 55)
