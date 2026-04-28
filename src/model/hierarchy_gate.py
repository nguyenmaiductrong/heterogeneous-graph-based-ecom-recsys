import torch
import torch.nn as nn


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
