from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Optional, Tuple

NODE_TYPES = ["user", "product", "category", "brand"]

BEHAVIOR_EDGES = [
    ("user", "view", "product"),
    ("user", "cart", "product"),
    ("user", "purchase", "product"),
    ("product", "rev_view", "user"),
    ("product", "rev_cart", "user"),
    ("product", "rev_purchase", "user"),
]

STRUCTURAL_EDGES = [
    ("product", "belongs_to", "category"),
    ("category", "contains", "product"),
    ("product", "producedBy", "brand"),
    ("brand", "brands", "product"),
]

ALL_EDGE_TYPES = BEHAVIOR_EDGES + STRUCTURAL_EDGES

BEHAVIOR_ORIGIN: Dict[str, int] = {"view": 0, "cart": 1, "purchase": 2}

_REL_IDX: Dict[Tuple[str, str, str], int] = {et: i for i, et in enumerate(ALL_EDGE_TYPES)}

_BEH_IDX: Dict[str, int] = {
    "view": 0,
    "cart": 1,
    "purchase": 2,
    "rev_view": 0,
    "rev_cart": 1,
    "rev_purchase": 2,
    "belongs_to": 3,
    "contains": 3,
    "producedBy": 3,
    "brands": 3,
}

_REV_BEH_KEYS = {"rev_view": "view", "rev_cart": "cart", "rev_purchase": "purchase"}


class BehaviorAwareWeight(nn.Module):
    """Bien doi low-rank theo relation va behavior.

    W_{rho,beta} = W_rho + A_rho * diag(z_beta) * B_rho^T

    Tham so:
        rho: chi so relation (loai canh)
        beta: behavior origin (view=0, cart=1, purchase=2, struct=3)
        z_beta: vector scale hoc duoc theo behavior
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        rank: int = 16,
        n_relations: int = len(ALL_EDGE_TYPES),
        n_beta: int = 4,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.n_relations = n_relations
        self.n_beta = n_beta

        # W_ρ: per-relation base weight
        self.W_rho = nn.Parameter(torch.empty(n_relations, out_dim, in_dim))
        # A_ρ: per-relation low-rank factor
        self.A_rho = nn.Parameter(torch.empty(n_relations, out_dim, rank))
        # B_ρ: per-relation low-rank factor
        self.B_rho = nn.Parameter(torch.empty(n_relations, in_dim, rank))
        # z_β: per-behavior scaling (the key difference from old impl)
        self.z_beta = nn.Parameter(torch.empty(n_beta, rank))

        nn.init.kaiming_uniform_(self.W_rho)
        nn.init.kaiming_uniform_(self.A_rho)
        nn.init.kaiming_uniform_(self.B_rho)
        nn.init.ones_(self.z_beta)  # Initialize to 1 for stable start

    def forward(self, rho: int, beta: int) -> Tensor:
        """Compute W_{ρ,β} = W_ρ + A_ρ · diag(z_β) · B_ρᵀ"""
        b_idx = beta if beta >= 0 else 3
        # A_ρ · diag(z_β) = A_ρ * z_β (element-wise broadcast on last dim)
        A_scaled = self.A_rho[rho] * self.z_beta[b_idx]  # [out_dim, rank]
        return self.W_rho[rho] + A_scaled @ self.B_rho[rho].T

class TemporalPurchaseIntentDecoder(nn.Module):
    """Bo giai ma y dinh mua hang theo thoi gian (TPID).

    Ket hop 3 expert:
    - s_graph: diem user-item tu graph
    - s_seq: diem tu chuoi L su kien gan nhat
    - s_pop: diem popularity co time decay

    Trong so fusion tinh qua MLP tren user features.
    """

    def __init__(
        self,
        dim: int,
        n_items: int,
        seq_len: int = 20,
        n_behaviors: int = 3,
        n_freqs: int = 16,
        tau_pop: float = 30.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.n_items = n_items
        self.seq_len = seq_len
        self.tau_pop = tau_pop

        self.beh_emb = nn.Embedding(n_behaviors + 1, dim)
        self.pos_emb = nn.Embedding(seq_len, dim)
        self.time_enc = FourierTimeEncoding(n_freqs)
        self.time_proj = nn.Linear(n_freqs * 2, dim)

        self.seq_encoder = nn.GRU(dim, dim, batch_first=True)

        self.item_bias = nn.Embedding(n_items, 1)

        self.raw_eta = nn.Parameter(torch.zeros(n_behaviors))
        self.raw_kappa = nn.Parameter(torch.zeros(n_behaviors))

        self.fusion_mlp = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
        )

    def encode_sequence(
        self,
        item_seq: Tensor,
        beh_seq: Tensor,
        ts_seq: Tensor,
        ref_time: float,
        item_emb: Tensor,
    ) -> Tensor:
        """Encode user sequence into a single vector.

        Args:
            item_seq: [B, L] item indices
            beh_seq: [B, L] behavior indices
            ts_seq: [B, L] timestamps
            ref_time: reference time T
            item_emb: [N, dim] item embeddings

        Returns:
            z_u: [B, dim] user sequence embedding
        """
        B, L = item_seq.shape
        device = item_seq.device

        e_item = item_emb[item_seq.clamp(0, item_emb.size(0) - 1)]
        e_beh = self.beh_emb(beh_seq.clamp(0, 3))
        e_pos = self.pos_emb(torch.arange(L, device=device).unsqueeze(0).expand(B, -1))

        delta_t = (ref_time - ts_seq.float()) / 86400.0
        delta_t = delta_t.clamp(min=0)
        phi = self.time_enc(delta_t.view(-1)).view(B, L, -1)
        e_time = self.time_proj(phi)

        x = e_item + e_beh + e_pos + e_time

        mask = item_seq >= 0
        x = x * mask.unsqueeze(-1).float()

        _, h_n = self.seq_encoder(x)
        return h_n.squeeze(0)