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