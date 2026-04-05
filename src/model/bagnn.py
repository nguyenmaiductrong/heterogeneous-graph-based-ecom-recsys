"""
BAGNN — Behavior-Aware Graph Neural Network
============================================
Node types  : user, product, category, brand
Edge types  : view, cart, purchase (+ reverses), belongs_to, contains, producedBy, brands

Core novelty: W(φ, β) = W_base + A_φ @ B_β.T
  - Structural edges (belongs_to, producedBy) inherit β from upstream user behavior
  - Per-edge beta computed via einsum to avoid O(E·d·d) materialization
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from torch_geometric.data import HeteroData
from torch_geometric.utils import softmax
from typing import Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBED_DIM: int = 128

NODE_TYPES = ["user", "product", "category", "brand"]

BEHAVIOR_EDGES = [
    ("user",    "view",         "product"),
    ("user",    "cart",         "product"),
    ("user",    "purchase",     "product"),
    ("product", "rev_view",     "user"),
    ("product", "rev_cart",     "user"),
    ("product", "rev_purchase", "user"),
]

STRUCTURAL_EDGES = [
    ("product",  "belongs_to", "category"),
    ("category", "contains",   "product"),
    ("product",  "producedBy", "brand"),
    ("brand",    "brands",     "product"),
]

ALL_EDGE_TYPES = BEHAVIOR_EDGES + STRUCTURAL_EDGES

BEHAVIOR_ORIGIN: Dict[str, int] = {"view": 0, "cart": 1, "purchase": 2}

# Precomputed for O(1) lookup in forward pass
_REL_IDX: Dict[Tuple[str, str, str], int] = {et: i for i, et in enumerate(ALL_EDGE_TYPES)}

# β index for each edge name: behavior edges → 0/1/2; structural/reverse → 3
_BEH_IDX: Dict[str, int] = {
    "view": 0, "cart": 1, "purchase": 2,
    "rev_view": 0, "rev_cart": 1, "rev_purchase": 2,
    "belongs_to": 3, "contains": 3, "producedBy": 3, "brands": 3,
}


# ---------------------------------------------------------------------------
# N1.1 — Low-rank behavior-aware weight decomposition
# W(φ, β) = W_base + A_φ @ B_β.T
# ---------------------------------------------------------------------------

class BehaviorAwareWeight(nn.Module):
    """
    W(φ, β) = W_base  +  A_φ  @  B_β.T

    β ∈ {0=view, 1=cart, 2=purchase, 3=structural}
    φ ∈ {0=user, 1=product, 2=category, 3=brand}
    """

    def __init__(
        self,
        in_dim:  int,
        out_dim: int,
        rank:    int = 16,
        n_phi:   int = len(NODE_TYPES),
        n_beta:  int = 4,              # 0-2 behavior + 3 structural
    ) -> None:
        super().__init__()
        self.W_base = nn.Parameter(torch.empty(out_dim, in_dim))
        self.A = nn.Parameter(torch.empty(n_phi, out_dim, rank))
        self.B = nn.Parameter(torch.empty(n_beta, in_dim, rank))
        nn.init.kaiming_uniform_(self.W_base)
        nn.init.kaiming_uniform_(self.A)
        nn.init.kaiming_uniform_(self.B)

    def forward(self, phi: int, beta: int) -> Tensor:
        """Returns W(φ, β) of shape (out_dim, in_dim). beta=-1 maps to β=3 (structural)."""
        b_idx = beta if beta >= 0 else 3
        return self.W_base + self.A[phi] @ self.B[b_idx].T


# ---------------------------------------------------------------------------
# N1.2 — BAGNNConv: one heterogeneous convolution step
# ---------------------------------------------------------------------------

class BAGNNConv(nn.Module):
    """
    Single BAGNN convolution layer.

    For behavior edges: uniform β per edge type.
    For structural edges: per-edge β from edge_attr (behavior_origin inherited
      from hop-1 sampling), enabling W(φ=product, β=view) vs W(φ=product, β=purchase).
    """

    def __init__(
        self,
        in_dim:      int = EMBED_DIM,
        out_dim:     int = EMBED_DIM,
        rank:        int = 16,
        n_relations: int = len(ALL_EDGE_TYPES),
    ) -> None:
        super().__init__()
        self.in_dim  = in_dim
        self.out_dim = out_dim

        self.baw     = BehaviorAwareWeight(in_dim, out_dim, rank)
        self.rel_emb = nn.Embedding(n_relations, out_dim)
        self.beh_emb = nn.Embedding(4, out_dim)      # 0-2 behavior + 3 structural
        self.a_att   = nn.Parameter(torch.empty(4 * out_dim))
        self.norm    = nn.LayerNorm(out_dim)
        nn.init.xavier_uniform_(self.a_att.unsqueeze(0))

    def forward(
        self,
        x_dict:          Dict[str, Tensor],
        edge_index_dict: Dict[Tuple, Tensor],
        edge_attr_dict:  Optional[Dict[Tuple, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        all_r_emb = self.rel_emb.weight   # (n_relations, out_dim)
        all_b_emb = self.beh_emb.weight   # (4, out_dim)

        agg: Dict[str, Optional[Tensor]] = {t: None for t in NODE_TYPES}

        for edge_type, edge_index in edge_index_dict.items():
            src_type, edge_name, dst_type = edge_type

            if src_type not in x_dict or dst_type not in x_dict:
                continue
            if edge_index.numel() == 0:
                continue

            src_idx, dst_idx = edge_index
            h_src = x_dict[src_type][src_idx]
            h_dst = x_dict[dst_type][dst_idx]

            phi      = NODE_TYPES.index(src_type)
            beta_raw = BEHAVIOR_ORIGIN.get(edge_name.removeprefix("rev_"), -1)
            attr     = (edge_attr_dict or {}).get(edge_type)

            if beta_raw < 0 and attr is not None and attr.numel() > 0:
                # Structural edge with per-edge behavior_origin tags
                origin = attr.view(-1).long().clamp(0, 2)

                base_msg   = torch.einsum("oi,ei->eo", self.baw.W_base, h_src)
                A_phi      = self.baw.A[phi]                              # (out_dim, rank)
                # Avoid materializing (E, in_dim, rank) — compute mid per-group
                mid = torch.zeros(origin.size(0), self.baw.B.size(-1),
                                  device=h_src.device, dtype=h_src.dtype)
                for b_idx in origin.unique():
                    mask = origin == b_idx
                    mid[mask] = (h_src[mask] @ self.baw.B[b_idx]).to(mid.dtype)  # (E_b, rank)
                msg        = base_msg + mid @ A_phi.T

                b_emb      = all_b_emb[origin]                        # (E, d) — per-edge
                h_dst_proj = torch.einsum("oi,ei->eo", self.baw.W_base, h_dst)
            else:
                # Behavior edge or structural edge without attr: uniform β
                beta_idx   = beta_raw if beta_raw >= 0 else 3
                W          = self.baw(phi, beta_raw)
                msg        = torch.einsum("oi,ei->eo", W, h_src)
                b_emb      = None                                      # uniform; handled via scalar below
                h_dst_proj = F.linear(h_dst, W)

            # Fused attention: avoid materializing (E, 4*d) concat
            # r_emb and possibly b_emb are uniform per edge-type → reduce to scalars
            d = self.out_dim
            a0, a1, a2, a3 = (self.a_att[:d], self.a_att[d:2*d],
                               self.a_att[2*d:3*d], self.a_att[3*d:])

            r_scalar = (all_r_emb[_REL_IDX[edge_type]] * a2).sum()   # scalar

            if beta_raw < 0 and attr is not None and attr.numel() > 0:
                # per-edge b_emb
                b_term = (b_emb * a3).sum(-1)                         # (E,)
            else:
                b_term = (all_b_emb[beta_idx] * a3).sum()             # scalar

            e = ((msg * a0).sum(-1)
                 + (h_dst_proj * a1).sum(-1)
                 + r_scalar
                 + b_term)
            alpha = softmax(e, dst_idx)

            weighted = alpha.unsqueeze(-1) * msg
            N_dst = x_dict[dst_type].size(0)
            if agg[dst_type] is None:
                agg[dst_type] = weighted.new_zeros(N_dst, self.out_dim)
            agg[dst_type].scatter_add_(0, dst_idx.unsqueeze(-1).expand_as(weighted), weighted)

        out_dict: Dict[str, Tensor] = {}
        for t in NODE_TYPES:
            if agg[t] is not None:
                h = self.norm(agg[t])
                if t in x_dict and x_dict[t].shape == h.shape:
                    h = h + x_dict[t]
                out_dict[t] = F.elu(h)
            elif t in x_dict:
                out_dict[t] = x_dict[t]

        return out_dict


# ---------------------------------------------------------------------------
# N1.3 — BAGNNLayer stack + BAGNNModel
# ---------------------------------------------------------------------------

class BAGNNLayer(nn.Module):
    """Stacks `n_layers` BAGNNConv layers with dropout between layers."""

    def __init__(
        self,
        in_dim:   int = EMBED_DIM,
        hid_dim:  int = EMBED_DIM,
        out_dim:  int = EMBED_DIM,
        n_layers: int = 2,
        rank:     int = 16,
        dropout:  float = 0.1,
    ) -> None:
        super().__init__()
        assert n_layers >= 1
        dims = [in_dim] + [hid_dim] * (n_layers - 1) + [out_dim]
        self.convs   = nn.ModuleList([
            BAGNNConv(in_dim=dims[i], out_dim=dims[i + 1], rank=rank)
            for i in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_dict:          Dict[str, Tensor],
        edge_index_dict: Dict[Tuple, Tensor],
        edge_attr_dict:  Optional[Dict[Tuple, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        h = x_dict
        for i, conv in enumerate(self.convs):
            # Gradient checkpointing: recompute activations during backward
            # instead of storing them, halving peak activation memory.
            if self.training:
                # checkpoint requires tensor inputs; pass node tensors explicitly
                node_types = list(h.keys())
                node_vals  = [h[t] for t in node_types]

                def _conv(*vals, _conv=conv, _nt=node_types,
                          _eid=edge_index_dict, _ead=edge_attr_dict):
                    x = dict(zip(_nt, vals))
                    return list(_conv(x, _eid, _ead).values())

                out_vals = grad_checkpoint(_conv, *node_vals, use_reentrant=False)
                h = dict(zip(node_types, out_vals))
            else:
                h = conv(h, edge_index_dict, edge_attr_dict)
            if i < len(self.convs) - 1:
                h = {t: self.dropout(v) for t, v in h.items()}
        return h


class BAGNNModel(nn.Module):
    """
    Full BAGNN model for heterogeneous e-commerce recommendation.

    Architecture
    ------------
    1. Node-type input projections  (raw feat → embed_dim)
    2. BAGNNLayer stack             (behavior-aware message passing)
    3. Dot-product decoder          score(u, i) = h_u · h_i
    """

    def __init__(
        self,
        feat_dims: Optional[Dict[str, int]] = None,
        n_nodes:   Optional[Dict[str, int]] = None,
        embed_dim: int   = EMBED_DIM,
        n_layers:  int   = 2,
        rank:      int   = 16,
        dropout:   float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        self.input_proj: nn.ModuleDict = nn.ModuleDict()
        for t in NODE_TYPES:
            if feat_dims and t in feat_dims and feat_dims[t] is not None:
                self.input_proj[t] = nn.Linear(feat_dims[t], embed_dim)
            else:
                n = n_nodes[t] if n_nodes and t in n_nodes else 1000
                self.input_proj[t] = nn.Embedding(n, embed_dim)

        self.bagnn = BAGNNLayer(
            in_dim=embed_dim, hid_dim=embed_dim, out_dim=embed_dim,
            n_layers=n_layers, rank=rank, dropout=dropout,
        )

    def encode(self, data: HeteroData) -> Dict[str, Tensor]:
        device = next(self.parameters()).device
        x_dict: Dict[str, Tensor] = {}
        for t in NODE_TYPES:
            if t not in data.node_types:
                continue
            proj   = self.input_proj[t]
            node_x = data[t].x
            if isinstance(proj, nn.Embedding):
                ids = node_x.to(device) if (node_x is not None and node_x.dtype == torch.long) \
                      else torch.arange(data[t].num_nodes, device=device)
                x_dict[t] = proj(ids)
            else:
                x_dict[t] = proj(node_x.to(device=device, dtype=torch.float))

        valid = set(ALL_EDGE_TYPES)
        edge_index_dict = {et: data[et].edge_index for et in data.edge_types if et in valid}

        edge_attr_dict: Dict[Tuple, Tensor] = {
            et: data[et].edge_attr
            for et in data.edge_types
            if et in valid
            and getattr(data[et], "edge_attr", None) is not None
            and data[et].edge_attr.numel() > 0
        }

        return self.bagnn(x_dict, edge_index_dict, edge_attr_dict or None)

    def forward(self, data: HeteroData) -> Tuple[Tensor, Tensor]:
        """Returns (user_emb, item_emb): (N_u, d), (N_i, d)."""
        h = self.encode(data)
        user_emb = h.get("user",    torch.empty(0, self.embed_dim, device=next(self.parameters()).device))
        item_emb = h.get("product", torch.empty(0, self.embed_dim, device=next(self.parameters()).device))
        return user_emb, item_emb

    def score(self, data: HeteroData, user_idx: Tensor, item_idx: Tensor) -> Tensor:
        """Dot-product scores for (user, item) pairs. Returns (B,)."""
        user_emb, item_emb = self.forward(data)
        return (user_emb[user_idx] * item_emb[item_idx]).sum(dim=-1)

    def embedding_l2_norm(self) -> Tensor:
        """L2 norm of input projection weights only (not GNN weights)."""
        return sum(p.pow(2).sum() for p in self.input_proj.parameters())
