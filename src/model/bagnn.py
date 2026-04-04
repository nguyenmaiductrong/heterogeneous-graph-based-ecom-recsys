"""
BAGNN — Behavior-Aware Graph Neural Network
============================================
Node types  : user, product, category, brand
Edge types  : view, cart, purchase (+ reverses), belongs_to, contains, has_brand, brands
Embed dim   : 128  (EMBED_DIM)

Tasks covered
-------------
N1.1  W(φ,β) = W_base + A_φ @ B_β.T   — low-rank behavior-aware weight decomposition
N1.2  BAGNNConv.forward()              — einsum-vectorized, 4-tuple attention
N1.3  BAGNNLayer stack + BAGNNModel.forward() skeleton
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.utils import softmax
from typing import Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants (mirror nodes.yaml / relations.yaml)
# ---------------------------------------------------------------------------

EMBED_DIM: int = 128

NODE_TYPES = ["user", "product", "category", "brand"]

# Behavior edges only (have behavior_origin ∈ {0,1,2})
BEHAVIOR_EDGES = [
    ("user",    "view",         "product"),
    ("user",    "cart",         "product"),
    ("user",    "purchase",     "product"),
    ("product", "rev_view",     "user"),
    ("product", "rev_cart",     "user"),
    ("product", "rev_purchase", "user"),
]

# Structural edges (no behavior_origin)
STRUCTURAL_EDGES = [
    ("product",  "belongs_to", "category"),
    ("category", "contains",   "product"),
    ("product",  "has_brand",  "brand"),
    ("brand",    "brands",     "product"),
]

ALL_EDGE_TYPES = BEHAVIOR_EDGES + STRUCTURAL_EDGES

# behavior_origin values (view=0, cart=1, purchase=2)
BEHAVIOR_ORIGIN: Dict[str, int] = {"view": 0, "cart": 1, "purchase": 2}


# ---------------------------------------------------------------------------
# N1.1 — Low-rank behavior-aware weight decomposition
# W(φ, β) = W_base + A_φ @ B_β.T
# ---------------------------------------------------------------------------

class BehaviorAwareWeight(nn.Module):
    """
    Decomposes the message-passing weight matrix into a shared base plus a
    low-rank behavior-specific correction:

        W(φ, β) = W_base  +  A_φ  @  B_β.T

    Parameters
    ----------
    in_dim   : input feature dimension
    out_dim  : output feature dimension
    rank     : low-rank bottleneck dimension r  (r << in_dim)
    n_phi    : number of node-type   specialisations  (|φ|)
    n_beta   : number of behavior    specialisations  (|β|)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        rank: int = 16,
        n_phi: int = len(NODE_TYPES),
        n_beta: int = 3,           # view / cart / purchase
    ) -> None:
        super().__init__()
        self.in_dim  = in_dim
        self.out_dim = out_dim
        self.rank    = rank

        # Shared base weight  (out_dim × in_dim)
        self.W_base = nn.Parameter(torch.empty(out_dim, in_dim))

        # Node-type factors   A_φ  : n_phi  × out_dim × rank
        self.A = nn.Parameter(torch.empty(n_phi, out_dim, rank))

        # Behavior factors    B_β  : n_beta × in_dim  × rank
        self.B = nn.Parameter(torch.empty(n_beta, in_dim, rank))

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_uniform_(self.W_base)
        nn.init.kaiming_uniform_(self.A)
        nn.init.kaiming_uniform_(self.B)

    def forward(self, phi: int, beta: int) -> Tensor:
        """
        Returns W(φ, β) of shape (out_dim, in_dim).

        Parameters
        ----------
        phi  : node-type   index  (0 = user, 1 = product, …)
        beta : behavior    index  (0 = view, 1 = cart, 2 = purchase)
               pass -1 for structural edges → returns W_base only
        """
        if beta < 0:
            return self.W_base                          # structural edge

        A_phi  = self.A[phi]                            # (out_dim, rank)
        B_beta = self.B[beta]                           # (in_dim,  rank)
        delta  = A_phi @ B_beta.T                       # (out_dim, in_dim)
        return self.W_base + delta


# ---------------------------------------------------------------------------
# N1.2 — BAGNNConv: one heterogeneous convolution step
# ---------------------------------------------------------------------------

class BAGNNConv(nn.Module):
    """
    Single BAGNN convolution layer.

    For each target node t and each incoming edge relation r = (s, e, t):
      1. Compute behavior-aware messages:   m_i = W(φ_s, β_e) @ h_i
      2. Compute 4-tuple attention score:   α = att(h_src, h_dst, r_emb, β_emb)
      3. Aggregate with softmax attention over neighbors.

    The 4-tuple attention vector is:
        [h_src || h_dst || r_emb || β_emb]
    projected to a scalar via a learned vector a_att.
    """

    def __init__(
        self,
        in_dim:  int = EMBED_DIM,
        out_dim: int = EMBED_DIM,
        rank:    int = 16,
        n_relations: int = len(ALL_EDGE_TYPES),
    ) -> None:
        super().__init__()
        self.in_dim  = in_dim
        self.out_dim = out_dim

        # N1.1 weight decomposition
        self.baw = BehaviorAwareWeight(in_dim, out_dim, rank)

        # Relation embeddings  r_emb  (one per edge type)
        self.rel_emb = nn.Embedding(n_relations, out_dim)

        # Behavior embeddings  β_emb  (view / cart / purchase + structural=-1→idx 3)
        self.beh_emb = nn.Embedding(4, out_dim)         # 0-2 behavior + 3 structural

        # 4-tuple attention projection: [h_src || h_dst || r_emb || β_emb] → scalar
        self.a_att = nn.Parameter(torch.empty(4 * out_dim))
        nn.init.xavier_uniform_(self.a_att.unsqueeze(0))

        # Output normalisation
        self.norm = nn.LayerNorm(out_dim)

    # ------------------------------------------------------------------
    def _relation_index(self, edge_type: Tuple[str, str, str]) -> int:
        return ALL_EDGE_TYPES.index(edge_type)

    def _beta_index(self, edge_name: str) -> int:
        """Return behavior index (0-2) or 3 for structural."""
        return BEHAVIOR_ORIGIN.get(edge_name.lstrip("rev_").rstrip("_").split("_")[0]
                                   if edge_name.startswith("rev_")
                                   else edge_name, 3)

    # ------------------------------------------------------------------
    def forward(
        self,
        x_dict:       Dict[str, Tensor],          # node embeddings per type
        edge_index_dict: Dict[Tuple, Tensor],      # edge indices per relation
    ) -> Dict[str, Tensor]:
        """
        Returns updated node embeddings for every node type.

        Parameters
        ----------
        x_dict          : { node_type: Tensor (N_t, in_dim) }
        edge_index_dict : { (src_type, edge, dst_type): Tensor (2, E) }

        Returns
        -------
        out_dict : { node_type: Tensor (N_t, out_dim) }
        """
        # Accumulate messages per destination node type
        agg: Dict[str, Optional[Tensor]] = {t: None for t in NODE_TYPES}
        cnt: Dict[str, int]              = {t: 0     for t in NODE_TYPES}

        for edge_type, edge_index in edge_index_dict.items():
            src_type, edge_name, dst_type = edge_type

            if src_type not in x_dict or dst_type not in x_dict:
                continue

            src_idx, dst_idx = edge_index                          # (E,)
            h_src = x_dict[src_type][src_idx]                     # (E, in_dim)
            h_dst = x_dict[dst_type][dst_idx]                     # (E, in_dim)

            # ---- N1.1: behavior-aware weight ----
            phi  = NODE_TYPES.index(src_type)
            beta_raw = BEHAVIOR_ORIGIN.get(edge_name, -1)
            W = self.baw(phi, beta_raw)                            # (out_dim, in_dim)

            # Messages via einsum: m_i = W @ h_src_i
            # h_src : (E, in_dim)  →  msg : (E, out_dim)
            msg = torch.einsum("oi,ei->eo", W, h_src)             # (E, out_dim)

            # ---- N1.2: 4-tuple attention ----
            r_idx   = self._relation_index(edge_type)
            b_idx   = self._beta_index(edge_name)

            r_emb   = self.rel_emb(
                torch.tensor(r_idx, device=h_src.device)
            ).expand(h_src.size(0), -1)                           # (E, out_dim)

            b_emb   = self.beh_emb(
                torch.tensor(b_idx, device=h_src.device)
            ).expand(h_src.size(0), -1)                           # (E, out_dim)

            # Project W @ h_src to out_dim for attention tuple
            h_src_proj = msg                                       # (E, out_dim)
            h_dst_proj = F.linear(h_dst, W)                       # (E, out_dim)  reuse W

            att_input = torch.cat(
                [h_src_proj, h_dst_proj, r_emb, b_emb], dim=-1
            )                                                      # (E, 4*out_dim)

            # Scalar attention logit per edge
            e = (att_input * self.a_att).sum(dim=-1)               # (E,)
            alpha = softmax(e, dst_idx)                            # (E,)  PyG softmax

            # Weighted messages
            weighted_msg = alpha.unsqueeze(-1) * msg               # (E, out_dim)

            # Scatter-add into destination
            N_dst = x_dict[dst_type].size(0)
            if agg[dst_type] is None:
                agg[dst_type] = weighted_msg.new_zeros(N_dst, self.out_dim)
            agg[dst_type].scatter_add_(
                0,
                dst_idx.unsqueeze(-1).expand_as(weighted_msg),
                weighted_msg,
            )
            cnt[dst_type] += 1

        # Post-aggregate norm + residual (if dims match)
        out_dict: Dict[str, Tensor] = {}
        for t in NODE_TYPES:
            if agg[t] is not None:
                h = self.norm(agg[t])
                if t in x_dict and x_dict[t].shape == h.shape:
                    h = h + x_dict[t]                              # residual
                out_dict[t] = F.elu(h)
            elif t in x_dict:
                out_dict[t] = x_dict[t]                            # no edges → pass-through

        return out_dict


# ---------------------------------------------------------------------------
# N1.3 — BAGNNLayer stack + BAGNNModel skeleton
# ---------------------------------------------------------------------------

class BAGNNLayer(nn.Module):
    """
    Stacks `n_layers` BAGNNConv layers with optional dropout between layers.
    """

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

        dims = (
            [in_dim]
            + [hid_dim] * (n_layers - 1)
            + [out_dim]
        )
        self.convs = nn.ModuleList([
            BAGNNConv(in_dim=dims[i], out_dim=dims[i + 1], rank=rank)
            for i in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_dict:          Dict[str, Tensor],
        edge_index_dict: Dict[Tuple, Tensor],
    ) -> Dict[str, Tensor]:
        h = x_dict
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index_dict)
            if i < len(self.convs) - 1:
                h = {t: self.dropout(x) for t, x in h.items()}
        return h


class BAGNNModel(nn.Module):
    """
    Full BAGNN model for heterogeneous e-commerce recommendation.

    Architecture
    ------------
    1. Node-type input projections  (raw feat → EMBED_DIM)
    2. BAGNNLayer stack             (message passing)
    3. Purchase-score decoder       dot(h_user, h_product)

    Parameters
    ----------
    feat_dims   : raw feature dimension per node type  { node_type: int }
                  Pass None to use learned embeddings (mock / no raw features).
    n_nodes     : number of nodes per type             { node_type: int }
                  Required only when feat_dims[t] is None.
    embed_dim   : EMBED_DIM (default 128)
    n_layers    : number of BAGNNConv layers
    rank        : low-rank bottleneck r
    dropout     : dropout probability
    """

    def __init__(
        self,
        feat_dims:  Optional[Dict[str, int]] = None,
        n_nodes:    Optional[Dict[str, int]] = None,
        embed_dim:  int   = EMBED_DIM,
        n_layers:   int   = 2,
        rank:       int   = 16,
        dropout:    float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        # Input layer: either linear projection or learned embedding table
        self.input_proj: nn.ModuleDict = nn.ModuleDict()
        for t in NODE_TYPES:
            if feat_dims and t in feat_dims and feat_dims[t] is not None:
                self.input_proj[t] = nn.Linear(feat_dims[t], embed_dim)
            else:
                n = n_nodes[t] if n_nodes and t in n_nodes else 1000
                self.input_proj[t] = nn.Embedding(n, embed_dim)

        # BAGNN stack
        self.bagnn = BAGNNLayer(
            in_dim=embed_dim,
            hid_dim=embed_dim,
            out_dim=embed_dim,
            n_layers=n_layers,
            rank=rank,
            dropout=dropout,
        )

    # ------------------------------------------------------------------
    def encode(
        self,
        data: HeteroData,
    ) -> Dict[str, Tensor]:
        """
        Encode all node types → embeddings dict.

        Supports both:
        - Mock HeteroData  (random tensors, integer node IDs)
        - Real  HeteroData (data[node_type].x available)
        """
        x_dict: Dict[str, Tensor] = {}
        for t in NODE_TYPES:
            if t not in data.node_types:
                continue
            proj = self.input_proj[t]
            if isinstance(proj, nn.Embedding):
                # Mock mode: use arange as node IDs
                n = data[t].num_nodes
                ids = torch.arange(n, device=next(self.parameters()).device)
                x_dict[t] = proj(ids)
            else:
                x_dict[t] = proj(data[t].x)

        edge_index_dict = {
            et: data[et].edge_index
            for et in data.edge_types
            if et in {tuple(e) for e in ALL_EDGE_TYPES}
        }

        return self.bagnn(x_dict, edge_index_dict)

    def forward(
        self,
        data:      HeteroData,
        user_idx:  Tensor,
        item_idx:  Tensor,
    ) -> Tensor:
        """
        Compute purchase scores for (user, item) pairs.

        Parameters
        ----------
        data      : HeteroData graph
        user_idx  : Tensor (B,)  — user node indices in the batch
        item_idx  : Tensor (B,)  — product node indices in the batch

        Returns
        -------
        scores : Tensor (B,)  — dot-product purchase score
        """
        h = self.encode(data)

        h_user    = h["user"][user_idx]        # (B, embed_dim)
        h_product = h["product"][item_idx]     # (B, embed_dim)

        scores = (h_user * h_product).sum(dim=-1)   # (B,)
        return scores