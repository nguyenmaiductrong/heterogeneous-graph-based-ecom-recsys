from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from torch_geometric.data import HeteroData
from torch_geometric.utils import softmax
from typing import Dict, Optional, Tuple

from src.model.hierarchy_gate import HierarchyGate
from src.core.contracts import EMBED_DIM

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

_REL_IDX: Dict[Tuple[str, str, str], int] = {et: i for i, et in enumerate(ALL_EDGE_TYPES)}

_BEH_IDX: Dict[str, int] = {
    "view": 0, "cart": 1, "purchase": 2,
    "rev_view": 0, "rev_cart": 1, "rev_purchase": 2,
    "belongs_to": 3, "contains": 3, "producedBy": 3, "brands": 3,
}

_REV_BEH_KEYS = {"rev_view": "view", "rev_cart": "cart", "rev_purchase": "purchase"}


class BehaviorAwareWeight(nn.Module):
    def __init__(
        self,
        in_dim:  int,
        out_dim: int,
        rank:    int = 16,
        n_phi:   int = len(NODE_TYPES),
        n_beta:  int = 4,
    ) -> None:
        super().__init__()
        self.W_base = nn.Parameter(torch.empty(out_dim, in_dim))
        self.A = nn.Parameter(torch.empty(n_phi, out_dim, rank))
        self.B = nn.Parameter(torch.empty(n_beta, in_dim, rank))
        nn.init.kaiming_uniform_(self.W_base)
        nn.init.kaiming_uniform_(self.A)
        nn.init.kaiming_uniform_(self.B)

    def forward(self, phi: int, beta: int) -> Tensor:
        b_idx = beta if beta >= 0 else 3
        return self.W_base + self.A[phi] @ self.B[b_idx].T


BEH_BUCKETS: Tuple[str, str, str, str] = ("view", "cart", "purchase", "struct")
_BEH_W_INIT = torch.tensor([0.30, 0.50, 1.00, 0.40])


class BehaviorNormalizedAgg(nn.Module):
    """Per-(node_type, behavior) aggregation with learned mixing weights.

    Replaces the single agg[dst_type] sum-scatter in BAGNNConv. Each behavior's
    contribution is LayerNorm'd separately so view (which has ~100x more edges
    than purchase on REES46) cannot dominate the dst-node representation by raw
    edge count alone.
    """

    def __init__(self, out_dim: int) -> None:
        super().__init__()
        self.beh_w = nn.ParameterDict({
            t: nn.Parameter(_BEH_W_INIT.clone()) for t in NODE_TYPES
        })
        self.norms = nn.ModuleDict({
            f"{t}__{b}": nn.LayerNorm(out_dim)
            for t in NODE_TYPES for b in BEH_BUCKETS
        })

    def forward(
        self,
        agg_pb: Dict[str, Dict[str, Optional[Tensor]]],
        x_dict: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        out: Dict[str, Tensor] = {}
        for t in NODE_TYPES:
            present = [(i, b) for i, b in enumerate(BEH_BUCKETS) if agg_pb[t][b] is not None]
            if not present:
                if t in x_dict:
                    out[t] = x_dict[t]
                continue
            w = F.softmax(self.beh_w[t], dim=0)
            mixed = sum(
                w[i] * self.norms[f"{t}__{b}"](agg_pb[t][b])
                for i, b in present
            )
            if t in x_dict and x_dict[t].shape == mixed.shape:
                mixed = mixed + x_dict[t]
            out[t] = F.elu(mixed)
        return out


class BAGNNConv(nn.Module):
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
        self.beh_emb = nn.Embedding(4, out_dim)
        self.a_att   = nn.Parameter(torch.empty(4 * out_dim))
        self.behavior_agg = BehaviorNormalizedAgg(out_dim)
        nn.init.xavier_uniform_(self.a_att.unsqueeze(0))

    def forward(
        self,
        x_dict:          Dict[str, Tensor],
        edge_index_dict: Dict[Tuple, Tensor],
        edge_attr_dict:  Optional[Dict[Tuple, Tensor]] = None,
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        all_r_emb = self.rel_emb.weight
        all_b_emb = self.beh_emb.weight

        agg_pb: Dict[str, Dict[str, Optional[Tensor]]] = {
            t: {b: None for b in BEH_BUCKETS} for t in NODE_TYPES
        }

        n_users = x_dict["user"].size(0) if "user" in x_dict else 0
        ref = next(iter(x_dict.values()))
        beh_user_agg: Dict[str, Tensor] = {
            k: torch.zeros(n_users, self.out_dim, device=ref.device, dtype=ref.dtype)
            for k in ("view", "cart", "purchase")
        }

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
                origin = attr.view(-1).long().clamp(0, 2)

                base_msg   = torch.einsum("oi,ei->eo", self.baw.W_base, h_src)
                A_phi      = self.baw.A[phi]
                mid = torch.zeros(origin.size(0), self.baw.B.size(-1),
                                  device=h_src.device, dtype=h_src.dtype)
                for b_idx in origin.unique():
                    mask = origin == b_idx
                    mid[mask] = (h_src[mask] @ self.baw.B[b_idx]).to(mid.dtype)
                msg        = base_msg + mid @ A_phi.T

                b_emb      = all_b_emb[origin]
                h_dst_proj = torch.einsum("oi,ei->eo", self.baw.W_base, h_dst)
            else:
                beta_idx   = beta_raw if beta_raw >= 0 else 3
                W          = self.baw(phi, beta_raw)
                msg        = torch.einsum("oi,ei->eo", W, h_src)
                b_emb      = None
                h_dst_proj = F.linear(h_dst, W)

            d = self.out_dim
            a0, a1, a2, a3 = (self.a_att[:d], self.a_att[d:2*d],
                               self.a_att[2*d:3*d], self.a_att[3*d:])

            r_scalar = (all_r_emb[_REL_IDX[edge_type]] * a2).sum()

            if beta_raw < 0 and attr is not None and attr.numel() > 0:
                b_term = (b_emb * a3).sum(-1)
            else:
                b_term = (all_b_emb[beta_idx] * a3).sum()

            e = ((msg * a0).sum(-1)
                 + (h_dst_proj * a1).sum(-1)
                 + r_scalar
                 + b_term)
            alpha = softmax(e, dst_idx)

            weighted = alpha.unsqueeze(-1) * msg
            N_dst = x_dict[dst_type].size(0)
            bucket = BEH_BUCKETS[_BEH_IDX[edge_name]]
            if agg_pb[dst_type][bucket] is None:
                agg_pb[dst_type][bucket] = weighted.new_zeros(N_dst, self.out_dim)
            agg_pb[dst_type][bucket].scatter_add_(
                0, dst_idx.unsqueeze(-1).expand_as(weighted), weighted
            )

            beh_key = _REV_BEH_KEYS.get(edge_name)
            if beh_key is not None and n_users > 0:
                beh_user_agg[beh_key].scatter_add_(
                    0, dst_idx.unsqueeze(-1).expand_as(weighted), weighted
                )

        out_dict = self.behavior_agg(agg_pb, x_dict)

        return out_dict, beh_user_agg


class BAGNNLayer(nn.Module):
    def __init__(
        self,
        in_dim:   int = EMBED_DIM,
        hid_dim:  int = EMBED_DIM,
        out_dim:  int = EMBED_DIM,
        n_layers: int = 2,
        rank:     int = 16,
        dropout:  float = 0.1,
        use_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        assert n_layers >= 1
        self.use_checkpoint = use_checkpoint
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
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        BEH_KEYS = ["view", "cart", "purchase"]
        h = x_dict
        beh_user_agg: Dict[str, Tensor] = {}

        for i, conv in enumerate(self.convs):
            if self.training and self.use_checkpoint:
                node_types = [t for t in NODE_TYPES if t in h]
                node_vals  = [h[t] for t in node_types]

                def _conv(*vals, _conv=conv, _nt=node_types,
                          _eid=edge_index_dict, _ead=edge_attr_dict, _bk=BEH_KEYS):
                    x = dict(zip(_nt, vals))
                    out_dict, b_agg = _conv(x, _eid, _ead)
                    return [out_dict[t] for t in _nt] + [b_agg[k] for k in _bk]

                out_all  = grad_checkpoint(_conv, *node_vals, use_reentrant=False)
                n        = len(node_types)
                h        = dict(zip(node_types, out_all[:n]))
                beh_user_agg = {k: out_all[n + j] for j, k in enumerate(BEH_KEYS)}
            else:
                h, beh_user_agg = conv(h, edge_index_dict, edge_attr_dict)

            if i < len(self.convs) - 1:
                h = {t: self.dropout(v) for t, v in h.items()}

        return h, beh_user_agg


class IntentCodebook(nn.Module):
    """Shared low-rank intent codebook. Per-node attention over a small set
    of E intent embeddings; weighted-sum is added back as a residual.

    Decoupled from behavior on purpose: a user's intent is the SAME across
    view/cart/purchase. Counter-position vs MixRec's H^(u)_k which decouples
    intents per behavior — sharing the codebook gives cross-behavior intent
    sharing for free.
    """
    def __init__(self, n_intents: int = 32, dim: int = EMBED_DIM):
        super().__init__()
        self.n_intents = n_intents
        self.dim = dim
        self.user_intents = nn.Parameter(torch.empty(n_intents, dim))
        self.item_intents = nn.Parameter(torch.empty(n_intents, dim))
        nn.init.xavier_uniform_(self.user_intents)
        nn.init.xavier_uniform_(self.item_intents)
        self._scale = dim ** -0.5

    def _attend(self, x: Tensor, codebook: Tensor) -> Tensor:
        attn = (x @ codebook.T) * self._scale
        attn = torch.softmax(attn, dim=-1)
        return attn @ codebook

    def forward(self, x_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        out = dict(x_dict)
        if "user" in out:
            out["user"] = out["user"] + self._attend(out["user"], self.user_intents)
        if "product" in out:
            out["product"] = out["product"] + self._attend(out["product"], self.item_intents)
        return out


class BAGNNModel(nn.Module):
    def __init__(
        self,
        feat_dims: Optional[Dict[str, int]] = None,
        n_nodes:   Optional[Dict[str, int]] = None,
        embed_dim: int   = EMBED_DIM,
        n_layers:  int   = 2,
        rank:      int   = 16,
        dropout:   float = 0.1,
        use_grad_checkpoint: bool = True,
        n_intents: int   = 32,
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

        self.intent_codebook = (
            IntentCodebook(n_intents=n_intents, dim=embed_dim)
            if n_intents > 0 else None
        )

        self.bagnn = BAGNNLayer(
            in_dim=embed_dim, hid_dim=embed_dim, out_dim=embed_dim,
            n_layers=n_layers, rank=rank, dropout=dropout,
            use_checkpoint=use_grad_checkpoint,
        )

        self.beh_proj = nn.ModuleDict({
            "view":     nn.Linear(embed_dim, embed_dim),
            "cart":     nn.Linear(embed_dim, embed_dim),
            "purchase": nn.Linear(embed_dim, embed_dim),
        })
        self.hierarchy_gate = HierarchyGate(embed_dim)

    def encode(self, data: HeteroData) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
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

        if self.intent_codebook is not None:
            x_dict = self.intent_codebook(x_dict)

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

    def forward(self, data: HeteroData, return_beh_embs: bool = False):
        h, beh_user_agg = self.encode(data)
        device   = next(self.parameters()).device
        user_raw = h.get("user",    torch.empty(0, self.embed_dim, device=device))
        item_emb = h.get("product", torch.empty(0, self.embed_dim, device=device))

        emb_view     = self.beh_proj["view"](beh_user_agg.get("view",     user_raw))
        emb_cart     = self.beh_proj["cart"](beh_user_agg.get("cart",     user_raw))
        emb_purchase = self.beh_proj["purchase"](beh_user_agg.get("purchase", user_raw))

        if user_raw.size(0) > 0:
            user_emb, _ = self.hierarchy_gate(emb_view, emb_cart, emb_purchase)
        else:
            user_emb = user_raw

        if return_beh_embs:
            return user_emb, item_emb, {"view": emb_view, "cart": emb_cart, "purchase": emb_purchase}
        return user_emb, item_emb

    def score(self, data: HeteroData, user_idx: Tensor, item_idx: Tensor) -> Tensor:
        user_emb, item_emb = self.forward(data)
        return (user_emb[user_idx] * item_emb[item_idx]).sum(dim=-1)

    def embedding_l2_norm(self) -> Tensor:
        l2 = sum(p.pow(2).sum() for p in self.input_proj.parameters())
        l2 = l2 + sum(p.pow(2).sum() for p in self.beh_proj.parameters())
        return l2
