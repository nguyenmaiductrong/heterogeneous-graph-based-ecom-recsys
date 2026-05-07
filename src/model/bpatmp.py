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

class FourierTimeEncoding(nn.Module):
    """Bien doi thoi gian delta_t thanh vector Fourier features.

    Su dung tan so hoc duoc (learnable frequencies) de model tu tim
    cac chu ky thoi gian quan trong tu data.

    Input:  delta_t  [*]         (khoang thoi gian, don vi ngay)
    Output: phi      [*, 2*n_freqs]  (cos/sin features)

    Cong thuc:
        omega_k = softplus(raw_omega_k)          -- dam bao omega > 0
        t'      = log(1 + delta_t)               -- nen scale thoi gian
        phi     = [cos(t' * omega), sin(t' * omega)]
    """

    def __init__(self, n_freqs: int = 16) -> None:
        super().__init__()
        self.n_freqs = n_freqs
        self.raw_omega = nn.Parameter(torch.randn(n_freqs))

    def forward(self, delta_t: Tensor) -> Tensor:
        """
        Args:
            delta_t: [*] arbitrary-shape tensor of time deltas (in days).

        Returns:
            [*, 2 * n_freqs] Fourier time features.
        """
        omega = F.softplus(self.raw_omega)
        t = torch.log1p(delta_t).unsqueeze(-1)
        phase = t * omega
        return torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1)


class TemporalAttention(nn.Module):
    """Temporal attention voi decay-in-logit va value gate.

    Cong thuc:
        logit = Q·K/√d + b_ρ + time_bias − λ_β · log(1 + Δt/τ)
        alpha = scatter_softmax(logit, dst_idx)
        gate  = σ(W_gate · [h_src ‖ time_feat])

    Thanh phan:
        - Q·K/√d:     content match giua src va dst
        - b_ρ:         bias theo loai relation
        - time_bias:   Fourier features cua Δt project ve scalar
        - λ_β·log():   decay term, λ khac nhau theo behavior
                       (purchase decay cham hon view)
        - gate:        kiem soat luong thong tin tu src di qua
    """

    def __init__(
        self,
        dim: int,
        n_relations: int = len(ALL_EDGE_TYPES),
        n_beta: int = 4,
        n_freqs: int = 16,
        tau: float = 7.0,
        in_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim if in_dim is not None else dim
        self.dim = dim
        self.scale = dim ** -0.5
        self.tau = tau

        self.W_q = nn.Linear(self.in_dim, dim, bias=False)
        self.W_k = nn.Linear(self.in_dim, dim, bias=False)

        self.b_rho = nn.Parameter(torch.zeros(n_relations))

        self.time_enc = FourierTimeEncoding(n_freqs)
        self.time_bias_proj = nn.Linear(n_freqs * 2, 1, bias=False)

        self.raw_lambda = nn.Parameter(torch.zeros(n_beta))

        self.gate_proj = nn.Linear(self.in_dim + n_freqs * 2, 1)

    def forward(
        self,
        h_src: Tensor,
        h_dst: Tensor,
        delta_t: Tensor,
        rho: int,
        beta: Tensor,
        dst_idx: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            h_src:   [E, dim]  source node embeddings
            h_dst:   [E, dim]  destination node embeddings
            delta_t: [E]       time deltas in days
            rho:     int       relation index
            beta:    [E]       behavior index per edge (0-3)
            dst_idx: [E]       destination node indices

        Returns:
            alpha: [E]  attention weights (scatter-softmax normalised per dst)
            gate:  [E]  value gate (sigmoid)
        """
        q = self.W_q(h_dst)
        k = self.W_k(h_src)
        qk = (q * k).sum(dim=-1) * self.scale

        bias = self.b_rho[rho]

        phi = self.time_enc(delta_t)
        time_bias = self.time_bias_proj(phi).squeeze(-1)

        lam = F.softplus(self.raw_lambda)
        decay = lam[beta] * torch.log1p(delta_t / self.tau)

        logit = qk + bias + time_bias - decay
        alpha = self._scatter_softmax(logit, dst_idx)

        gate_in = torch.cat([h_src, phi], dim=-1) 
        gate = torch.sigmoid(self.gate_proj(gate_in).squeeze(-1))

        return alpha, gate

    @staticmethod
    def _scatter_softmax(logit: Tensor, index: Tensor) -> Tensor:
        """Numerically stable softmax grouped by destination node."""
        N = int(index.max().item()) + 1
        max_val = logit.new_full((N,), torch.finfo(logit.dtype).min)
        max_val.scatter_reduce_(0, index, logit.detach(), reduce="amax", include_self=True)
        exp_logit = (logit - max_val[index]).exp()

        sum_exp = logit.new_zeros(N)
        sum_exp.scatter_add_(0, index, exp_logit)
        return exp_logit / (sum_exp[index] + 1e-12)


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
    
    def compute_popularity_score(
        self,
        item_idx: Tensor,
        item_counts: Tensor,
        item_decay_sum: Tensor,
    ) -> Tensor:
        """Compute time-decayed popularity score.

        Args:
            item_idx: [B] item indices
            item_counts: [N, 3] per-behavior counts
            item_decay_sum: [N, 3] sum of time-decayed weights

        Returns:
            s_pop: [B] popularity scores
        """
        eta = F.softplus(self.raw_eta)
        counts = item_counts[item_idx]
        decay_sum = item_decay_sum[item_idx]
        weighted = (eta * decay_sum).sum(dim=-1)
        return torch.log1p(weighted)

    def forward(
        self,
        user_emb: Tensor,
        item_emb: Tensor,
        user_idx: Tensor,
        item_idx: Tensor,
        item_seq: Optional[Tensor] = None,
        beh_seq: Optional[Tensor] = None,
        ts_seq: Optional[Tensor] = None,
        ref_time: Optional[float] = None,
        user_n_events: Optional[Tensor] = None,
        user_last_dt: Optional[Tensor] = None,
        item_pop_decay: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute final score with expert fusion.

        Returns:
            score: [B] final scores
        """
        h_u = user_emb[user_idx]
        h_i = item_emb[item_idx]
        s_graph = (h_u * h_i).sum(dim=-1) + self.item_bias(item_idx).squeeze(-1)

        B = user_idx.size(0)
        device = user_idx.device

        if item_seq is not None and ref_time is not None:
            z_u = self.encode_sequence(item_seq, beh_seq, ts_seq, ref_time, item_emb)
            e_i = item_emb[item_idx]
            s_seq = (z_u * e_i).sum(dim=-1)
        else:
            s_seq = torch.zeros(B, device=device)

        if item_pop_decay is not None:
            s_pop = self.compute_popularity_score(item_idx, item_pop_decay, item_pop_decay)
        else:
            s_pop = torch.zeros(B, device=device)

        if user_n_events is None:
            user_n_events = torch.ones(B, device=device)
        if user_last_dt is None:
            user_last_dt = torch.ones(B, device=device)

        feat = torch.stack(
            [
                torch.log1p(user_n_events[user_idx].float()),
                torch.log1p(user_last_dt[user_idx].float()),
                (h_u * h_u).sum(dim=-1).sqrt(),
            ],
            dim=-1,
        )

        omega = F.softmax(self.fusion_mlp(feat), dim=-1)
        score = omega[:, 0] * s_graph + omega[:, 1] * s_seq + omega[:, 2] * s_pop

        return score
    
BEH_BUCKETS: Tuple[str, str, str, str] = ("view", "cart", "purchase", "struct")
_BEH_W_INIT = torch.tensor([0.30, 0.50, 1.00, 0.40])


class BehaviorNormalizedAgg(nn.Module):
    def __init__(self, out_dim: int) -> None:
        super().__init__()
        self.beh_w = nn.ParameterDict({t: nn.Parameter(_BEH_W_INIT.clone()) for t in NODE_TYPES})
        self.norms = nn.ModuleDict(
            {f"{t}__{b}": nn.LayerNorm(out_dim) for t in NODE_TYPES for b in BEH_BUCKETS}
        )

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
            mixed = sum(w[i] * self.norms[f"{t}__{b}"](agg_pb[t][b]) for i, b in present)
            if t in x_dict and x_dict[t].shape == mixed.shape:
                mixed = mixed + x_dict[t]
            out[t] = F.elu(mixed)
        return out

class BPATMPConv(nn.Module):
    def __init__(
        self,
        in_dim: int = EMBED_DIM,
        out_dim: int = EMBED_DIM,
        rank: int = 16,
        n_relations: int = len(ALL_EDGE_TYPES),
        n_freqs: int = 16,
        tau: float = 7.0,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.baw = BehaviorAwareWeight(in_dim, out_dim, rank, n_relations=n_relations)
        self.temporal_attn = TemporalAttention(
            dim=out_dim,
            n_relations=n_relations,
            n_beta=4,
            n_freqs=n_freqs,
            tau=tau,
            in_dim=in_dim,
        )
        self.behavior_agg = BehaviorNormalizedAgg(out_dim)

    def forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[Tuple, Tensor],
        edge_attr_dict: Optional[Dict[Tuple, Tensor]] = None,
        edge_ts_dict: Optional[Dict[Tuple, Tensor]] = None,
        ref_time: Optional[float] = None,
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        """Forward pass with temporal attention.

        Args:
            x_dict: node embeddings per type
            edge_index_dict: edge indices per edge type
            edge_attr_dict: behavior origin for structural edges
            edge_ts_dict: timestamps per edge type (Unix seconds)
            ref_time: reference time T for computing delta_t
        """
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
            E = h_src.size(0)

            rho = _REL_IDX[edge_type]
            beta_raw = BEHAVIOR_ORIGIN.get(edge_name.removeprefix("rev_"), -1)
            attr = (edge_attr_dict or {}).get(edge_type)

            edge_ts = (edge_ts_dict or {}).get(edge_type)
            if edge_ts is not None:
                edge_ts = edge_ts.to(device=ref.device)

            attr = attr.to(device=ref.device) if attr is not None else None
            if edge_ts is not None and ref_time is not None:
                keep = edge_ts.float() < float(ref_time)
                if not keep.any():
                    continue
                if not bool(keep.all()):
                    src_idx = src_idx[keep]
                    dst_idx = dst_idx[keep]
                    h_src = h_src[keep]
                    h_dst = h_dst[keep]
                    edge_ts = edge_ts[keep]
                    if attr is not None and attr.numel() > 0:
                        attr = attr.view(-1)[keep].view(-1, 1)
                    E = h_src.size(0)

            if edge_ts is not None and ref_time is not None:
                delta_t = (ref_time - edge_ts.float()) / 86400.0
                delta_t = delta_t.clamp(min=0)
            else:
                delta_t = torch.zeros(E, device=ref.device)

            if beta_raw < 0 and attr is not None and attr.numel() > 0:
                origin = attr.view(-1).long().clamp(0, 2)
                W_rho = self.baw.W_rho[rho]
                A_rho = self.baw.A_rho[rho]
                B_rho = self.baw.B_rho[rho]

                base_msg = torch.einsum("oi,ei->eo", W_rho, h_src)
                h_B = h_src @ B_rho
                scaled = torch.zeros_like(h_B)
                for b_idx in origin.unique():
                    mask = origin == b_idx
                    scaled[mask] = h_B[mask] * self.baw.z_beta[b_idx]
                msg = base_msg + scaled @ A_rho.T
                beta_tensor = origin
            else:
                beta_idx = beta_raw if beta_raw >= 0 else 3
                W = self.baw(rho, beta_raw)
                msg = torch.einsum("oi,ei->eo", W, h_src)
                beta_tensor = torch.full((E,), beta_idx, device=ref.device, dtype=torch.long)

            alpha, gate = self.temporal_attn(h_src, h_dst, delta_t, rho, beta_tensor, dst_idx)

            weighted = (alpha * gate).unsqueeze(-1) * msg
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

class BPATMPLayer(nn.Module):
    def __init__(
        self,
        in_dim: int = EMBED_DIM,
        hid_dim: int = EMBED_DIM,
        out_dim: int = EMBED_DIM,
        n_layers: int = 2,
        rank: int = 16,
        dropout: float = 0.1,
        use_checkpoint: bool = True,
        n_freqs: int = 16,
        tau: float = 7.0,
    ) -> None:
        super().__init__()
        assert n_layers >= 1
        self.use_checkpoint = use_checkpoint
        dims = [in_dim] + [hid_dim] * (n_layers - 1) + [out_dim]
        self.convs = nn.ModuleList(
            [
                BPATMPConv(
                    in_dim=dims[i],
                    out_dim=dims[i + 1],
                    rank=rank,
                    n_freqs=n_freqs,
                    tau=tau,
                )
                for i in range(n_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[Tuple, Tensor],
        edge_attr_dict: Optional[Dict[Tuple, Tensor]] = None,
        edge_ts_dict: Optional[Dict[Tuple, Tensor]] = None,
        ref_time: Optional[float] = None,
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        BEH_KEYS = ["view", "cart", "purchase"]
        h = x_dict
        beh_user_agg: Dict[str, Tensor] = {}

        for i, conv in enumerate(self.convs):
            if self.training and self.use_checkpoint:
                node_types = [t for t in NODE_TYPES if t in h]
                node_vals = [h[t] for t in node_types]

                def _conv(
                    *vals,
                    _conv=conv,
                    _nt=node_types,
                    _eid=edge_index_dict,
                    _ead=edge_attr_dict,
                    _ets=edge_ts_dict,
                    _rt=ref_time,
                    _bk=BEH_KEYS,
                ):
                    x = dict(zip(_nt, vals))
                    out_dict, b_agg = _conv(x, _eid, _ead, _ets, _rt)
                    return [out_dict[t] for t in _nt] + [b_agg[k] for k in _bk]

                out_all = grad_checkpoint(_conv, *node_vals, use_reentrant=False)
                n = len(node_types)
                h = dict(zip(node_types, out_all[:n]))
                beh_user_agg = {k: out_all[n + j] for j, k in enumerate(BEH_KEYS)}
            else:
                h, beh_user_agg = conv(h, edge_index_dict, edge_attr_dict, edge_ts_dict, ref_time)

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
        self._scale = dim**-0.5

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
