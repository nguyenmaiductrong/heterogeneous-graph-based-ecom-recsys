from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from typing import Dict, Optional, Tuple

from src.core.contracts import EMBED_DIM

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
        logit = Q·K/√d + b_ρ + u_{ρ,β}ᵀ Φ(Δt) − λ_β · log(1 + Δt/τ)
        alpha = scatter_softmax(logit, dst_idx)
        gate  = σ(c_{ρ,β} + r_{ρ,β}ᵀ Φ(Δt) − μ_β · log(1 + Δt/τ))

    Thanh phan:
        - Q·K/√d:        content match giua src va dst
        - b_ρ:           bias theo loai relation
        - u_{ρ,β}ᵀ Φ:    time bias per relation-behavior
        - λ_β·log():     decay term trong attention, khac nhau theo behavior
        - c_{ρ,β}:       gate bias per relation-behavior
        - r_{ρ,β}ᵀ Φ:    gate time projection per relation-behavior
        - μ_β·log():     decay term trong gate, khac nhau theo behavior
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
        self.n_relations = n_relations
        self.n_beta = n_beta

        self.W_q = nn.Linear(self.in_dim, dim, bias=False)
        self.W_k = nn.Linear(self.in_dim, dim, bias=False)

        self.b_rho = nn.Parameter(torch.zeros(n_relations))

        self.time_enc = FourierTimeEncoding(n_freqs)
        self.time_bias_proj = nn.Linear(n_freqs * 2, 1, bias=False)

        # Init bat doi xung de pha symmetry: raw_lambda=0 (softplus=0.693) cho moi behavior
        # khien decay rate khong khac biet -> module behavior-aware temporal decay vo dung.
        # view decays nhanh, purchase decay cham. struct=0 (mid).
        # softplus values: view~1.31, cart~0.91, purchase~0.62, struct=0.69
        self.raw_lambda = nn.Parameter(torch.tensor([1.0, 0.5, 0.0, 0.0]))

        self.c_rho_beta = nn.Parameter(torch.zeros(n_relations, n_beta))
        self.r_rho_beta = nn.Parameter(torch.zeros(n_relations, n_beta, n_freqs * 2))
        self.raw_mu = nn.Parameter(torch.tensor([0.7, 0.3, -0.3, 0.0]))

        nn.init.xavier_uniform_(self.r_rho_beta)

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

        c = self.c_rho_beta[rho, beta]
        r = self.r_rho_beta[rho, beta]
        r_phi = (r * phi).sum(dim=-1)
        mu = F.softplus(self.raw_mu)
        gate_decay = mu[beta] * torch.log1p(delta_t / self.tau)
        gate = torch.sigmoid(c + r_phi - gate_decay)

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
                    scaled[mask] = (h_B[mask] * self.baw.z_beta[b_idx]).to(scaled.dtype)
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

class HyperPropagation(nn.Module):
    """MixRec-style hypergraph propagation, per-behavior, applied per GNN layer.

    Cong thuc:
        uuH = u_emb_0 @ u_hyper                         # [N_user, hyper_num]
        edge_b = ReLU(uuH^T @ u_emb)                    # [hyper_num, dim]
        edge_b = FC_b(edge_b)                           # per-behavior transform
        residual_b = uuH @ edge_b                       # [N_user, dim]
        out = sum_b w_b * residual_b                    # mix theo behavior

    Tao non-local global communication: moi user "thuoc" hyperNum hyperedge
    voi soft assignment, message di tu user -> hyperedge -> tat ca user khac.
    Day la chia khoa MixRec dat 0.109 NDCG@20.
    """

    BEHAVIORS = ("view", "cart", "purchase")

    def __init__(self, dim: int, hyper_num: int = 128) -> None:
        super().__init__()
        self.dim = dim
        self.hyper_num = hyper_num
        self.u_hyper = nn.Parameter(torch.empty(dim, hyper_num))
        self.i_hyper = nn.Parameter(torch.empty(dim, hyper_num))
        nn.init.xavier_uniform_(self.u_hyper)
        nn.init.xavier_uniform_(self.i_hyper)
        # Per-behavior FC tren hyperedge (giong MixRec FC sau hyperEdge agg).
        # Init nho (std=0.02) thay vi xavier default: residual luc dau co magnitude
        # nho de khong destabilize, NHUNG khac 0 de gradient van flow den u_hyper.
        # Zero-init -> u_hyper.grad=0 vinh vien (verified) -> sai.
        self.fc_u = nn.ModuleList(
            [nn.Linear(hyper_num, hyper_num, bias=False) for _ in self.BEHAVIORS]
        )
        self.fc_i = nn.ModuleList(
            [nn.Linear(hyper_num, hyper_num, bias=False) for _ in self.BEHAVIORS]
        )
        for fc in list(self.fc_u) + list(self.fc_i):
            nn.init.normal_(fc.weight, mean=0.0, std=2e-2)
        # Mix weights per behavior (softmax). Init bias purchase nang hon.
        self.beh_w_u = nn.Parameter(torch.tensor([0.0, 0.2, 0.4]))
        self.beh_w_i = nn.Parameter(torch.tensor([0.0, 0.2, 0.4]))

    def forward(
        self,
        u_emb: Tensor,
        i_emb: Tensor,
        u_emb_0: Tensor,
        i_emb_0: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        # Soft hyperedge assignment (giong MixRec uuHyper = uEmbed0 @ uhyper).
        uuH = u_emb_0 @ self.u_hyper       # [Nu, H]
        iiH = i_emb_0 @ self.i_hyper       # [Ni, H]

        wu = F.softmax(self.beh_w_u, dim=0)
        wi = F.softmax(self.beh_w_i, dim=0)

        u_res = u_emb.new_zeros(u_emb.shape)
        i_res = i_emb.new_zeros(i_emb.shape)
        for b in range(len(self.BEHAVIORS)):
            # User hyperedge: edge = FC(ReLU(uuH^T @ u_emb))
            edge_u = F.relu(uuH.transpose(0, 1) @ u_emb)   # [H, dim]
            edge_u = self.fc_u[b](edge_u.transpose(0, 1)).transpose(0, 1)  # [H, dim]
            u_res = u_res + wu[b] * (uuH @ edge_u)         # [Nu, dim]

            edge_i = F.relu(iiH.transpose(0, 1) @ i_emb)   # [H, dim]
            edge_i = self.fc_i[b](edge_i.transpose(0, 1)).transpose(0, 1)
            i_res = i_res + wi[b] * (iiH @ edge_i)

        return u_res, i_res


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
        hyper_num: int = 128,
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
        # Hypergraph propagation per layer (MixRec hyperPropagate).
        self.hypers = nn.ModuleList(
            [HyperPropagation(dim=dims[i + 1], hyper_num=hyper_num) for i in range(n_layers)]
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
        all_h: list = [x_dict]  # h^0 included for LightGCN-style mean pooling

        # MixRec hypergraph dung embedding GOC (h^0) lam soft assignment.
        u0 = x_dict.get("user")
        i0 = x_dict.get("product")

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

            # Hypergraph residual (giong MixRec: ulats[i+1] = ulat + hyperULat + ulats[i]).
            if u0 is not None and "user" in h and i0 is not None and "product" in h:
                u_res, i_res = self.hypers[i](h["user"], h["product"], u0, i0)
                h = dict(h)
                h["user"] = h["user"] + u_res
                h["product"] = h["product"] + i_res

            all_h.append(h)

            if i < len(self.convs) - 1:
                h = {t: self.dropout(v) for t, v in h.items()}

        # Mean pooling across h^0 .. h^L (LightGCN-style aggregation)
        h_out: Dict[str, Tensor] = {}
        for t in all_h[-1].keys():
            layers = [lh[t] for lh in all_h if t in lh]
            h_out[t] = torch.stack(layers, dim=0).mean(dim=0)

        return h_out, beh_user_agg
    
class IntentCodebook(nn.Module):
    """Per-behavior decoupled intent codebooks (MixRec H^(u)_k style).

    Mỗi behavior (view/cart/purchase) có codebook riêng cho user và item.
    Kết quả attention 3 codebook được trộn bằng learnable behavior weights
    (softmax) -> residual cộng vào embedding gốc.

    So với codebook đơn shared trước đây, kiến trúc này có ~3x tham số intent
    và biểu diễn được intent đặc thù theo behavior — giống cách MixRec đạt
    SOTA trên REES46.
    """

    BEHAVIORS = ("view", "cart", "purchase")

    def __init__(self, n_intents: int = 32, dim: int = EMBED_DIM):
        super().__init__()
        self.n_intents = n_intents
        self.dim = dim
        self.user_intents = nn.ParameterDict({
            b: nn.Parameter(torch.empty(n_intents, dim)) for b in self.BEHAVIORS
        })
        self.item_intents = nn.ParameterDict({
            b: nn.Parameter(torch.empty(n_intents, dim)) for b in self.BEHAVIORS
        })
        for p in self.user_intents.values():
            nn.init.xavier_uniform_(p)
        for p in self.item_intents.values():
            nn.init.xavier_uniform_(p)
        # learnable behavior mixing weights (softmax-normalized).
        # init: purchase > cart > view (final intent should weigh strongest signal more).
        self.beh_w_user = nn.Parameter(torch.tensor([0.0, 0.3, 0.6]))
        self.beh_w_item = nn.Parameter(torch.tensor([0.0, 0.3, 0.6]))
        self._scale = dim ** -0.5

    def _attend(self, x: Tensor, codebook: Tensor) -> Tensor:
        attn = (x @ codebook.T) * self._scale
        attn = torch.softmax(attn, dim=-1)
        return attn @ codebook

    def _mix(
        self,
        x: Tensor,
        codebooks: nn.ParameterDict,
        weights: Tensor,
    ) -> Tensor:
        w = F.softmax(weights, dim=0)
        out = x.new_zeros(x.shape)
        for i, b in enumerate(self.BEHAVIORS):
            out = out + w[i] * self._attend(x, codebooks[b])
        return out

    def forward(self, x_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        out = dict(x_dict)
        if "user" in out:
            out["user"] = out["user"] + self._mix(
                out["user"], self.user_intents, self.beh_w_user
            )
        if "product" in out:
            out["product"] = out["product"] + self._mix(
                out["product"], self.item_intents, self.beh_w_item
            )
        return out


class BPATMPModel(nn.Module):
    """Full BPATMP model for training with embeddings and message passing."""

    def __init__(
        self,
        num_nodes_dict: Dict[str, int],
        embed_dim: int = EMBED_DIM,
        n_layers: int = 2,
        dropout: float = 0.1,
        n_intents: int = 32,
        rank: int = 16,
        use_grad_checkpoint: bool = True,
        n_freqs: int = 16,
        tau: float = 7.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_nodes_dict = num_nodes_dict

        self.input_proj = nn.ModuleDict({
            ntype: nn.Embedding(num_nodes, embed_dim)
            for ntype, num_nodes in num_nodes_dict.items()
        })
        for emb in self.input_proj.values():
            nn.init.xavier_uniform_(emb.weight)

        self.beh_proj = nn.ModuleDict({
            beh: nn.Linear(embed_dim, embed_dim, bias=False)
            for beh in ("view", "cart", "purchase")
        })

        self.encoder = BPATMPLayer(
            in_dim=embed_dim,
            hid_dim=embed_dim,
            out_dim=embed_dim,
            n_layers=n_layers,
            rank=rank,
            dropout=dropout,
            use_checkpoint=use_grad_checkpoint,
            n_freqs=n_freqs,
            tau=tau,
        )

        self.intent_codebook = IntentCodebook(n_intents=n_intents, dim=embed_dim)

        self.item_bias = nn.Embedding(num_nodes_dict["product"], 1)
        nn.init.zeros_(self.item_bias.weight)

    def embedding_l2_norm(self) -> Tensor:
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        for emb in self.input_proj.values():
            total = total + emb.weight.pow(2).sum()
        total = total + self.item_bias.weight.pow(2).sum()
        return total

    def forward(
        self,
        subgraph,
        return_beh_embs: bool = False,
        ref_time: Optional[float] = None,
    ) -> Tuple[Tensor, Tensor, Optional[Dict[str, Tensor]]]:
        x_dict = {}
        for ntype, emb in self.input_proj.items():
            if ntype in subgraph.node_types and hasattr(subgraph[ntype], "x"):
                node_ids = subgraph[ntype].x
                x_dict[ntype] = emb(node_ids)

        edge_index_dict = {}
        edge_ts_dict = {}
        edge_attr_dict = {}
        for edge_type in subgraph.edge_types:
            store = subgraph[edge_type]
            if hasattr(store, "edge_index") and store.edge_index.numel() > 0:
                edge_index_dict[edge_type] = store.edge_index
                if hasattr(store, "edge_ts"):
                    edge_ts_dict[edge_type] = store.edge_ts
                elif hasattr(store, "ts"):
                    edge_ts_dict[edge_type] = store.ts
                if hasattr(store, "edge_attr"):
                    edge_attr_dict[edge_type] = store.edge_attr

        h_dict, beh_user_agg = self.encoder(
            x_dict,
            edge_index_dict,
            edge_attr_dict if edge_attr_dict else None,
            edge_ts_dict if edge_ts_dict else None,
            ref_time,
        )
        h_dict = self.intent_codebook(h_dict)

        user_emb = h_dict.get("user", torch.zeros(0, self.embed_dim))
        item_emb = h_dict.get("product", torch.zeros(0, self.embed_dim))

        if return_beh_embs:
            beh_embs = {
                beh: self.beh_proj[beh](beh_user_agg.get(beh, torch.zeros_like(user_emb)))
                for beh in ("view", "cart", "purchase")
            }
            return user_emb, item_emb, beh_embs

        return user_emb, item_emb
