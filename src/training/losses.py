import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast


class PopularityBiasedNegativeSampler:
    def __init__(
        self,
        item_counts: dict[str, torch.Tensor],
        num_items: int,
        alpha: float = 0.75,
        device: str = "cpu",
    ):
        self.num_items = num_items
        self.alpha = alpha
        self.device = device
        self._distributions: dict[str, torch.Tensor] = {}

        for beh_name, counts in item_counts.items():
            assert counts.shape[0] == num_items, (
                f"item_counts['{beh_name}'] has {counts.shape[0]} entries, expected {num_items}"
            )
            smoothed = (counts.float() + 1.0).pow(alpha)
            prob = smoothed / smoothed.sum()
            self._distributions[beh_name] = prob.to(device)

        if not self._distributions:
            self._distributions["global"] = torch.full(
                (self.num_items,), 1.0 / self.num_items, device=self.device
            )
        else:
            global_prob = torch.stack(list(self._distributions.values())).mean(dim=0)
            self._distributions["global"] = (global_prob / global_prob.sum()).to(device)

    def sample(
        self,
        batch_size: int,
        num_neg: int = 1,
        behavior: str | None = None,
        exclude_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        key = behavior if behavior and behavior in self._distributions else "global"
        neg_flat = torch.multinomial(self._distributions[key], batch_size * num_neg, replacement=True)
        return neg_flat.view(batch_size, num_neg)


def bpr_loss(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
) -> torch.Tensor:
    if pos_scores.dim() == 1:
        pos_scores = pos_scores.unsqueeze(-1)
    if neg_scores.dim() == 1:
        neg_scores = neg_scores.unsqueeze(-1)
    return -F.logsigmoid(pos_scores - neg_scores).mean()


class MultiTaskBPRLoss(nn.Module):
    """Per-behavior BPR with count-aware weights:

        w_b = clip((N_purchase / N_b) ** alpha, w_min, 1.0)

    purchase weight is 1.0 by construction. View/cart get scaled down because
    REES46 has ~100x more views than purchases.
    """

    BEHAVIOR_ORDER = ["view", "cart", "purchase"]

    def __init__(
        self,
        behavior_counts: dict[str, int],
        l2_lambda: float = 1e-5,
        alpha: float = 0.5,
        w_min: float = 0.05,
    ):
        super().__init__()
        self.l2_lambda = l2_lambda
        n_purchase = max(int(behavior_counts.get("purchase", 1)), 1)
        weights = []
        for b in self.BEHAVIOR_ORDER:
            n_b = max(int(behavior_counts.get(b, 1)), 1)
            w = (n_purchase / n_b) ** alpha
            w = max(min(w, 1.0), w_min)
            weights.append(w)
        self.register_buffer(
            "task_weights", torch.tensor(weights, dtype=torch.float32)
        )

    @autocast("cuda", enabled=False)
    def forward(
        self,
        behavior_losses: dict[str, torch.Tensor],
        model_params: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        log_dict = {}
        total = torch.tensor(0.0, device=self.task_weights.device)
        n_present = 0

        for idx, beh in enumerate(self.BEHAVIOR_ORDER):
            if beh not in behavior_losses:
                continue
            n_present += 1
            beh_loss = behavior_losses[beh].float()
            w = self.task_weights[idx]
            total = total + w * beh_loss
            log_dict[f"loss/{beh}"] = beh_loss.item()
            log_dict[f"weight/{beh}"] = w.item()

        if n_present > 0:
            total = total * (len(self.BEHAVIOR_ORDER) / n_present)

        if model_params is not None and self.l2_lambda > 0:
            l2_term = self.l2_lambda * model_params.float()
            total = total + l2_term
            log_dict["loss/l2"] = l2_term.item()

        log_dict["loss/total"] = total.item()
        return total, log_dict


class HierarchicalMBCL(nn.Module):
    """Directional, hierarchical InfoNCE with per-pair gating (Step 2.5).

    Each (weak, strong, weight) pair computes its InfoNCE on the intersection
    of users that have both behaviors in the current batch — independently of
    other pairs. A user with view+cart but no purchase still contributes to
    the (view, cart) loss.
    """
    def __init__(
        self,
        tau: float = 0.1,
        behaviors: list[str] | None = None,
        pair_weights: list[tuple[str, str, float]] | None = None,
        hard_k: int = 32,
        min_pair_overlap: int = 4,
    ) -> None:
        super().__init__()
        from src.core.contracts import BEHAVIOR_TYPES
        self.tau = tau
        self.hard_k = hard_k
        self.min_pair_overlap = min_pair_overlap
        self.behaviors = list(behaviors) if behaviors else list(BEHAVIOR_TYPES)
        K = len(self.behaviors)

        if pair_weights is None:
            cons_w = (
                torch.linspace(0.2, 1.0, K - 1).tolist() if K > 1 else []
            )
            pair_weights = [
                (self.behaviors[i], self.behaviors[i + 1], cons_w[i])
                for i in range(K - 1)
            ]
            if K >= 2:
                pair_weights.append((self.behaviors[0], self.behaviors[-1], 1.0))
        self.pair_weights = list(pair_weights)

    @staticmethod
    def _normalize(x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=-1)

    def _directional(self, z_weak: torch.Tensor, z_strong: torch.Tensor) -> torch.Tensor:
        N = z_strong.size(0)
        if N < 2:
            return z_strong.new_zeros(())
        z_w = self._normalize(z_weak.detach())
        z_s = self._normalize(z_strong)
        sim = (z_s @ z_w.T) / self.tau
        labels = torch.arange(N, device=z_s.device)
        n_hard = min(self.hard_k, N - 1)
        if 0 < n_hard < N - 1:
            pos_logit = sim.gather(1, labels.unsqueeze(1))
            neg_only = sim.clone()
            neg_only.scatter_(1, labels.unsqueeze(1), float("-inf"))
            hard_neg, _ = neg_only.topk(n_hard, dim=-1)
            sim = torch.cat([pos_logit, hard_neg], dim=-1)
            labels = torch.zeros(N, dtype=torch.long, device=z_s.device)
        return F.cross_entropy(sim, labels)

    def forward(
        self,
        beh_embs: dict[str, torch.Tensor],
        users_per_beh: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        first = next(iter(beh_embs.values()))
        loss = first.new_zeros(())

        for weak, strong, w in self.pair_weights:
            u_w = users_per_beh.get(weak)
            u_s = users_per_beh.get(strong)
            if u_w is None or u_s is None or u_w.numel() == 0 or u_s.numel() == 0:
                continue
            common_pair = u_w[torch.isin(u_w, u_s)]
            if common_pair.numel() < self.min_pair_overlap:
                continue
            z_w = beh_embs[weak][common_pair]
            z_s = beh_embs[strong][common_pair]
            loss = loss + w * self._directional(z_w, z_s)

        return loss


def build_user_history_csr(
    triplets: torch.Tensor,
    n_users: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CSR over (user -> seen items) across ALL behaviors so train-time
    negatives can be masked the same way eval's exclude_items does."""
    user = triplets[:, 0].long()
    item = triplets[:, 1].long()
    order = user.argsort()
    user_s = user[order]
    item_s = item[order]
    counts = torch.bincount(user_s, minlength=n_users)
    ptr = torch.zeros(n_users + 1, dtype=torch.long)
    ptr[1:] = counts.cumsum(0)
    return ptr, item_s


def sample_aligned_negatives_local(
    pp_b: torch.Tensor,            # (B,) local pos positions in subgraph
    user_b_global: torch.Tensor,   # (B,) global user ids
    pos_b_global: torch.Tensor,    # (B,) global pos item ids
    N_items: int,                  # subgraph item count
    num_neg: int,
    prod_x: torch.Tensor,          # (N_items,) subgraph item -> global id
    pop_dist_global: torch.Tensor, # (n_items_global,)
    history_ptr: torch.Tensor,
    history_item: torch.Tensor,
    user_emb_b: torch.Tensor,      # (B, d) DETACHED
    item_emb_local: torch.Tensor,  # (N_items, d) DETACHED
    frac_random: float = 0.25,
    frac_pop: float = 0.25,
) -> torch.Tensor:                 # (B, num_neg) LOCAL positions in subgraph
    """Mixed-strategy negatives in subgraph-local index space, with global
    history masking. Distribution: uniform | popularity | in-batch hard."""
    device = pp_b.device
    B = pp_b.size(0)
    if N_items <= 1 or B == 0:
        return pp_b.unsqueeze(1).expand(B, num_neg).contiguous()

    n_rand = max(1, int(num_neg * frac_random))
    n_pop = max(1, int(num_neg * frac_pop))
    n_hard = max(0, num_neg - n_rand - n_pop)

    pop_local = pop_dist_global[prod_x.long()].clone()
    pop_local = pop_local / (pop_local.sum() + 1e-12)

    rand_negs = torch.randint(0, N_items, (B, n_rand), device=device)
    pop_negs = torch.multinomial(
        pop_local, B * n_pop, replacement=True
    ).view(B, n_pop)

    if n_hard > 0:
        with torch.no_grad():
            scores = user_emb_b @ item_emb_local.T
            scores = scores.scatter(
                1, pp_b.unsqueeze(1), float("-inf")
            )
            k = min(n_hard, N_items - 1)
            _, hard_negs = scores.topk(k, dim=-1)
            if k < n_hard:
                pad = torch.randint(
                    0, N_items, (B, n_hard - k), device=device
                )
                hard_negs = torch.cat([hard_negs, pad], dim=-1)
    else:
        hard_negs = torch.empty((B, 0), dtype=torch.long, device=device)

    negs_local = torch.cat([rand_negs, pop_negs, hard_negs], dim=-1)

    starts = history_ptr[user_b_global]
    ends = history_ptr[user_b_global + 1]
    lens = ends - starts
    max_len = int(lens.max().item()) if lens.numel() > 0 else 0

    if max_len > 0:
        offsets = torch.arange(max_len, device=device)
        pad_idx = (starts.unsqueeze(1) + offsets.unsqueeze(0)).clamp(
            max=history_item.size(0) - 1
        )
        valid = offsets.unsqueeze(0) < lens.unsqueeze(1)
        seen = history_item[pad_idx]
        seen = seen.masked_fill(~valid, -1)

        for _ in range(2):
            negs_global = prod_x[negs_local.clamp(max=N_items - 1).long()]
            bad = (
                negs_global.unsqueeze(2) == seen.unsqueeze(1)
            ).any(dim=-1)
            if not bad.any():
                break
            repl = torch.randint(0, N_items, bad.shape, device=device)
            negs_local = torch.where(bad, repl, negs_local)

    same = negs_local == pp_b.unsqueeze(1)
    if same.any():
        repl = torch.randint(0, N_items, negs_local.shape, device=device)
        negs_local = torch.where(same, repl, negs_local)

    return negs_local


class FunnelPriorLoss(nn.Module):
    """Conversion-funnel score-ordering prior.

    Enforces s_strong >= s_weak + margin for every consecutive pair in the
    purchase funnel (view -> cart -> purchase):

        L_conv = mean( relu(s_weak - s_strong + margin) ** 2 )

    If the ordering is already satisfied the loss is exactly zero.
    """

    FUNNEL_ORDER: list[str] = ["view", "cart", "purchase"]

    def __init__(self, margin: float = 0.1) -> None:
        super().__init__()
        self.margin = margin
        self.pairs: list[tuple[str, str]] = [
            (self.FUNNEL_ORDER[i], self.FUNNEL_ORDER[i + 1])
            for i in range(len(self.FUNNEL_ORDER) - 1)
        ]

    def forward(
        self,
        scores: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute funnel prior loss.

        Args:
            scores: ``{behavior_name: (B,) score tensor}`` for the **same**
                    set of users/items.  Only consecutive pairs present in
                    *scores* contribute to the loss.

        Returns:
            Scalar loss (zero when all ordering constraints are satisfied).
        """
        device = next(iter(scores.values())).device
        loss = torch.tensor(0.0, device=device)
        n_pairs = 0

        for weak, strong in self.pairs:
            if weak not in scores or strong not in scores:
                continue
            s_weak = scores[weak].float()
            s_strong = scores[strong].float()
            violation = F.relu(s_weak - s_strong + self.margin)
            loss = loss + (violation ** 2).mean()
            n_pairs += 1

        if n_pairs > 0:
            loss = loss / n_pairs

        return loss


class MonotonicDecayPriorLoss(nn.Module):
    """Monotonic temporal-decay ordering prior.

    Enforces lam_weak >= lam_strong so that weaker signals (e.g. view) fade
    faster than stronger ones (e.g. cart, purchase):

        L_mono = mean( relu(lam_strong - lam_weak) ** 2 )

    Operates on per-behavior decay rates (post-softplus, positive) used in
    the temporal attention where decay = lam * log(1 + dt / tau).
    """

    DECAY_ORDER: list[str] = ["view", "cart", "purchase"]

    def forward(
        self,
        lambdas: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute monotonic-decay prior loss.

        Args:
            lambdas: ``{behavior_name: scalar or (B,) decay-rate tensor}``.
                     Values should be *post-softplus* (positive).

        Returns:
            Scalar loss (zero when lam_view >= lam_cart >= lam_purchase).
        """
        device = next(iter(lambdas.values())).device
        loss = torch.tensor(0.0, device=device)
        n_pairs = 0

        for i in range(len(self.DECAY_ORDER) - 1):
            weak = self.DECAY_ORDER[i]
            strong = self.DECAY_ORDER[i + 1]
            if weak not in lambdas or strong not in lambdas:
                continue
            lam_weak = lambdas[weak].float()
            lam_strong = lambdas[strong].float()
            # weak signal must decay faster: lam_weak >= lam_strong
            violation = F.relu(lam_strong - lam_weak)
            loss = loss + (violation ** 2).mean()
            n_pairs += 1

        if n_pairs > 0:
            loss = loss / n_pairs

        return loss


class BPATMPTotalLoss(nn.Module):
    """Combined loss for the BPATMP model.

    Aggregates every training objective into a single call::

        L = L_bpr + lambda_cl * L_CL + lambda_conv * L_conv
            + lambda_mono * L_mono + lambda_wd * ||theta||^2

    Each sub-loss is optional: if the corresponding input is ``None`` or
    empty, that term is silently skipped (and logged as 0).

    Args:
        behavior_counts: ``{"view": N, "cart": N, "purchase": N}`` for
            :class:`MultiTaskBPRLoss` count-weighted normalisation.
        lambda_cl:   coefficient for :class:`HierarchicalMBCL`.
        lambda_conv: coefficient for :class:`FunnelPriorLoss`.
        lambda_mono: coefficient for :class:`MonotonicDecayPriorLoss`.
        lambda_wd:   L2 weight-decay coefficient on model parameters.
        margin:      margin ``m`` for :class:`FunnelPriorLoss`.
        tau:         temperature for :class:`HierarchicalMBCL`.
    """

    def __init__(
        self,
        behavior_counts: dict[str, int],
        lambda_cl: float = 0.1,
        lambda_conv: float = 0.0,
        lambda_mono: float = 0.0,
        lambda_wd: float = 1e-5,
        margin: float = 0.1,
        tau: float = 0.1,
        alpha: float = 0.5,
        w_min: float = 0.05,
        cl_hard_k: int = 32,
        cl_min_pair_overlap: int = 4,
        cl_pair_weights: list[tuple[str, str, float]] | None = None,
    ) -> None:
        super().__init__()

        self.lambda_cl = lambda_cl
        self.lambda_conv = lambda_conv
        self.lambda_mono = lambda_mono
        self.lambda_wd = lambda_wd

        # Sub-modules — L2 handled here, so BPR gets l2_lambda=0
        self.bpr = MultiTaskBPRLoss(
            behavior_counts, l2_lambda=0.0, alpha=alpha, w_min=w_min
        )
        self.cl = HierarchicalMBCL(
            tau=tau,
            pair_weights=cl_pair_weights,
            hard_k=cl_hard_k,
            min_pair_overlap=cl_min_pair_overlap,
        )
        self.funnel = FunnelPriorLoss(margin=margin)
        self.mono = MonotonicDecayPriorLoss()

    @autocast("cuda", enabled=False)
    def forward(
        self,
        behavior_losses: dict[str, torch.Tensor],
        beh_embs: dict[str, torch.Tensor] | None = None,
        users_per_beh: dict[str, torch.Tensor] | None = None,
        scores: dict[str, torch.Tensor] | None = None,
        lambdas: dict[str, torch.Tensor] | None = None,
        model_params: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the total BPATMP loss.

        Args:
            behavior_losses: per-behavior BPR losses ``{"view": ..., ...}``.
            beh_embs: per-behavior user embeddings for contrastive learning.
            users_per_beh: user-id tensors per behavior (for CL overlap).
            scores: per-behavior prediction scores (for funnel prior).
            lambdas: per-behavior decay rates (for monotonic prior).
            model_params: pre-computed ``sum(p.pow(2).sum() for p in params)``
                          i.e. the squared L2 norm of all model parameters.

        Returns:
            ``(total_loss, log_dict)`` where *log_dict* maps every
            component name to its scalar value for W&B / TensorBoard.
        """
        device = next(iter(behavior_losses.values())).device
        log_dict: dict[str, float] = {}

        # 1. BPR (main ranking loss)
        l_bpr, bpr_log = self.bpr(behavior_losses, model_params=None)
        bpr_log.pop("loss/total", None)
        log_dict.update(bpr_log)
        log_dict["loss/bpr"] = l_bpr.item()

        total = l_bpr

        # 2. Contrastive learning
        if (
            self.lambda_cl > 0
            and beh_embs is not None
            and users_per_beh is not None
            and len(beh_embs) > 0
        ):
            l_cl = self.cl(beh_embs, users_per_beh)
            total = total + self.lambda_cl * l_cl
            log_dict["loss/cl"] = l_cl.item()
        else:
            log_dict["loss/cl"] = 0.0

        # 3. Funnel prior
        if self.lambda_conv > 0 and scores is not None and len(scores) > 0:
            l_conv = self.funnel(scores)
            total = total + self.lambda_conv * l_conv
            log_dict["loss/conv"] = l_conv.item()
        else:
            log_dict["loss/conv"] = 0.0

        # 4. Monotonic decay prior
        if self.lambda_mono > 0 and lambdas is not None and len(lambdas) > 0:
            l_mono = self.mono(lambdas)
            total = total + self.lambda_mono * l_mono
            log_dict["loss/mono"] = l_mono.item()
        else:
            log_dict["loss/mono"] = 0.0

        # 5. Weight decay (L2)
        if self.lambda_wd > 0 and model_params is not None:
            l_wd = self.lambda_wd * model_params.float()
            total = total + l_wd
            log_dict["loss/wd"] = l_wd.item()
        else:
            log_dict["loss/wd"] = 0.0

        log_dict["loss/total"] = total.item()
        return total, log_dict
