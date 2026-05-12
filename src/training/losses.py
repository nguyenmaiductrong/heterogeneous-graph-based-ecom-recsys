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


def sampled_softmax_loss(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Sampled-softmax cross-entropy ranking loss.

        ℓ = -log( exp(s⁺/T) / ( exp(s⁺/T) + Σ_j exp(s⁻_j/T) ) )

    Uses all sampled negatives jointly (one softmax over 1 + num_neg logits),
    averaged over the batch. ``pos_scores`` is ``(B,)`` or ``(B, 1)``;
    ``neg_scores`` is ``(B, num_neg)``.
    """
    if pos_scores.dim() == 2:
        pos_scores = pos_scores.squeeze(-1)
    if neg_scores.dim() == 1:
        neg_scores = neg_scores.unsqueeze(-1)
    logits = torch.cat([pos_scores.unsqueeze(-1), neg_scores], dim=-1) / temperature
    target = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, target)


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


class BPATMPTotalLoss(nn.Module):
    """Combined training loss for the BPATMP model.

        L = Σ_b  w_b · CE_b   +   λ_emb · R_emb

    * ``CE_b``: per-behavior sampled-softmax cross-entropy (computed upstream
      with the model scores and passed in as a scalar).
    * ``w_b = clip((N_purchase / N_b) ** alpha, w_min, 1.0)`` — count-aware
      down-weighting of view/cart (REES46 has ~100x more views than purchases).
      ``purchase`` weight is 1.0 by construction. The weighted sum is rescaled
      by ``len(BEHAVIOR_ORDER) / n_present`` so a missing behavior in a batch
      doesn't shrink the gradient.
    * ``R_emb``: mean squared L2 norm of the user / positive-item / negative-item
      vectors actually used to score this batch — a cheap "active-rows-only"
      regulariser on the embeddings that bounds score magnitude and curbs the
      overfitting seen with the huge user table.
    """

    BEHAVIOR_ORDER = ["view", "cart", "purchase"]

    def __init__(
        self,
        behavior_counts: dict[str, int],
        lambda_emb: float = 1e-3,
        alpha: float = 0.5,
        w_min: float = 0.05,
    ) -> None:
        super().__init__()
        self.lambda_emb = lambda_emb
        n_purchase = max(int(behavior_counts.get("purchase", 1)), 1)
        weights = []
        for b in self.BEHAVIOR_ORDER:
            n_b = max(int(behavior_counts.get(b, 1)), 1)
            w = (n_purchase / n_b) ** alpha
            w = max(min(w, 1.0), w_min)
            weights.append(w)
        self.register_buffer("task_weights", torch.tensor(weights, dtype=torch.float32))

    @autocast("cuda", enabled=False)
    def forward(
        self,
        behavior_losses: dict[str, torch.Tensor],
        embedding_reg: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        log_dict: dict[str, float] = {}
        total = torch.zeros((), device=self.task_weights.device)
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
        log_dict["loss/ce"] = total.item()

        if self.lambda_emb > 0 and embedding_reg is not None:
            reg = self.lambda_emb * embedding_reg.float()
            total = total + reg
            log_dict["loss/emb_reg"] = reg.item()
        else:
            log_dict["loss/emb_reg"] = 0.0

        log_dict["loss/total"] = total.item()
        return total, log_dict
