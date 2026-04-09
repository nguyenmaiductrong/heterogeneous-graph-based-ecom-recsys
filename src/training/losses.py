import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
import numpy as np

from src.core.contracts import BEHAVIOR_LOSS_WEIGHTS


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


class ContrastiveLearning(nn.Module):
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        N = z1.size(0)
        sim = torch.mm(z1, z2.T) / self.temperature
        labels = torch.arange(N, device=z1.device)
        return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2


class MultiTaskBPRLoss(nn.Module):
    BEHAVIOR_ORDER = ["view", "cart", "purchase"]

    def __init__(self, behavior_counts: dict[str, int], l2_lambda: float = 1e-5):
        super().__init__()
        self.l2_lambda = l2_lambda
        raw = torch.tensor(
            [BEHAVIOR_LOSS_WEIGHTS[b] / np.sqrt(behavior_counts[b]) for b in self.BEHAVIOR_ORDER],
            dtype=torch.float32,
        )
        normalized = raw / raw.sum() * len(self.BEHAVIOR_ORDER)
        self.register_buffer("task_weights", normalized)

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
