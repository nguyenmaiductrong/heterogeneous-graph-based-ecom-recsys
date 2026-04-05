import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
import numpy as np

from src.core.contracts import BEHAVIOR_LOSS_WEIGHTS


class PopularityBiasedNegativeSampler:
    """
    Tạo ra các mẫu Negative Samples để dạy model theo độ phổ biến
    """
    def __init__(
        self,
        item_counts: dict[str, torch.Tensor], # số lần item i được tương tác 
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
    """
    L_total = Σ_b  w_b * BPR_b  +  λ * ||Θ||²
    Trong đó w_b là task weight cho behavior b (w_b ∝ 1/sqrt(N_b) — tự động cân bằng theo tần suất.)
    """

    BEHAVIOR_ORDER = ["view", "cart", "purchase"]

    def __init__(self, behavior_counts: dict[str, int], l2_lambda: float = 1e-5):
        super().__init__()
        self.l2_lambda = l2_lambda
        raw = torch.tensor(
            [BEHAVIOR_LOSS_WEIGHTS[b] / np.sqrt(behavior_counts[b]) for b in self.BEHAVIOR_ORDER],
            dtype=torch.float32,
        )
        # Chuẩn hóa để tổng weights = số lượng behaviors
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
        
        # Cộng dồn loss có trọng số
        for idx, beh in enumerate(self.BEHAVIOR_ORDER):
            if beh not in behavior_losses:
                continue
            beh_loss = behavior_losses[beh].float()
            w = self.task_weights[idx]
            total = total + w * beh_loss
            log_dict[f"loss/{beh}"] = beh_loss.item()
            log_dict[f"weight/{beh}"] = w.item()

        if model_params is not None and self.l2_lambda > 0:
            l2_term = self.l2_lambda * model_params.float()
            total = total + l2_term
            log_dict["loss/l2"] = l2_term.item()

        log_dict["loss/total"] = total.item()
        return total, log_dict
