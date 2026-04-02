import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
import numpy as np

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
                f"item_counts['{beh_name}'] has {counts.shape[0]} entries,"
                f"expected {num_items}"
            )
            smoothed = (counts.float() + 1.0).pow(alpha)
            prob = smoothed / smoothed.sum()
            self._distributions[beh_name] = prob.to(device)

    def sample(
        self,
        batch_size: int,
        num_neg: int = 1,
        behavior: str | None = None,
        exclude_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        key = behavior if behavior and behavior in self._distributions else "global"
        dist = self._distributions[key]

        total_samples = batch_size * num_neg
        neg_flat = torch.multinomial(
            dist, total_samples, replacement=True
        )
        return neg_flat.view(batch_size, num_neg)

def bpr_loss(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
) -> torch.Tensor:
    if pos_scores.dim() == 1:
        pos_scores = pos_scores.unsqueeze(-1)
    if neg_scores.dim() == 1:
        neg_scores = neg_scores.unsqueeze(-1)
    
    diff = pos_scores - neg_scores  # [B, K]
    loss = -F.logsigmoid(diff).mean()
    return loss

class MultiTaskBPRLoss(nn.Module):
    """
    L_total = Σ_b  w_b * BPR_b  +  λ * ||Θ||²
    Trong đó w_b là task weight cho behavior b (w_b ∝ 1/sqrt(N_b) — tự động cân bằng theo tần suất.)
    """
    BEHAVIOR_ORDER = ["view", "cart", "purchase"]
    
    def __init__(
        self,
        behavior_counts: dict[str, int],
        l2_lambda: float = 1e-5,
    ):
        super().__init__()
        self.l2_lambda = l2_lambda
        # Tính toán Inverse Frequency weights
        raw = torch.tensor(
            [1.0 / np.sqrt(behavior_counts[b]) for b in self.BEHAVIOR_ORDER],
            dtype=torch.float32,
        )
        # Chuẩn hóa để tổng weights = số lượng behaviors
        normalized = raw / raw.sum() * len(self.BEHAVIOR_ORDER)
        # Tạo thành buffer 
        self.register_buffer("task_weights", normalized)

    @autocast('cuda', enabled=False)
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
            
            total = total + (w * beh_loss)

            log_dict[f"loss/{beh}"] = beh_loss.item()
            log_dict[f"weight/{beh}"] = w.item()

        # Cộng thêm L2 Regularization
        if model_params is not None and self.l2_lambda > 0:
            l2_term = self.l2_lambda * model_params.float()
            total = total + l2_term
            log_dict["loss/l2"] = l2_term.item()

        log_dict["loss/total"] = total.item()
        return total, log_dict

class BPRTrainingStep:
    def __init__(
        self,
        multi_task_loss: MultiTaskBPRLoss,
        neg_sampler: PopularityBiasedNegativeSampler,
        num_neg: int = 1,
        max_grad_norm: float = 1.0,
        amp_enabled: bool = True,
    ):
        self.multi_task_loss = multi_task_loss
        self.neg_sampler = neg_sampler
        self.num_neg = num_neg
        self.max_grad_norm = max_grad_norm
        self.amp_enabled = amp_enabled

        if amp_enabled:
            self.scaler = torch.amp.GradScaler('cuda',
                init_scale=2**16,       # Start with moderate scale
                growth_factor=2.0,
                backoff_factor=0.5,
                growth_interval=2000,   # Grow scale every 2000 steps without inf/nan
            )

    def compute_scores(
        self,
        user_emb: torch.Tensor,
        item_emb_all: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute pos/neg scores via dot product.
        
        Args:
            user_emb:     [B, d] — user embeddings (from GNN encoder)
            item_emb_all: [N, d] — ALL item embeddings
            pos_items:    [B] — positive item indices
            neg_items:    [B, K] — negative item indices
            
        Returns:
            pos_scores: [B, 1]
            neg_scores: [B, K]
        """
        pos_emb = item_emb_all[pos_items] # [B, d]
        neg_emb = item_emb_all[neg_items] # [B, K, d]

        # Dot product
        pos_scores = (user_emb * pos_emb).sum(dim=-1, keepdim=True)  # [B, 1]
        neg_scores = torch.bmm(
            neg_emb, user_emb.unsqueeze(-1) # [B, K, d] x [B, d, 1]
        ).squeeze(-1)                       # [B, K]

        return pos_scores, neg_scores

    def step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        batch: dict[str, dict[str, torch.Tensor]],
        user_emb: torch.Tensor,
        item_emb_all: torch.Tensor,
    ) -> dict[str, float]: 
        optimizer.zero_grad(set_to_none=True) 

        with torch.amp.autocast('cuda', enabled=self.amp_enabled):
            behavior_losses = {}

            for beh in MultiTaskBPRLoss.BEHAVIOR_ORDER:
                if beh not in batch:
                    continue

                users = batch[beh]["user"]       # [B_beh]
                pos_items = batch[beh]["pos_item"]  # [B_beh]
                B_beh = users.shape[0]

                # Negative sampling
                neg_items = self.neg_sampler.sample(
                    batch_size=B_beh,
                    num_neg=self.num_neg,
                    behavior=beh,
                ).to(users.device)  # [B_beh, K]

                # Score computation
                u_emb = user_emb[users]  # [B_beh, d]
                pos_scores, neg_scores = self.compute_scores(
                    u_emb, item_emb_all, pos_items, neg_items
                )

                behavior_losses[beh] = bpr_loss(pos_scores, neg_scores)

        # L2 regularization on embeddings only (not all params — saves compute)
        # Chỉ regularize initial embeddings, không regularize GNN weights (theo LightGCN, CRGCN convention)
        l2_norm = None
        if hasattr(model, 'embedding_l2_norm'):
            l2_norm = model.embedding_l2_norm()
        else:
            # Fallback: regularize all parameters
            l2_norm = sum(p.pow(2).sum() for p in model.parameters() if p.requires_grad)

        # Multi-task combine (forced float32 inside)
        total_loss, log_dict = self.multi_task_loss(behavior_losses, l2_norm)

        # AMP backward
        if self.amp_enabled:
            self.scaler.scale(total_loss).backward()

            # Unscale before clipping
            self.scaler.unscale_(optimizer)

            # Gradient clipping — ESSENTIAL for AMP stability
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), self.max_grad_norm
            )
            log_dict["grad_norm"] = grad_norm.item()

            self.scaler.step(optimizer)
            self.scaler.update()

            log_dict["amp_scale"] = self.scaler.get_scale()
        else:
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), self.max_grad_norm
            )
            log_dict["grad_norm"] = grad_norm.item()
            optimizer.step()
            
        return log_dict