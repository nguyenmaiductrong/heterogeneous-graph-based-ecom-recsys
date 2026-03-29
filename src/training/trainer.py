import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

from .losses import BPRTrainingStep

logger = logging.getLogger(__name__)


class Trainer:
    """
    config keys:
        epochs          (int)              : max training epochs
        patience        (int)              : early stopping patience
        save_dir        (str)              : directory to save checkpoints
        bpr_step        (BPRTrainingStep)  : training step handler
        amp             (bool, optional)   : enable AMP, default True
    """

    def __init__(self, model, optimizer, train_loader, val_loader, device, config):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config

        self.amp_enabled: bool = config.get("amp", True)
        self.bpr_step: BPRTrainingStep = config["bpr_step"]

        if self.amp_enabled and hasattr(self.bpr_step, "scaler"):
            self.scaler: GradScaler = self.bpr_step.scaler
        else:
            self.scaler = GradScaler("cuda", enabled=self.amp_enabled)

        # Early stopping 
        self.best_recall: float = -float("inf")
        self.best_epoch: int = 0
        self.patience_counter: int = 0

        # Checkpoint directory
        self.save_dir = Path(config["save_dir"])
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_recall": self.best_recall,
        }
        latest_path = self.save_dir / "checkpoint_latest.pt"
        torch.save(state, latest_path)
        logger.info(f"[Epoch {epoch}] Latest checkpoint -> {latest_path}")

        if is_best:
            best_path = self.save_dir / "checkpoint_best.pt"
            torch.save(state, best_path)
            logger.info(f"[Epoch {epoch}] Best checkpoint -> {best_path}")

    def load_checkpoint(self, checkpoint_path: str) -> int:
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        self.best_recall = ckpt.get("best_recall", -float("inf"))
        epoch = int(ckpt["epoch"])
        logger.info(f"Resumed from '{checkpoint_path}' at epoch {epoch}")
        return epoch

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in self.train_loader:
            graph_data = batch["graph"].to(self.device)
            interactions = {
                beh: {k: v.to(self.device) for k, v in beh_data.items()}
                for beh, beh_data in batch["interactions"].items()
            }

            with autocast("cuda", enabled=self.amp_enabled):
                user_emb, item_emb_all = self.model(graph_data)  # [N_u,d], [N_i,d]

            log_dict = self.bpr_step.step(
                model=self.model,
                optimizer=self.optimizer,
                batch=interactions,
                user_emb=user_emb,
                item_emb_all=item_emb_all,
            )

            total_loss += log_dict.get("loss/total", 0.0)
            num_batches += 1

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def validate(self) -> tuple[float, float, float]:
        self.model.eval()
        total_loss = 0.0
        hits_at_10 = 0
        hits_at_20 = 0
        num_users = 0

        for batch in self.val_loader:
            graph_data = batch["graph"].to(self.device)
            interactions = {
                beh: {k: v.to(self.device) for k, v in beh_data.items()}
                for beh, beh_data in batch["interactions"].items()
            }

            with autocast("cuda", enabled=self.amp_enabled):
                user_emb, item_emb_all = self.model(graph_data)  # [N_u,d], [N_i,d]

            if "purchase" not in interactions:
                continue

            users = interactions["purchase"]["user"]       # [B]
            pos_items = interactions["purchase"]["pos_item"]  # [B]

            u_emb = user_emb[users]          # [B, d]
            pos_emb = item_emb_all[pos_items]  # [B, d]

            neg_idx = torch.randint(
                0, item_emb_all.shape[0], (users.shape[0],), device=self.device
            )
            neg_emb = item_emb_all[neg_idx]   # [B, d]
            pos_scores = (u_emb * pos_emb).sum(dim=-1, keepdim=True)  # [B,1]
            neg_scores = (u_emb * neg_emb).sum(dim=-1, keepdim=True)  # [B,1]
            val_loss_batch = -F.logsigmoid(pos_scores - neg_scores).mean()
            total_loss += val_loss_batch.item()

            # Recall@K via full-catalogue ranking
            scores = u_emb @ item_emb_all.T                       # [B, N_i]
            pos_score_vals = scores.gather(1, pos_items.unsqueeze(1))  # [B, 1]
            # rank = number of items scored >= positive item score
            ranks = (scores >= pos_score_vals).sum(dim=-1)         # [B]

            hits_at_10 += (ranks <= 10).sum().item()
            hits_at_20 += (ranks <= 20).sum().item()
            num_users += users.shape[0]

        recall_at_10 = hits_at_10 / max(num_users, 1)
        recall_at_20 = hits_at_20 / max(num_users, 1)
        avg_loss = total_loss / max(len(self.val_loader), 1)
        return avg_loss, recall_at_10, recall_at_20

    def train(self, resume_from: str | None = None) -> None:
        start_epoch = 0
        if resume_from:
            start_epoch = self.load_checkpoint(resume_from)

        for epoch in range(start_epoch, self.config["epochs"]):
            train_loss = self.train_epoch()

            val_loss, recall_at_10, recall_at_20 = self.validate()

            logger.info(
                f"Epoch {epoch:04d} | train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"recall@10={recall_at_10:.4f} | recall@20={recall_at_20:.4f}"
            )

            monitor_metric = recall_at_10
            is_best = monitor_metric > self.best_recall

            if is_best:
                self.best_recall = monitor_metric
                self.best_epoch = epoch
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            self.save_checkpoint(epoch, is_best=is_best)

            if self.patience_counter >= self.config["patience"]:
                logger.info(
                    f"Early stopping triggered at epoch {epoch}. "
                    f"Best epoch: {self.best_epoch} | "
                    f"best val_recall@10 = {self.best_recall:.4f}"
                )
                break
            
        logger.info(
            f"Training complete. "
            f"Best val_recall@10 = {self.best_recall:.4f} at epoch {self.best_epoch}."
        )