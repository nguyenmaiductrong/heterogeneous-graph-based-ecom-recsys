import gc
import glob
import logging
import os
import time
from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.optim as optim

from src.core.contracts import EMBED_DIM

logger = logging.getLogger(__name__)

@dataclass
class LightGCNConfig:

    data_dir: str = ""
    checkpoint_dir: str = "checkpoints/lightgcn"
    embed_dim: int = EMBED_DIM
    num_layers: int = 3
    epochs: int = 1000
    batch_size: int = 8192
    lr: float = 1e-3
    reg_weight: float = 1e-4
    save_every: int = 10
    keep_top_k: int = 5
    device: str = "cuda"

class LightGCN(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        dim: int = EMBED_DIM,
        num_layers: int = 3,
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.dim = dim
        self.num_layers = num_layers

        # sparse=True is intentionally NOT used here: LightGCN accesses
        # ALL embeddings in every forward pass via full-graph sparse-mm,
        # so sparse gradients provide no memory benefit and would require
        # SparseAdam instead of Adam.
        self.user_emb = nn.Embedding(num_users, dim)
        self.item_emb = nn.Embedding(num_items, dim)
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)

    def forward(self, norm_adj: torch.sparse.FloatTensor):
        all_embs = torch.cat([self.user_emb.weight, self.item_emb.weight])
        
        alpha = 1.0 / (self.num_layers + 1)
        final_embs = all_embs * alpha  
        
        for _ in range(self.num_layers):
            all_embs = torch.sparse.mm(norm_adj, all_embs)
            final_embs = final_embs + (all_embs * alpha)
            
        user_final, item_final = torch.split(
            final_embs, [self.num_users, self.num_items],
        )
        return user_final, item_final


def build_normalized_adj(
    user_np: np.ndarray,
    item_np: np.ndarray,
    num_users: int,
    num_items: int,
) -> torch.sparse.FloatTensor:
    n = num_users + num_items

    row = np.concatenate([user_np, item_np + num_users]).astype(np.int64)
    col = np.concatenate([item_np + num_users, user_np]).astype(np.int64)
    data = np.ones(len(row), dtype=np.float32)

    A = sp.csr_matrix((data, (row, col)), shape=(n, n))
    A.eliminate_zeros()

    rowsum = np.array(A.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(rowsum + 1e-12, -0.5) 
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    D = sp.diags(d_inv_sqrt)

    norm_adj = (D @ A @ D).tocoo()

    del A, D, row, col, data, rowsum, d_inv_sqrt
    gc.collect()

    indices = torch.from_numpy(
        np.vstack((norm_adj.row, norm_adj.col)).astype(np.int64)
    )
    values = torch.from_numpy(norm_adj.data.copy().astype(np.float32))
    shape = torch.Size([n, n])

    del norm_adj
    gc.collect()

    return torch.sparse_coo_tensor(indices, values, shape)


def bpr_loss(
    users: torch.Tensor,
    pos_items: torch.Tensor,
    neg_items: torch.Tensor,
    user_emb: torch.Tensor,
    item_emb: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    u_e = user_emb[users]
    pos_e = item_emb[pos_items]
    neg_e = item_emb[neg_items]

    pos_scores = (u_e * pos_e).sum(dim=1)
    neg_scores = (u_e * neg_e).sum(dim=1)

    reg = 0.5 * (
        u_e.norm(2).pow(2)
        + pos_e.norm(2).pow(2)
        + neg_e.norm(2).pow(2)
    ) / float(len(users))
    bpr = -torch.nn.functional.logsigmoid(pos_scores - neg_scores).mean()
    return bpr, reg

def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    global_step: int,
    loss: float,
    metrics: dict[str, float] | None = None,
) -> None:
    """Save a full checkpoint (model + optimizer + training state)."""
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "metrics": metrics or {},
        },
        path,
    )
    logger.info("Checkpoint saved: %s (epoch %d, loss %.4f)", path, epoch, loss)


def find_latest_checkpoint(ckpt_dir: str) -> str | None:
    """Return the path of the most recent epoch checkpoint, or None."""
    if not os.path.isdir(ckpt_dir):
        return None
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "lightgcn_epoch_*.pth")))
    return ckpts[-1] if ckpts else None


def cleanup_old_checkpoints(ckpt_dir: str, keep: int = 5) -> None:
    """Keep only the *keep* most recent epoch checkpoints."""
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "lightgcn_epoch_*.pth")))
    for old in ckpts[:-keep]:
        os.remove(old)
        logger.info("Removed old checkpoint: %s", old)



def train(cfg: LightGCNConfig) -> None:
    device = torch.device(
        cfg.device
        if cfg.device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    logger.info("Device: %s", device.type.upper())

    # memory-mapped load — avoids doubling RAM usage for huge .npy files
    train_u = np.load(
        os.path.join(cfg.data_dir, "loo_purchase_train_src.npy"),
        mmap_mode="r",
    )
    train_i = np.load(
        os.path.join(cfg.data_dir, "loo_purchase_train_dst.npy"),
        mmap_mode="r",
    )

    num_users = int(train_u.max()) + 1
    num_items = int(train_i.max()) + 1
    num_edges = len(train_u)
    logger.info(
        "Edges: %s | Users: %s | Items: %s",
        f"{num_edges:,}", f"{num_users:,}", f"{num_items:,}",
    )

    # build normalised adjacency 
    norm_adj = build_normalized_adj(
        np.asarray(train_u), np.asarray(train_i), num_users, num_items,
    ).to(device)

    # model + optimizer 
    model = LightGCN(
        num_users, num_items, dim=cfg.embed_dim, num_layers=cfg.num_layers,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    # auto-resume from latest checkpoint
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    start_epoch = 1
    global_step = 0

    latest = find_latest_checkpoint(cfg.checkpoint_dir)
    if latest is not None:
        ckpt = torch.load(latest, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt.get("global_step", 0)
        logger.info(
            "Resumed from %s (epoch %d, loss %.4f)",
            latest, ckpt["epoch"], ckpt["loss"],
        )
        del ckpt
        gc.collect()

    # contiguous int64 copies for efficient shuffled indexing
    train_u_np = np.asarray(train_u, dtype=np.int64)
    train_i_np = np.asarray(train_i, dtype=np.int64)
    indices = np.arange(num_edges)


    for epoch in range(start_epoch, cfg.epochs + 1):
        model.train()
        np.random.shuffle(indices)

        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        # Tính toán Forward 1 lần cho cả đồ thị
        user_final_all, item_final_all = model(norm_adj)

        for i in range(0, num_edges, cfg.batch_size):
            batch_idx = indices[i : i + cfg.batch_size]

            b_users = torch.from_numpy(train_u_np[batch_idx]).to(device)
            b_pos = torch.from_numpy(train_i_np[batch_idx]).to(device)
            b_neg = torch.randint(0, num_items, (len(batch_idx),), device=device)

            optimizer.zero_grad()

            bpr, reg = bpr_loss(
                b_users, b_pos, b_neg, user_final_all, item_final_all,
            )
            
            loss = bpr + cfg.reg_weight * reg
            
            loss.backward(retain_graph=True if i + cfg.batch_size < num_edges else False)
            
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

        elapsed = time.time() - t0
        avg_loss = epoch_loss / max(n_batches, 1)
        logger.info(
            "Epoch [%d/%d] Loss: %.4f Time: %.1fs",
            epoch, cfg.epochs, avg_loss, elapsed,
        )

        if epoch % cfg.save_every == 0:
            ckpt_path = os.path.join(
                cfg.checkpoint_dir, f"lightgcn_epoch_{epoch:04d}.pth",
            )
            save_checkpoint(
                ckpt_path, model, optimizer, epoch, global_step, avg_loss,
            )
            cleanup_old_checkpoints(cfg.checkpoint_dir, cfg.keep_top_k)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    p = argparse.ArgumentParser(description="LightGCN purchase-only baseline")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints/lightgcn")
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=8192)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--reg_weight", type=float, default=1e-4)
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--keep_top_k", type=int, default=5)
    args = p.parse_args()

    train(
        LightGCNConfig(
            data_dir=args.data_dir,
            checkpoint_dir=args.checkpoint_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            reg_weight=args.reg_weight,
            save_every=args.save_every,
            keep_top_k=args.keep_top_k,
        ),
    )