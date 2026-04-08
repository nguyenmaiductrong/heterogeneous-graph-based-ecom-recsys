"""
GraphSAGE-CF — Purchase-only bipartite GraphSAGE baseline
==========================================================

Graph   : User–Product purchase edges only (same input as LightGCN).
Aggreg. : Mean aggregator  (original GraphSAGE "mean" variant).

Per-layer update (both user side and item side):
    neigh_u = D_u^{-1} A_{ui} h_i          # row-normalised mean pooling
    h_u'    = ReLU( W_u · [h_u ‖ neigh_u] )  # self + neighbourhood → linear
    h_u'    = L2_norm(h_u')

    neigh_i = D_i^{-1} A_{iu} h_u
    h_i'    = ReLU( W_i · [h_i ‖ neigh_i] )
    h_i'    = L2_norm(h_i')

Scoring : dot( h_u^{(L)}, h_i^{(L)} )
Loss    : BPR  +  L2 regularisation on raw (layer-0) embeddings

This creates a clean 3-way ablation against the BAGNN paper:
  LightGCN (linear, no W)  →  GraphSAGE (non-linear, W, homogeneous)
  →  BAGNN (multi-behaviour, heterogeneous, behaviour-aware weight decomp.)
"""

from __future__ import annotations

import gc
import glob
import json
import logging
import os
import pickle
import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class GraphSAGEConfig:
    data_dir: str = ""
    checkpoint_dir: str = "checkpoints/graphsage"

    # Model
    embed_dim: int = 64        # must equal contracts.EMBED_DIM (64)
    num_layers: int = 2
    dropout: float = 0.1

    # Training
    epochs: int = 100
    batch_size: int = 8192
    lr: float = 1e-3
    reg_weight: float = 1e-4
    max_grad_norm: float = 5.0

    # Checkpointing
    save_every: int = 10
    keep_top_k: int = 5

    # Evaluation
    eval_every: int = 5
    eval_batch_size: int = 16384
    num_neg_eval: int = 999
    patience: int = 10

    device: str = "cuda"

    @classmethod
    def from_yaml(cls, raw: dict) -> "GraphSAGEConfig":
        m = raw.get("model", {})
        t = raw.get("training", {})
        d = raw.get("data", {})
        cfg = cls()
        cfg.data_dir       = d.get("data_dir",      cfg.data_dir)
        cfg.embed_dim      = m.get("embed_dim",      cfg.embed_dim)
        cfg.num_layers     = m.get("n_layers",       cfg.num_layers)
        cfg.dropout        = m.get("dropout",        cfg.dropout)
        cfg.epochs         = t.get("epochs",         cfg.epochs)
        cfg.batch_size     = t.get("batch_size",     cfg.batch_size)
        cfg.lr             = t.get("lr",             cfg.lr)
        cfg.reg_weight     = t.get("l2_lambda",      cfg.reg_weight)
        cfg.max_grad_norm  = t.get("max_grad_norm",  cfg.max_grad_norm)
        cfg.device         = t.get("device",         cfg.device)
        cfg.checkpoint_dir = t.get("save_dir",       cfg.checkpoint_dir)
        cfg.eval_every     = t.get("eval_every",     cfg.eval_every)
        cfg.save_every     = t.get("save_every",     cfg.save_every)
        cfg.keep_top_k     = t.get("keep_top_k",     cfg.keep_top_k)
        cfg.patience       = t.get("patience",       cfg.patience)
        return cfg


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class GraphSAGECF(nn.Module):
    """
    GraphSAGE with mean aggregator on a bipartite user–item purchase graph.

    Two *separate* weight matrices per layer (one for user side, one for item
    side) keep the parameter count symmetric and the ablation clean.

    Parameters
    ----------
    num_users, num_items : int
    dim : int
        Embedding dimension. Must equal ``contracts.EMBED_DIM`` (64) so the
        shared  ``TemporalSplitEvaluator`` can consume the embeddings.
    num_layers : int
    dropout : float
        Applied to each layer output during training only.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_users  = num_users
        self.num_items  = num_items
        self.dim        = dim
        self.num_layers = num_layers

        # Raw (layer-0) look-up tables
        self.user_emb = nn.Embedding(num_users, dim)
        self.item_emb = nn.Embedding(num_items, dim)
        nn.init.xavier_normal_(self.user_emb.weight)
        nn.init.xavier_normal_(self.item_emb.weight)

        # Per-layer linear projections: [h_self ‖ h_neigh] (2·dim) → dim
        self.W_user = nn.ModuleList(
            [nn.Linear(2 * dim, dim, bias=False) for _ in range(num_layers)]
        )
        self.W_item = nn.ModuleList(
            [nn.Linear(2 * dim, dim, bias=False) for _ in range(num_layers)]
        )
        for layer in (*self.W_user, *self.W_item):
            nn.init.xavier_normal_(layer.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        adj_u2i: torch.Tensor,  # sparse (num_users, num_items) row-normalised
        adj_i2u: torch.Tensor,  # sparse (num_items, num_users) row-normalised
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        h_user : (num_users, dim)
        h_item : (num_items, dim)
        """
        h_u = self.user_emb.weight   # (num_users, dim)
        h_i = self.item_emb.weight   # (num_items, dim)

        for l in range(self.num_layers):
            # ── Neighbourhood mean via sparse-dense matmul ──────────────────
            neigh_u = torch.sparse.mm(adj_u2i, h_i)   # (num_users, dim)
            neigh_i = torch.sparse.mm(adj_i2u, h_u)   # (num_items, dim)

            # ── Self + neighbour concat → W → ReLU → L2 normalise ───────────
            h_u_new = F.relu(self.W_user[l](torch.cat([h_u, neigh_u], dim=1)))
            h_u_new = F.normalize(h_u_new, p=2, dim=1)

            h_i_new = F.relu(self.W_item[l](torch.cat([h_i, neigh_i], dim=1)))
            h_i_new = F.normalize(h_i_new, p=2, dim=1)

            # ── Dropout during training ─────────────────────────────────────
            if self.training:
                h_u_new = self.dropout(h_u_new)
                h_i_new = self.dropout(h_i_new)

            h_u, h_i = h_u_new, h_i_new

        return h_u, h_i


# ---------------------------------------------------------------------------
# Sparse adjacency construction
# ---------------------------------------------------------------------------

def build_row_norm_adj(
    user_np: np.ndarray,
    item_np: np.ndarray,
    num_users: int,
    num_items: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build two row-normalised sparse COO tensors:
      adj_u2i[u, i] = 1 / deg(u)  if (u, i) ∈ E
      adj_i2u[i, u] = 1 / deg(i)  if (u, i) ∈ E

    Both are moved to *device* before return.
    """

    def _to_torch_sparse(
        row: np.ndarray,
        col: np.ndarray,
        n_rows: int,
        n_cols: int,
    ) -> torch.Tensor:
        A = sp.csr_matrix(
            (np.ones(len(row), dtype=np.float32), (row, col)),
            shape=(n_rows, n_cols),
        )
        A.eliminate_zeros()
        rowsum  = np.asarray(A.sum(axis=1)).flatten()
        d_inv   = np.where(rowsum > 0, 1.0 / rowsum, 0.0).astype(np.float32)
        D_inv   = sp.diags(d_inv)
        norm    = sp.csr_matrix(D_inv @ A).tocoo()

        indices = torch.from_numpy(
            np.vstack((norm.row, norm.col)).astype(np.int64)
        )
        values = torch.from_numpy(norm.data.copy())
        return torch.sparse_coo_tensor(
            indices, values, torch.Size([n_rows, n_cols])
        ).to(device)

    adj_u2i = _to_torch_sparse(user_np, item_np, num_users, num_items)
    adj_i2u = _to_torch_sparse(item_np, user_np, num_items, num_users)
    return adj_u2i, adj_i2u


# ---------------------------------------------------------------------------
# BPR loss  (regularisation on raw layer-0 embeddings)
# ---------------------------------------------------------------------------

def bpr_loss(
    users:      torch.Tensor,
    pos_items:  torch.Tensor,
    neg_items:  torch.Tensor,
    user_final: torch.Tensor,
    item_final: torch.Tensor,
    user_raw:   torch.Tensor,
    item_raw:   torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    u_e   = user_final[users]
    pos_e = item_final[pos_items]
    neg_e = item_final[neg_items]

    bpr = -F.logsigmoid(
        (u_e * pos_e).sum(dim=1) - (u_e * neg_e).sum(dim=1)
    ).mean()

    # L2 reg on raw embeddings (standard BPR-MF convention)
    u_raw   = user_raw[users]
    pos_raw = item_raw[pos_items]
    neg_raw = item_raw[neg_items]
    reg = 0.5 * (
        u_raw.norm(2).pow(2)
        + pos_raw.norm(2).pow(2)
        + neg_raw.norm(2).pow(2)
    ) / float(len(users))

    return bpr, reg


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    global_step: int,
    loss: float,
    metrics: dict[str, float] | None = None,
) -> None:
    torch.save(
        {
            "epoch":                epoch,
            "global_step":          global_step,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss":                 loss,
            "metrics":              metrics or {},
        },
        path,
    )
    logger.info("Checkpoint saved: %s (epoch %d, loss %.4f)", path, epoch, loss)


def find_latest_checkpoint(ckpt_dir: str) -> str | None:
    if not os.path.isdir(ckpt_dir):
        return None
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "graphsage_epoch_*.pth")))
    return ckpts[-1] if ckpts else None


def cleanup_old_checkpoints(ckpt_dir: str, keep: int = 5) -> None:
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "graphsage_epoch_*.pth")))
    for old in ckpts[:-keep]:
        os.remove(old)
        logger.info("Removed old checkpoint: %s", old)


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_eval(
    model:          GraphSAGECF,
    adj_u2i:        torch.Tensor,
    adj_i2u:        torch.Tensor,
    eval_user_ids:  torch.Tensor,
    ground_truth:   dict[int, int],
    exclude_items:  dict[int, list[int]],
    ks:             list[int],
    num_neg_eval:   int,
    eval_batch_size: int,
    device:         torch.device,
    seed:           int = 42,
) -> dict[str, float]:
    """
    Run ``TemporalSplitEvaluator.evaluate_sampled`` on *eval_user_ids*.

    ``eval_user_ids`` contains **global** user indices in [0, num_users).
    ``user_embeddings`` sent to the evaluator is ordered by position in
    ``eval_user_ids`` (row i = embedding for global user eval_user_ids[i]).
    """
    from src.core.contracts import EvalInput
    from src.core.evaluator import TemporalSplitEvaluator

    model.eval()
    user_all, item_all = model(adj_u2i, adj_i2u)

    eval_input = EvalInput(
        user_embeddings=user_all[eval_user_ids].cpu(),
        item_embeddings=item_all.cpu(),
        eval_user_ids=eval_user_ids.cpu(),
        ground_truth=ground_truth,
        exclude_items=exclude_items,
    )

    evaluator = TemporalSplitEvaluator(
        ks=ks,
        num_neg_samples=num_neg_eval,
        device=str(device),
    )
    return evaluator.evaluate_sampled(
        eval_input, batch_size=eval_batch_size, seed=seed
    )


# ---------------------------------------------------------------------------
# HuggingFace download helper (optional)
# ---------------------------------------------------------------------------

def download_data_from_hf(data_dir: str) -> None:
    """
    Download the REES46 processed dataset from HuggingFace Hub into
    *data_dir/temporal/*.

    Requires ``huggingface_hub`` (``pip install huggingface_hub``).
    GraphSAGE needs only the purchase split + evaluation files; the full
    dataset including view/cart edges is also available on HF but is not
    downloaded here.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required for --download_hf. "
            "Install with:  pip install huggingface_hub"
        ) from exc

    repo_id = "nguyenmaiductrong/rees46-processed-data"
    needed_files = [
        "temporal/purchase_train_src.npy",
        "temporal/purchase_train_dst.npy",
        "temporal/val_user_idx.npy",
        "temporal/val_product_idx.npy",
        "temporal/test_user_idx.npy",
        "temporal/test_product_idx.npy",
        "temporal/train_mask.pkl",
        "temporal/node_counts.json",
    ]

    os.makedirs(data_dir, exist_ok=True)
    for remote_path in needed_files:
        local_name = os.path.basename(remote_path)
        dest = os.path.join(data_dir, local_name)
        if os.path.exists(dest):
            logger.info("Already exists, skipping: %s", dest)
            continue
        logger.info("Downloading %s …", remote_path)
        tmp = hf_hub_download(
            repo_id=repo_id,
            filename=remote_path,
            repo_type="dataset",
        )
        import shutil
        shutil.copy(tmp, dest)
        logger.info("  → saved to %s", dest)

    logger.info("Download complete. Data dir: %s", data_dir)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(cfg: GraphSAGEConfig) -> None:
    device = torch.device(
        cfg.device
        if cfg.device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    logger.info("Device: %s", device.type.upper())

    # ── Load training edges ────────────────────────────────────────────────
    train_u = np.load(
        os.path.join(cfg.data_dir, "purchase_train_src.npy"), mmap_mode="r"
    )
    train_i = np.load(
        os.path.join(cfg.data_dir, "purchase_train_dst.npy"), mmap_mode="r"
    )

    # Infer vocabulary sizes from node_counts.json when available
    nc_path = os.path.join(cfg.data_dir, "node_counts.json")
    if os.path.exists(nc_path):
        with open(nc_path) as f:
            nc = json.load(f)
        num_users = int(nc["user"])
        num_items = int(nc["product"])
    else:
        num_users = int(train_u.max()) + 1
        num_items = int(train_i.max()) + 1

    num_edges = len(train_u)
    logger.info(
        "Edges: %s | Users: %s | Items: %s",
        f"{num_edges:,}", f"{num_users:,}", f"{num_items:,}",
    )

    # ── Build sparse adjacency matrices ───────────────────────────────────
    logger.info("Building row-normalised adjacency matrices …")
    adj_u2i, adj_i2u = build_row_norm_adj(
        np.asarray(train_u), np.asarray(train_i), num_users, num_items, device
    )

    # ── Load validation split ──────────────────────────────────────────────
    val_u_np = np.load(os.path.join(cfg.data_dir, "val_user_idx.npy"))
    val_i_np = np.load(os.path.join(cfg.data_dir, "val_product_idx.npy"))
    val_user_ids = torch.from_numpy(val_u_np).long()
    val_ground_truth: dict[int, int] = {
        int(u): int(i) for u, i in zip(val_u_np, val_i_np)
    }

    # Build exclude set: training positive purchases per user
    with open(os.path.join(cfg.data_dir, "train_mask.pkl"), "rb") as f:
        raw_mask = pickle.load(f)
    exclude_items: dict[int, list[int]] = {
        int(k): [int(x) for x in v] for k, v in raw_mask.items()
    }
    for u, i in zip(val_u_np, val_i_np):
        u, i = int(u), int(i)
        if u not in exclude_items:
            exclude_items[u] = []
        if i not in exclude_items[u]:
            exclude_items[u].append(i)

    # ── Model + optimizer ─────────────────────────────────────────────────
    model = GraphSAGECF(
        num_users=num_users,
        num_items=num_items,
        dim=cfg.embed_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %s", f"{total_params:,}")

    # ── Auto-resume from latest periodic checkpoint ────────────────────────
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    start_epoch  = 1
    global_step  = 0
    best_ndcg20  = -1.0
    patience_cnt = 0

    latest = find_latest_checkpoint(cfg.checkpoint_dir)
    if latest is not None:
        ckpt = torch.load(latest, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch  = ckpt["epoch"] + 1
        global_step  = ckpt.get("global_step", 0)
        best_ndcg20  = ckpt.get("metrics", {}).get("NDCG@20", -1.0)
        logger.info(
            "Resumed from %s (epoch %d, loss %.4f)",
            latest, ckpt["epoch"], ckpt["loss"],
        )
        del ckpt
        gc.collect()

    # ── Contiguous int64 copies for fast indexed access ────────────────────
    train_u_np = np.ascontiguousarray(train_u, dtype=np.int64)
    train_i_np = np.ascontiguousarray(train_i, dtype=np.int64)
    indices    = np.arange(num_edges)

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg.epochs + 1):
        model.train()
        np.random.shuffle(indices)

        epoch_loss = 0.0
        n_batches  = 0
        t0         = time.time()

        # Single full-graph forward pass for the epoch
        user_final, item_final = model(adj_u2i, adj_i2u)
        user_raw = model.user_emb.weight
        item_raw = model.item_emb.weight

        last_batch_start = (num_edges // cfg.batch_size) * cfg.batch_size

        for i in range(0, num_edges, cfg.batch_size):
            batch_idx = indices[i : i + cfg.batch_size]
            b_users   = torch.from_numpy(train_u_np[batch_idx]).to(device)
            b_pos     = torch.from_numpy(train_i_np[batch_idx]).to(device)
            b_neg     = torch.randint(
                0, num_items, (len(batch_idx),), device=device
            )

            optimizer.zero_grad()

            loss_bpr, loss_reg = bpr_loss(
                b_users, b_pos, b_neg,
                user_final, item_final,
                user_raw, item_raw,
            )
            loss = loss_bpr + cfg.reg_weight * loss_reg

            is_last = i >= last_batch_start
            loss.backward(retain_graph=not is_last)

            if cfg.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.max_grad_norm
                )

            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1
            global_step += 1

        # Release computation graph
        del user_final, item_final
        gc.collect()

        avg_loss = epoch_loss / max(n_batches, 1)
        elapsed  = time.time() - t0
        logger.info(
            "Epoch [%d/%d]  Loss: %.4f  Time: %.1fs",
            epoch, cfg.epochs, avg_loss, elapsed,
        )

        # ── Periodic evaluation ────────────────────────────────────────────
        metrics: dict[str, float] = {}
        if epoch % cfg.eval_every == 0:
            metrics = run_eval(
                model, adj_u2i, adj_i2u,
                val_user_ids, val_ground_truth, exclude_items,
                ks=[10, 20, 50],
                num_neg_eval=cfg.num_neg_eval,
                eval_batch_size=cfg.eval_batch_size,
                device=device,
            )
            ndcg20 = metrics.get("NDCG@20", 0.0)
            logger.info(
                "Val  %s",
                "  ".join(f"{k}={v:.4f}" for k, v in sorted(metrics.items())),
            )

            if ndcg20 > best_ndcg20:
                best_ndcg20  = ndcg20
                patience_cnt = 0
                best_path = os.path.join(
                    cfg.checkpoint_dir, "graphsage_best.pth"
                )
                save_checkpoint(
                    best_path, model, optimizer,
                    epoch, global_step, avg_loss, metrics,
                )
                logger.info(
                    "New best NDCG@20=%.4f saved → %s", best_ndcg20, best_path
                )
            else:
                patience_cnt += 1
                logger.info(
                    "No improvement (%d/%d patience)", patience_cnt, cfg.patience
                )
                if patience_cnt >= cfg.patience:
                    logger.info(
                        "Early stopping triggered at epoch %d.", epoch
                    )
                    break

        # ── Periodic checkpoint ────────────────────────────────────────────
        if epoch % cfg.save_every == 0:
            ckpt_path = os.path.join(
                cfg.checkpoint_dir, f"graphsage_epoch_{epoch:04d}.pth"
            )
            save_checkpoint(
                ckpt_path, model, optimizer,
                epoch, global_step, avg_loss, metrics,
            )
            cleanup_old_checkpoints(cfg.checkpoint_dir, cfg.keep_top_k)

    logger.info("Training complete. Best Val NDCG@20=%.4f", best_ndcg20)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import yaml

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    p = argparse.ArgumentParser(
        description="GraphSAGE-CF purchase-only baseline"
    )
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to graphsage_baseline.yaml (CLI flags override YAML values)",
    )
    p.add_argument("--data_dir",       type=str,   default=None)
    p.add_argument("--checkpoint_dir", type=str,   default=None)
    p.add_argument("--embed_dim",      type=int,   default=None)
    p.add_argument("--num_layers",     type=int,   default=None)
    p.add_argument("--dropout",        type=float, default=None)
    p.add_argument("--epochs",         type=int,   default=None)
    p.add_argument("--batch_size",     type=int,   default=None)
    p.add_argument("--lr",             type=float, default=None)
    p.add_argument("--reg_weight",     type=float, default=None)
    p.add_argument("--device",         type=str,   default=None)
    p.add_argument("--eval_every",     type=int,   default=None)
    p.add_argument("--save_every",     type=int,   default=None)
    p.add_argument("--patience",       type=int,   default=None)
    p.add_argument(
        "--download_hf",
        action="store_true",
        help=(
            "Download required files from HuggingFace "
            "(nguyenmaiductrong/rees46-processed-data) into --data_dir "
            "before training.  Requires huggingface_hub."
        ),
    )
    args = p.parse_args()

    # Start from YAML defaults, then apply CLI overrides
    cfg = GraphSAGEConfig()
    if args.config:
        with open(args.config) as f:
            raw_yaml = yaml.safe_load(f)
        cfg = GraphSAGEConfig.from_yaml(raw_yaml)

    # CLI overrides (only when explicitly provided)
    _cli_map = {
        "data_dir":       "data_dir",
        "checkpoint_dir": "checkpoint_dir",
        "embed_dim":      "embed_dim",
        "num_layers":     "num_layers",
        "dropout":        "dropout",
        "epochs":         "epochs",
        "batch_size":     "batch_size",
        "lr":             "lr",
        "reg_weight":     "reg_weight",
        "device":         "device",
        "eval_every":     "eval_every",
        "save_every":     "save_every",
        "patience":       "patience",
    }
    for cli_attr, cfg_attr in _cli_map.items():
        val = getattr(args, cli_attr, None)
        if val is not None:
            setattr(cfg, cfg_attr, val)

    if not cfg.data_dir:
        p.error("--data_dir is required (or set data.data_dir in the config YAML)")

    if args.download_hf:
        download_data_from_hf(cfg.data_dir)

    train(cfg)
