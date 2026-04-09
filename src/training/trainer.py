from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.model.bagnn import BAGNNModel
from src.core.contracts import BEHAVIOR_TYPES
from src.graph.neighbor_sampler import BehaviorAwareNeighborSampler
from src.training.losses import bpr_loss, MultiTaskBPRLoss, ContrastiveLearning
from src.core.contracts import EvalInput
from src.core.evaluator import TemporalSplitEvaluator

logger = logging.getLogger(__name__)

@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 512
    lr: float = 3e-4
    weight_decay: float = 1e-3
    l2_lambda: float = 1e-4
    num_neg: int = 1
    max_grad_norm: float = 1.0
    amp: bool = True
    patience: int = 5
    eval_every: int = 1
    eval_batch_size: int = 512
    num_workers: int = 4
    save_dir: str = "checkpoints/rees46"
    use_wandb: bool = False
    wandb_project: str = "bagnn-recsys"
    wandb_entity: str = "nguyenmaiductrong37"
    wandb_run_name: str = "bagnn-training"
    wandb_artifact_name: str = "bagnn-checkpoint"
    wandb_save_every: int = 5
    cl_weight: float = 0.1
    use_bf16: bool = True
    max_view_triplets: int = -1

    @classmethod
    def from_yaml(cls, cfg: dict) -> "TrainConfig":
        t = cfg.get("training", {})
        w = cfg.get("wandb", {})
        return cls(
            epochs=t.get("epochs", cls.epochs),
            batch_size=t.get("batch_size", cls.batch_size),
            lr=t.get("lr", cls.lr),
            weight_decay=t.get("weight_decay", cls.weight_decay),
            l2_lambda=t.get("l2_lambda", cls.l2_lambda),
            num_neg=t.get("num_neg", cls.num_neg),
            max_grad_norm=t.get("max_grad_norm", cls.max_grad_norm),
            amp=t.get("amp", cls.amp),
            patience=t.get("patience", cls.patience),
            eval_every=t.get("eval_every", cls.eval_every),
            eval_batch_size=t.get("eval_batch_size", cls.eval_batch_size),
            num_workers=t.get("num_workers", cls.num_workers),
            save_dir=t.get("save_dir", cls.save_dir),
            use_wandb=w.get("enabled", cls.use_wandb),
            wandb_project=w.get("project", cls.wandb_project),
            wandb_entity=w.get("entity", cls.wandb_entity),
            wandb_run_name=w.get("run_name", cls.wandb_run_name),
            wandb_artifact_name=w.get("artifact_name", cls.wandb_artifact_name),
            wandb_save_every=w.get("save_every", cls.wandb_save_every),
            cl_weight=t.get("cl_weight", cls.cl_weight),
            use_bf16=t.get("use_bf16", cls.use_bf16),
            max_view_triplets=t.get("max_view_triplets", cls.max_view_triplets),
        )

def load_yaml_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)

def _find_latest_checkpoint(save_dir: Path) -> Path | None:
    ckpts = sorted(save_dir.glob("epoch_*.pt"))
    return ckpts[-1] if ckpts else None

def _save_checkpoint(
    save_dir: Path,
    epoch: int,
    model: BAGNNModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    loss: float,
    metrics: dict,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "loss": loss,
            "metrics": metrics,
        },
        save_dir / f"epoch_{epoch:03d}.pt",
    )

def _load_checkpoint(
    ckpt_path: Path,
    model: BAGNNModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> int:
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    try:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    except (ValueError, KeyError, RuntimeError):
        logger.warning("Optimizer/scaler state incompatible with checkpoint — starting with fresh optimizer state.")
    resumed_epoch = int(ckpt["epoch"])
    logger.info("Resumed from %s (epoch %d, loss=%.4f)", ckpt_path, resumed_epoch, ckpt.get("loss", float("nan")))
    return resumed_epoch + 1

class InteractionDataset(Dataset):
    def __init__(self, triplets: torch.Tensor) -> None:
        assert triplets.ndim == 2 and triplets.size(1) == 3
        self.triplets = triplets

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.triplets[idx]

def train_epoch(
    model: BAGNNModel,
    sampler: BehaviorAwareNeighborSampler,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: MultiTaskBPRLoss,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    cl_fn: ContrastiveLearning,
    num_neg: int = 1,
    max_grad_norm: float = 1.0,
    amp: bool = True,
    cl_weight: float = 0.1,
    use_bf16: bool = True,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_cl_loss = 0.0
    n_steps = 0

    pbar = tqdm(dataloader, desc="train", leave=False, dynamic_ncols=True)
    for raw_batch in pbar:
        raw_batch = raw_batch.to(device)
        users_g = raw_batch[:, 0]
        items_g = raw_batch[:, 1]
        beh_ids = raw_batch[:, 2]

        unique_users = users_g.unique()
        subgraph = sampler.sample(unique_users, seed_type="user").to(device)

        optimizer.zero_grad(set_to_none=True)

        l_cl = torch.zeros(1, device=device).squeeze()

        _amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
        with torch.amp.autocast("cuda", dtype=_amp_dtype, enabled=amp and device.type == "cuda"):
            user_emb, item_emb, beh_embs = model(subgraph, return_beh_embs=True)

            user_x = subgraph["user"].x.contiguous()
            u_loc = torch.searchsorted(user_x, users_g.contiguous())

            prod_x = subgraph["product"].x
            sorted_px, sort_ord = prod_x.sort()
            pos_p = torch.searchsorted(sorted_px, items_g.contiguous()).clamp(max=sorted_px.size(0) - 1)
            found_p = sorted_px[pos_p] == items_g
            pp_loc = sort_ord[pos_p]

            if not found_p.any():
                continue

            u_loc = u_loc[found_p]
            pp_loc = pp_loc[found_p]
            bev = beh_ids[found_p]

            N_items = item_emb.size(0)
            behavior_losses: dict[str, torch.Tensor] = {}

            for beh_id, beh_name in enumerate(BEHAVIOR_TYPES):
                mask = bev == beh_id
                if not mask.any():
                    continue

                u_b = u_loc[mask]
                pp_b = pp_loc[mask]
                B_b = u_b.size(0)

                if N_items <= 1:
                    continue
                neg_loc = torch.randint(0, N_items - 1, (B_b, num_neg), device=device)
                neg_loc[neg_loc >= pp_b.unsqueeze(-1)] += 1

                u_emb_b = user_emb[u_b]
                pos_emb_b = item_emb[pp_b]
                neg_emb_b = item_emb[neg_loc]

                pos_s = (u_emb_b * pos_emb_b).sum(-1, keepdim=True)
                neg_s = torch.bmm(neg_emb_b, u_emb_b.unsqueeze(-1)).squeeze(-1)

                behavior_losses[beh_name] = bpr_loss(pos_s, neg_s)

            view_u = u_loc[bev == 0].unique()
            purch_u = u_loc[bev == 2].unique()
            if view_u.numel() >= 2 and purch_u.numel() >= 2:
                common = sorted(set(view_u.tolist()) & set(purch_u.tolist()))
                if len(common) >= 2:
                    ct = torch.tensor(common, device=device)
                    l_cl = cl_fn(beh_embs["view"][ct].float(), beh_embs["purchase"][ct].float())

        if not behavior_losses:
            continue

        l2 = model.embedding_l2_norm()
        loss, log = loss_fn(behavior_losses, l2)
        loss = loss + cl_weight * l_cl.float()

        if amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        total_loss += log["loss/total"]
        total_cl_loss += l_cl.item()
        n_steps += 1
        pbar.set_postfix(loss=f"{log['loss/total']:.4f}", cl=f"{l_cl.item():.4f}")

    return {
        "train/loss": total_loss / max(n_steps, 1),
        "train/cl_loss": total_cl_loss / max(n_steps, 1),
    }

@torch.no_grad()
def export_embeddings(
    model: BAGNNModel,
    sampler: BehaviorAwareNeighborSampler,
    user_ids: torch.Tensor,
    n_items: int,
    device: torch.device,
    batch_size: int = 512,
    use_bf16: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    d = model.embed_dim
    _amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    item_emb = torch.zeros(n_items, d)
    for start in range(0, n_items, batch_size):
        end = min(start + batch_size, n_items)
        seeds = torch.arange(start, end, device=device)
        sub = sampler.sample(seeds, seed_type="product").to(device)
        with torch.amp.autocast("cuda", dtype=_amp_dtype, enabled=device.type == "cuda"):
            _, item_local = model(sub)
        item_emb[start:end] = item_local.float().cpu()

    user_emb = torch.zeros(len(user_ids), d)
    for start in range(0, len(user_ids), batch_size):
        end = min(start + batch_size, len(user_ids))
        seeds = user_ids[start:end].to(device)
        sub = sampler.sample(seeds, seed_type="user").to(device)
        with torch.amp.autocast("cuda", dtype=_amp_dtype, enabled=device.type == "cuda"):
            u_local, _ = model(sub)
        user_emb[start:end] = u_local.float().cpu()

    return user_emb, item_emb

@torch.no_grad()
def eval_epoch(
    model: BAGNNModel,
    sampler: BehaviorAwareNeighborSampler,
    eval_user_ids: torch.Tensor,
    ground_truth: dict[int, int],
    exclude_items: dict[int, list[int]],
    n_items: int,
    evaluator: TemporalSplitEvaluator,
    device: torch.device,
    batch_size: int = 512,
    use_bf16: bool = True,
) -> dict[str, float]:
    valid_users = list(ground_truth.keys())
    eval_user_ids_filtered = torch.tensor(valid_users, dtype=torch.long, device=eval_user_ids.device)

    user_emb, item_emb = export_embeddings(
        model, sampler, eval_user_ids_filtered, n_items, device, batch_size, use_bf16=use_bf16
    )
    
    eval_input = EvalInput(
        user_embeddings=user_emb,
        item_embeddings=item_emb,
        eval_user_ids=eval_user_ids_filtered, 
        ground_truth=ground_truth,
        exclude_items=exclude_items,
    )
    
    return evaluator.evaluate(eval_input, batch_size=batch_size, mode="full")

def train(
    model: BAGNNModel,
    sampler: BehaviorAwareNeighborSampler,
    train_triplets: torch.Tensor,
    eval_user_ids: torch.Tensor,
    ground_truth: dict[int, int],
    exclude_items: dict[int, list[int]],
    n_items: int,
    behavior_counts: dict[str, int],
    cfg: TrainConfig,
    device: torch.device,
) -> None:
    model.to(device)

    dataset = InteractionDataset(train_triplets)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
    )

    loss_fn = MultiTaskBPRLoss(behavior_counts, l2_lambda=cfg.l2_lambda).to(device)
    cl_fn = ContrastiveLearning(temperature=0.1).to(device)
    emb_params = list(model.input_proj.parameters()) + list(model.beh_proj.parameters())
    emb_ids = {id(p) for p in emb_params}
    other_params = [p for p in model.parameters() if id(p) not in emb_ids]
    optimizer = torch.optim.Adam([
        {"params": other_params,  "weight_decay": cfg.weight_decay},
        {"params": emb_params,    "weight_decay": 0.0},
    ], lr=cfg.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and not cfg.use_bf16 and device.type == "cuda")
    evaluator = TemporalSplitEvaluator(ks=[10, 20, 50], device=str(device))

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    wandb_manager = None
    wandb_run = None
    if cfg.use_wandb:
        from src.training.checkpoint_manager import CheckpointManager
        import wandb

        wandb_manager = CheckpointManager(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            run_name=cfg.wandb_run_name,
            artifact_name=cfg.wandb_artifact_name,
            save_every_n_epochs=cfg.wandb_save_every,
            local_dir=str(save_dir),
        )
        wandb_run = wandb_manager.init_wandb(config={
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
            "l2_lambda": cfg.l2_lambda,
            "num_neg": cfg.num_neg,
            "amp": cfg.amp,
            "cl_weight": cfg.cl_weight,
        })
        logger.info("W&B enabled — project=%s run=%s", cfg.wandb_project, wandb_run.id)

    start_epoch = 0
    if wandb_manager is not None:
        start_epoch = wandb_manager.load_checkpoint(model, optimizer, scaler, device)

    if start_epoch == 0:
        latest_ckpt = _find_latest_checkpoint(save_dir)
        if latest_ckpt is not None:
            start_epoch = _load_checkpoint(latest_ckpt, model, optimizer, scaler, device)

    best_ndcg = -1.0
    no_improve = 0
    metrics = {}
    
    epoch_pbar = tqdm(range(start_epoch, cfg.epochs), desc="epochs", dynamic_ncols=True)
    for epoch in epoch_pbar:
        train_log = train_epoch(
            model, sampler, loader, optimizer, loss_fn, scaler, device,
            cl_fn=cl_fn,
            num_neg=cfg.num_neg, max_grad_norm=cfg.max_grad_norm, amp=cfg.amp,
            cl_weight=cfg.cl_weight, use_bf16=cfg.use_bf16,
        )
        train_loss = train_log["train/loss"]

        row = f"Epoch {epoch:03d} | " + " | ".join(f"{k}={v:.4f}" for k, v in train_log.items())

        postfix: dict[str, str] = {"loss": f"{train_loss:.4f}"}

        if (epoch + 1) % cfg.eval_every == 0:
            metrics = eval_epoch(
                model, sampler,
                eval_user_ids, ground_truth, exclude_items, n_items,
                evaluator, device, cfg.eval_batch_size, use_bf16=cfg.use_bf16,
            )
            row += " | " + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())

            ndcg20 = metrics.get("NDCG@20", 0.0)
            postfix["ndcg20"] = f"{ndcg20:.4f}"
            postfix["best"] = f"{max(best_ndcg, ndcg20):.4f}"

            if ndcg20 > best_ndcg:
                best_ndcg = ndcg20
                no_improve = 0
                torch.save(
                    {"epoch": epoch, "model_state_dict": model.state_dict(),
                     "optimizer_state_dict": optimizer.state_dict(),
                     "scaler_state_dict": scaler.state_dict(),
                     "metrics": metrics},
                    save_dir / "best.pt",
                )
                row += "  <- best"
            else:
                no_improve += 1

        epoch_pbar.set_postfix(postfix)
        _save_checkpoint(save_dir, epoch, model, optimizer, scaler, train_loss, metrics)

        if wandb_manager is not None:
            wandb_run.log({**train_log, **metrics, "epoch": epoch})
            cloud_ok = wandb_manager.save_checkpoint(
                model, optimizer, epoch,
                scaler=scaler,
                loss=train_loss,
                metrics=metrics,
            )
            if not cloud_ok:
                logger.error(
                    "Epoch %d: W&B checkpoint NOT verified. "
                    "Local file preserved. DO NOT close Colab yet.",
                    epoch,
                )

        logger.info(row)

        if no_improve >= cfg.patience:
            logger.info("Early stopping at epoch %d. Best NDCG@20=%.4f", epoch, best_ndcg)
            break

    if wandb_run is not None:
        wandb_run.finish()

    logger.info("Training complete. Best NDCG@20=%.4f", best_ndcg)

if __name__ == "__main__":
    import argparse
    import pickle
    import numpy as np
    import pandas as pd
    from torch_geometric.data import HeteroData
    from src.graph.neighbor_sampler import BehaviorAwareNeighborSampler, NeighborSamplerConfig

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/training.yaml")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg_dict = load_yaml_config(args.config)
    train_cfg = TrainConfig.from_yaml(cfg_dict)

    _device_str = args.device or cfg_dict.get("training", {}).get("device", "cuda")
    device = torch.device(_device_str if torch.cuda.is_available() or _device_str == "cpu" else "cpu")
    logger.info("Device: %s", device)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    _DATA = cfg_dict["data"]["data_dir"]
    _STRUCT = cfg_dict["data"]["struct_dir"]

    _nc_path = os.path.join(_DATA, "node_counts.json")
    if os.path.exists(_nc_path):
        with open(_nc_path) as _f:
            NODE_COUNTS: dict[str, int] = json.load(_f)
        logger.info("Node counts loaded from %s: %s", _nc_path, NODE_COUNTS)
    else:
        NODE_COUNTS = cfg_dict["data"]["node_counts"]
        logger.warning(
            "node_counts.json not found in %s; using YAML values %s. ",
            _DATA, NODE_COUNTS,
        )

    def load_data():
        def npy_ei(src_f, dst_f):
            src = np.load(f"{_DATA}/{src_f}")
            dst = np.load(f"{_DATA}/{dst_f}")
            return torch.from_numpy(np.stack([src, dst])).long()

        view_ei = npy_ei("view_train_src.npy", "view_train_dst.npy")
        cart_ei = npy_ei("cart_train_src.npy", "cart_train_dst.npy")
        purchase_ei = npy_ei("purchase_train_src.npy", "purchase_train_dst.npy")

        pb = pd.read_parquet(f"{_STRUCT}/product_brand.parquet")
        pc = pd.read_parquet(f"{_STRUCT}/product_category.parquet")
        brand_ei = torch.from_numpy(pb[["product_idx", "brand_idx"]].values.T.copy()).long()
        category_ei = torch.from_numpy(pc[["product_idx", "category_idx"]].values.T.copy()).long()

        hetero = HeteroData()
        for ntype, n in NODE_COUNTS.items():
            hetero[ntype].x = torch.arange(n)
            hetero[ntype].num_nodes = n

        hetero[("user", "view", "product")].edge_index     = view_ei.contiguous()
        hetero[("user", "cart", "product")].edge_index     = cart_ei.contiguous()
        hetero[("user", "purchase", "product")].edge_index = purchase_ei.contiguous()
        hetero[("product", "rev_view", "user")].edge_index     = view_ei.flip(0).contiguous()
        hetero[("product", "rev_cart", "user")].edge_index     = cart_ei.flip(0).contiguous()
        hetero[("product", "rev_purchase", "user")].edge_index = purchase_ei.flip(0).contiguous()
        hetero[("product", "belongs_to", "category")].edge_index = category_ei.contiguous()
        hetero[("category", "contains", "product")].edge_index   = category_ei.flip(0).contiguous()
        hetero[("product", "producedBy", "brand")].edge_index    = brand_ei.contiguous()
        hetero[("brand", "brands", "product")].edge_index        = brand_ei.flip(0).contiguous()

        n_v = view_ei.size(1)
        n_c = cart_ei.size(1)
        n_p = purchase_ei.size(1)

        view_train_ei = view_ei
        n_v_train = n_v
        if 0 < train_cfg.max_view_triplets < n_v:
            perm = torch.randperm(n_v)[:train_cfg.max_view_triplets]
            view_train_ei = view_ei[:, perm]
            n_v_train = train_cfg.max_view_triplets

        train_triplets = torch.cat([
            torch.stack([view_train_ei[0], view_train_ei[1], torch.full((n_v_train,), 0, dtype=torch.long)], dim=1),
            torch.stack([cart_ei[0],       cart_ei[1],       torch.full((n_c,),       1, dtype=torch.long)], dim=1),
            torch.stack([purchase_ei[0],   purchase_ei[1],   torch.full((n_p,),       2, dtype=torch.long)], dim=1),
        ], dim=0)

        val_users = torch.from_numpy(np.load(f"{_DATA}/val_user_idx.npy")).long()
        val_items = np.load(f"{_DATA}/val_product_idx.npy")
        ground_truth = {int(u): int(i) for u, i in zip(val_users.tolist(), val_items.tolist())}

        with open(f"{_DATA}/train_mask.pkl", "rb") as f:
            raw_mask = pickle.load(f)
        exclude_items = {int(k): list(int(x) for x in v) for k, v in raw_mask.items()}

        behavior_counts = {
            "view": n_v_train,
            "cart": int(cart_ei.size(1)),
            "purchase": n_p,
        }

        return hetero, train_triplets, val_users, ground_truth, exclude_items, behavior_counts

    hetero, train_triplets, eval_user_ids, ground_truth, exclude_items, behavior_counts = load_data()
    
    m_cfg = cfg_dict.get("model", {})
    s_cfg = cfg_dict.get("sampler", {})

    sampler = BehaviorAwareNeighborSampler(
        data=hetero,
        num_nodes_dict=NODE_COUNTS,
        config=NeighborSamplerConfig(
            hop1_budget=s_cfg.get("hop1_budget", 10),
            hop2_budget=s_cfg.get("hop2_budget", 5),
            hop1_sample_replace=s_cfg.get("hop1_sample_replace", False),
        ),
        device=device,
    )

    model = BAGNNModel(
        n_nodes=NODE_COUNTS,
        embed_dim=m_cfg.get("embed_dim", 64),
        n_layers=m_cfg.get("n_layers", 2),
        rank=m_cfg.get("rank", 16),
        dropout=m_cfg.get("dropout", 0.4),
        use_grad_checkpoint=m_cfg.get("use_grad_checkpoint", True),
    ).to(device)

    train(
        model=model,
        sampler=sampler,
        train_triplets=train_triplets,
        eval_user_ids=eval_user_ids,
        ground_truth=ground_truth,
        exclude_items=exclude_items,
        n_items=NODE_COUNTS["product"],
        behavior_counts=behavior_counts,
        cfg=train_cfg,
        device=device,
    )