from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.model.bpatmp import BPATMPModel
from src.core.contracts import BEHAVIOR_TYPES, EvalInput
from src.graph.neighbor_sampler import BehaviorAwareNeighborSampler
from src.training.losses import (
    BPATMPTotalLoss,
    bpr_loss,
    build_user_history_csr,
    sample_aligned_negatives_local,
)
from src.core.evaluator import TemporalSplitEvaluator

logger = logging.getLogger(__name__)


@torch.no_grad()
def diagnostic_module_stats(model: torch.nn.Module) -> dict[str, float]:
    """Per-epoch diagnostic: are temporal-decay (lambda, mu) and behavior-aware
    z_beta actually learning, or stuck near init?

    - softplus(raw_lambda)/softplus(raw_mu): per-behavior temporal decay rates
      Init at softplus(0) = ln(2) ≈ 0.693. If they drift apart per behavior,
      decay is being learned. If all stay near 0.693, decay is dead.
    - ||z_beta||: per-behavior weight scale in W_{rho,beta} decomposition.
      Init at ones (||z|| = sqrt(rank) for rank=64 → 8.0). Drift = learning.
    """
    stats: dict[str, float] = {}
    layer_idx = 0
    baw_idx = 0
    for _name, mod in model.named_modules():
        cls = mod.__class__.__name__
        if cls == "TemporalAttention" and hasattr(mod, "raw_lambda"):
            lam = F.softplus(mod.raw_lambda).detach().cpu()
            mu = F.softplus(mod.raw_mu).detach().cpu()
            for b_idx, b in enumerate(("view", "cart", "purchase", "struct")):
                if b_idx < lam.numel():
                    stats[f"diag/L{layer_idx}_lambda_{b}"] = float(lam[b_idx])
                    stats[f"diag/L{layer_idx}_mu_{b}"] = float(mu[b_idx])
            layer_idx += 1
        elif cls == "BehaviorAwareWeight" and hasattr(mod, "z_beta"):
            z = mod.z_beta.detach().cpu()
            for b_idx, b in enumerate(("view", "cart", "purchase", "struct")):
                if b_idx < z.shape[0]:
                    stats[f"diag/baw{baw_idx}_zbeta_norm_{b}"] = float(z[b_idx].norm())
            baw_idx += 1
    return stats


@dataclass
class TrainConfig:
    # Basic training
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

    # W&B
    use_wandb: bool = False
    wandb_project: str = "bpatmp-recsys"
    wandb_entity: str = "nguyenmaiductrong37"
    wandb_run_name: str = "bpatmp-training"
    wandb_artifact_name: str = "bpatmp-checkpoint"
    wandb_save_every: int = 5

    # Loss: L_total = L_BPR + lambda_cl*L_CL + lambda_conv*L_conv + lambda_mono*L_mono + lambda_wd*||theta||^2
    cl_weight: float = 0.1          # lambda_cl
    lambda_conv: float = 0.0        # lambda_conv (funnel prior; 0 = disabled)
    lambda_mono: float = 0.0        # lambda_mono (monotonic decay prior; 0 = disabled)
    funnel_margin: float = 0.1
    bpr_alpha: float = 0.5          # exponent in w_b = (N_p / N_b) ** alpha
    bpr_w_min: float = 0.05         # floor for w_b
    cl_every_k: int = 1             # 1 = every step, K>1 = run CL every K steps
    use_bf16: bool = True
    max_view_triplets: int = -1

    # Evaluation
    eval_subsample: int = 10000
    eval_seed: int = 42
    eval_ks: list[int] = field(default_factory=lambda: [10, 20, 50])
    primary_metric: str = "NDCG@20"

    # Hierarchical CL
    hierarchy_cl_enabled: bool = True
    hierarchy_cl_tau: float = 0.1
    hierarchy_cl_hard_k: int = 32
    hierarchy_cl_min_pair_overlap: int = 4
    hierarchy_cl_pair_weights: list[tuple[str, str, float]] | None = None

    # A100 Optimizations
    gradient_accumulation: int = 1
    warmup_epochs: int = 0
    min_lr: float = 1e-6
    use_fused_adamw: bool = True
    compile_model: bool = False
    allow_tf32: bool = True
    cudnn_benchmark: bool = True
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    log_every: int = 50
    empty_cache_freq: int = 0

    @classmethod
    def from_yaml(cls, cfg: dict) -> "TrainConfig":
        t = cfg.get("training", {})
        loss = cfg.get("loss", {})
        w = cfg.get("wandb", {})
        e = cfg.get("evaluation", {})
        hcl = cfg.get("hierarchy_cl", {})
        a100 = cfg.get("a100", {})

        eval_ks = e.get("ks", [10, 20, 50])
        primary_metric = str(e.get("primary_metric", cls.primary_metric))
        pair_weights = hcl.get("pair_weights", None)
        if pair_weights is not None:
            pair_weights = [(str(a), str(b), float(c)) for a, b, c in pair_weights]

        return cls(
            # Basic training
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

            # W&B
            use_wandb=w.get("enabled", cls.use_wandb),
            wandb_project=w.get("project", cls.wandb_project),
            wandb_entity=w.get("entity", cls.wandb_entity),
            wandb_run_name=w.get("run_name", cls.wandb_run_name),
            wandb_artifact_name=w.get("artifact_name", cls.wandb_artifact_name),
            wandb_save_every=w.get("save_every", cls.wandb_save_every),

            # Loss
            cl_weight=loss.get("lambda_cl", t.get("cl_weight", cls.cl_weight)),
            lambda_conv=loss.get("lambda_conv", cls.lambda_conv),
            lambda_mono=loss.get("lambda_mono", cls.lambda_mono),
            funnel_margin=loss.get("funnel_margin", cls.funnel_margin),
            bpr_alpha=loss.get("alpha", cls.bpr_alpha),
            bpr_w_min=loss.get("w_min", cls.bpr_w_min),
            cl_every_k=int(t.get("cl_every_k", a100.get("cl_every_k", cls.cl_every_k))),
            use_bf16=t.get("use_bf16", cls.use_bf16),
            max_view_triplets=t.get("max_view_triplets", cls.max_view_triplets),

            # Evaluation
            eval_subsample=t.get("eval_subsample", cls.eval_subsample),
            eval_seed=t.get("eval_seed", cls.eval_seed),
            eval_ks=list(eval_ks),
            primary_metric=primary_metric,

            # Hierarchical CL
            hierarchy_cl_enabled=bool(hcl.get("enabled", cls.hierarchy_cl_enabled)),
            hierarchy_cl_tau=float(hcl.get("tau", cls.hierarchy_cl_tau)),
            hierarchy_cl_hard_k=int(hcl.get("hard_k", cls.hierarchy_cl_hard_k)),
            hierarchy_cl_min_pair_overlap=int(
                hcl.get("min_pair_overlap", cls.hierarchy_cl_min_pair_overlap)
            ),
            hierarchy_cl_pair_weights=pair_weights,

            # A100 Optimizations
            gradient_accumulation=t.get("gradient_accumulation", cls.gradient_accumulation),
            warmup_epochs=t.get("warmup_epochs", cls.warmup_epochs),
            min_lr=t.get("min_lr", cls.min_lr),
            use_fused_adamw=a100.get("use_fused_adamw", t.get("optimizer", "") == "adamw_fused"),
            compile_model=a100.get("compile_model", cls.compile_model),
            allow_tf32=a100.get("allow_tf32", cls.allow_tf32),
            cudnn_benchmark=a100.get("cudnn_benchmark", cls.cudnn_benchmark),
            pin_memory=t.get("pin_memory", cls.pin_memory),
            persistent_workers=t.get("persistent_workers", cls.persistent_workers),
            prefetch_factor=t.get("prefetch_factor", cls.prefetch_factor),
            log_every=w.get("log_every", cls.log_every),
            empty_cache_freq=a100.get("empty_cache_freq", cls.empty_cache_freq),
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
    model: BPATMPModel,
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
    model: BPATMPModel,
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
        logger.warning(
            "Optimizer/scaler state incompatible with checkpoint — starting with fresh optimizer state."
        )
    resumed_epoch = int(ckpt["epoch"])
    logger.info(
        "Resumed from %s (epoch %d, loss=%.4f)",
        ckpt_path,
        resumed_epoch,
        ckpt.get("loss", float("nan")),
    )
    return resumed_epoch + 1


class InteractionDataset(Dataset):
    def __init__(self, triplets: torch.Tensor) -> None:
        assert triplets.ndim == 2 and triplets.size(1) in (3, 4)
        self.triplets = triplets

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.triplets[idx]


def _extract_per_behavior_lambdas(model: BPATMPModel) -> dict[str, torch.Tensor]:
    """Mean across encoder layers of softplus(raw_lambda) per behavior.

    Used by MonotonicDecayPriorLoss; returns a 1-element scalar tensor per
    behavior so the prior can compute relu(lam_strong - lam_weak) ** 2.
    """
    base = model._orig_mod if hasattr(model, "_orig_mod") else model  # torch.compile unwrap
    convs = base.encoder.convs
    stacked = torch.stack(
        [F.softplus(c.temporal_attn.raw_lambda) for c in convs], dim=0
    )  # (n_layers, 3)
    means = stacked.mean(dim=0)  # (3,)
    return {beh: means[i] for i, beh in enumerate(BEHAVIOR_TYPES)}


def train_epoch(
    model: BPATMPModel,
    sampler: BehaviorAwareNeighborSampler,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: BPATMPTotalLoss,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    num_neg: int = 1,
    max_grad_norm: float = 1.0,
    amp: bool = True,
    use_bf16: bool = True,
    cl_every_k: int = 1,
    history_ptr: torch.Tensor | None = None,
    history_item: torch.Tensor | None = None,
    pop_dist: torch.Tensor | None = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_cl_loss = 0.0
    n_steps = 0
    n_skipped = 0  # batches dropped because no positive items found in subgraph
    purchase_id = BEHAVIOR_TYPES.index("purchase")
    model_raw = model._orig_mod if hasattr(model, "_orig_mod") else model

    pbar = tqdm(dataloader, desc="train", leave=False, dynamic_ncols=True)
    for step, raw_batch in enumerate(pbar):
        raw_batch = raw_batch.to(device)
        users_g = raw_batch[:, 0]
        items_g = raw_batch[:, 1]
        beh_ids = raw_batch[:, 2]
        # ref_time = batch_min: every t_e < min(batch_ts) <= t_pos for ALL positives
        # in the batch, so no future leakage. Tradeoff: late-batch positives lose
        # access to recent context within (min, t_pos). Mitigated by large
        # batch_size + i.i.d. shuffling so distribution is balanced.
        ref_time = float(raw_batch[:, 3].min().item()) if raw_batch.size(1) >= 4 else None

        unique_users = users_g.unique()
        subgraph = sampler.sample(unique_users, seed_type="user").to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        _amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
        with torch.amp.autocast("cuda", dtype=_amp_dtype, enabled=amp and device.type == "cuda"):
            user_emb, item_emb, beh_embs = model(
                subgraph,
                return_beh_embs=True,
                ref_time=ref_time,
            )

            user_x = subgraph["user"].x.contiguous()
            u_loc = torch.searchsorted(user_x, users_g.contiguous())

            prod_x = subgraph["product"].x
            sorted_px, sort_ord = prod_x.sort()
            pos_p = torch.searchsorted(sorted_px, items_g.contiguous()).clamp(
                max=sorted_px.size(0) - 1
            )
            found_p = sorted_px[pos_p] == items_g
            pp_loc = sort_ord[pos_p]

            if not found_p.any():
                n_skipped += 1
                continue

            u_loc = u_loc[found_p]
            pp_loc = pp_loc[found_p]
            bev = beh_ids[found_p]
            users_g_kept = users_g[found_p]
            items_g_kept = items_g[found_p]

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

                u_emb_b = user_emb[u_b]
                pos_emb_b = item_emb[pp_b]

                if history_ptr is not None and history_item is not None and pop_dist is not None:
                    neg_loc = sample_aligned_negatives_local(
                        pp_b=pp_b,
                        user_b_global=users_g_kept[mask],
                        pos_b_global=items_g_kept[mask],
                        N_items=N_items,
                        num_neg=num_neg,
                        prod_x=subgraph["product"].x,
                        pop_dist_global=pop_dist,
                        history_ptr=history_ptr,
                        history_item=history_item,
                        user_emb_b=u_emb_b.detach(),
                        item_emb_local=item_emb.detach(),
                    )
                else:
                    neg_loc = torch.randint(0, N_items - 1, (B_b, num_neg), device=device)
                    neg_loc[neg_loc >= pp_b.unsqueeze(-1)] += 1

                neg_emb_b = item_emb[neg_loc]
                pos_bias = model_raw.item_bias(items_g_kept[mask])  # [B_b, 1]
                pos_s = (u_emb_b * pos_emb_b).sum(-1, keepdim=True) + pos_bias
                neg_global = prod_x[neg_loc.clamp(0, prod_x.size(0) - 1).long()]
                neg_bias = model_raw.item_bias(neg_global).squeeze(-1)  # [B_b, num_neg]
                neg_s = torch.bmm(neg_emb_b, u_emb_b.unsqueeze(-1)).squeeze(-1) + neg_bias
                behavior_losses[beh_name] = bpr_loss(pos_s, neg_s)

            users_per_beh = {b: u_loc[bev == bid].unique() for bid, b in enumerate(BEHAVIOR_TYPES)}

            # Funnel scores on the SHARED set of purchase positives so the
            # ordering s_view < s_cart < s_purchase is computed on matched (u,i).
            funnel_scores: dict[str, torch.Tensor] | None = None
            if loss_fn.lambda_conv > 0:
                p_mask = bev == purchase_id
                if p_mask.any():
                    u_p = u_loc[p_mask]
                    pp_p = pp_loc[p_mask]
                    pos_emb_p = item_emb[pp_p]
                    funnel_scores = {
                        b: (beh_embs[b][u_p] * pos_emb_p).sum(-1)
                        for b in BEHAVIOR_TYPES
                    }

            lambdas_dict: dict[str, torch.Tensor] | None = None
            if loss_fn.lambda_mono > 0:
                lambdas_dict = _extract_per_behavior_lambdas(model)

            # Skip CL on most steps when cl_every_k > 1: ~K times cheaper when lambda_cl > 0
            run_cl = loss_fn.lambda_cl > 0 and (cl_every_k <= 1 or (step % cl_every_k == 0))
            beh_embs_for_cl = (
                {b: beh_embs[b].float() for b in BEHAVIOR_TYPES} if run_cl else None
            )
            users_per_beh_for_cl = users_per_beh if run_cl else None

        if not behavior_losses:
            n_skipped += 1
            continue

        model_l2 = model.embedding_l2_norm()
        loss, log = loss_fn(
            behavior_losses=behavior_losses,
            beh_embs=beh_embs_for_cl,
            users_per_beh=users_per_beh_for_cl,
            scores=funnel_scores,
            lambdas=lambdas_dict,
            model_params=model_l2,
        )

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

        if scheduler is not None:
            scheduler.step()

        total_loss += log["loss/total"]
        total_cl_loss += float(log.get("loss/cl", 0.0))
        n_steps += 1
        pbar.set_postfix(
            loss=f"{log['loss/total']:.4f}",
            cl=f"{log.get('loss/cl', 0.0):.4f}",
        )

    if n_skipped > 0:
        logger.warning(
            "train_epoch: %d/%d batches skipped (no positive item in subgraph)",
            n_skipped,
            n_skipped + n_steps,
        )

    return {
        "train/loss": total_loss / max(n_steps, 1),
        "train/cl_loss": total_cl_loss / max(n_steps, 1),
        "train/skipped_batches": float(n_skipped),
    }


@torch.no_grad()
def export_embeddings(
    model: BPATMPModel,
    sampler: BehaviorAwareNeighborSampler,
    user_ids: torch.Tensor,
    n_items: int,
    device: torch.device,
    batch_size: int = 512,
    use_bf16: bool = True,
    ref_time: float | None = None,
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
            _, item_local = model(sub, ref_time=ref_time)
        item_emb[start:end] = item_local.float().cpu()

    user_emb = torch.zeros(len(user_ids), d)
    for start in range(0, len(user_ids), batch_size):
        end = min(start + batch_size, len(user_ids))
        seeds = user_ids[start:end].to(device)
        sub = sampler.sample(seeds, seed_type="user").to(device)
        with torch.amp.autocast("cuda", dtype=_amp_dtype, enabled=device.type == "cuda"):
            u_local, _ = model(sub, ref_time=ref_time)
        user_emb[start:end] = u_local.float().cpu()

    return user_emb, item_emb


@torch.no_grad()
def eval_epoch(
    model: BPATMPModel,
    sampler: BehaviorAwareNeighborSampler,
    eval_user_ids: torch.Tensor,
    ground_truth: dict[int, int | list[int]],
    exclude_items: dict[int, list[int]],
    n_items: int,
    evaluator: TemporalSplitEvaluator,
    device: torch.device,
    batch_size: int = 512,
    use_bf16: bool = True,
    subsample: int = 0,
    seed: int = 42,
    ref_time: float | None = None,
) -> dict[str, float]:
    valid_users = list(ground_truth.keys())

    if 0 < subsample < len(valid_users):
        gen = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(valid_users), generator=gen)[:subsample].tolist()
        valid_users = [valid_users[i] for i in idx]
        ground_truth = {u: ground_truth[u] for u in valid_users}
        exclude_items = {u: exclude_items.get(u, []) for u in valid_users}

    eval_user_ids_filtered = torch.tensor(
        valid_users, dtype=torch.long, device=eval_user_ids.device
    )

    user_emb, item_emb = export_embeddings(
        model,
        sampler,
        eval_user_ids_filtered,
        n_items,
        device,
        batch_size,
        use_bf16=use_bf16,
        ref_time=ref_time,
    )

    # L2 normalize for cosine-similarity ranking
    user_emb = F.normalize(user_emb, dim=-1)
    item_emb = F.normalize(item_emb, dim=-1)

    # Append item bias as extra dim: score = normalize(u)·normalize(i) + item_bias[i]
    model_raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    item_bias = model_raw.item_bias.weight.detach().cpu()  # [n_items, 1]
    user_emb = torch.cat([user_emb, torch.ones(user_emb.size(0), 1)], dim=-1)
    item_emb = torch.cat([item_emb, item_bias], dim=-1)

    eval_input = EvalInput(
        user_embeddings=user_emb,
        item_embeddings=item_emb,
        eval_user_ids=eval_user_ids_filtered,
        ground_truth=ground_truth,
        exclude_items=exclude_items,
    )

    return evaluator.evaluate(eval_input, batch_size=batch_size, mode="full_tiled")


def _setup_a100_optimizations(cfg: TrainConfig, device: torch.device) -> None:
    """Apply A100-specific optimizations."""
    if device.type != "cuda":
        return

    # TF32 for faster matmul on A100/H100
    if cfg.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("TF32 enabled for matmul operations")

    # cuDNN benchmark for faster convolutions
    if cfg.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
        logger.info("cuDNN benchmark enabled")

    # Log GPU info
    gpu_name = torch.cuda.get_device_name(device)
    gpu_mem = torch.cuda.get_device_properties(device).total_memory / 1e9
    logger.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")


def _get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr: float = 1e-6,
):
    """Cosine schedule with linear warmup."""
    import math

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(
    model: BPATMPModel,
    sampler: BehaviorAwareNeighborSampler,
    train_triplets: torch.Tensor,
    eval_user_ids: torch.Tensor,
    ground_truth: dict[int, int | list[int]],
    exclude_items: dict[int, list[int]],
    n_items: int,
    n_users: int,
    behavior_counts: dict[str, int],
    cfg: TrainConfig,
    device: torch.device,
    eval_ref_time: float | None = None,
) -> None:
    # Apply A100 optimizations
    _setup_a100_optimizations(cfg, device)

    model.to(device)

    # Compile model with torch.compile (PyTorch 2.0+)
    if cfg.compile_model and hasattr(torch, "compile"):
        # mode="default" + dynamic=True: hetero subgraph có shape thay đổi mỗi step,
        # "reduce-overhead" dùng CUDA graphs nên sẽ recompile liên tục → chậm hơn.
        logger.info("Compiling model with torch.compile (mode=default, dynamic=True)...")
        try:
            from torch import _dynamo as _td
            _td.config.cache_size_limit = 64
        except ImportError:
            pass
        model = torch.compile(model, mode="default", dynamic=True)

    dataset = InteractionDataset(train_triplets)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory and device.type == "cuda",
        drop_last=True,
        persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
    )

    loss_fn = BPATMPTotalLoss(
        behavior_counts=behavior_counts,
        lambda_cl=cfg.cl_weight if cfg.hierarchy_cl_enabled else 0.0,
        lambda_conv=cfg.lambda_conv,
        lambda_mono=cfg.lambda_mono,
        lambda_wd=cfg.l2_lambda,
        margin=cfg.funnel_margin,
        tau=cfg.hierarchy_cl_tau,
        alpha=cfg.bpr_alpha,
        w_min=cfg.bpr_w_min,
        cl_hard_k=cfg.hierarchy_cl_hard_k,
        cl_min_pair_overlap=cfg.hierarchy_cl_min_pair_overlap,
        cl_pair_weights=cfg.hierarchy_cl_pair_weights,
    ).to(device)
    logger.info(
        "Loss: BPR(alpha=%.2f, w_min=%.2f, weights=%s) + lambda_cl=%.3f + lambda_conv=%.3f + lambda_mono=%.3f + lambda_wd=%.1e",
        cfg.bpr_alpha, cfg.bpr_w_min,
        loss_fn.bpr.task_weights.tolist(),
        loss_fn.lambda_cl, loss_fn.lambda_conv, loss_fn.lambda_mono, loss_fn.lambda_wd,
    )

    history_ptr, history_item = build_user_history_csr(train_triplets, n_users=n_users)
    history_ptr = history_ptr.to(device)
    history_item = history_item.to(device)

    item_pop_counts = torch.bincount(train_triplets[:, 1].long(), minlength=n_items).float()
    pop_dist = (item_pop_counts + 1.0).pow(0.75)
    pop_dist = (pop_dist / pop_dist.sum()).to(device)
    emb_params = list(model.input_proj.parameters()) + list(model.beh_proj.parameters())
    emb_ids = {id(p) for p in emb_params}
    other_params = [p for p in model.parameters() if id(p) not in emb_ids]
    use_fused = cfg.use_fused_adamw and device.type == "cuda"
    optimizer = torch.optim.AdamW(
        [
            {"params": other_params, "weight_decay": cfg.weight_decay},
            {"params": emb_params, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
        fused=use_fused,
    )
    logger.info("Optimizer: AdamW (fused=%s, wd=%.1e on non-embedding)", use_fused, cfg.weight_decay)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=cfg.amp and not cfg.use_bf16 and device.type == "cuda"
    )

    steps_per_epoch = max(1, len(loader))
    num_training_steps = cfg.epochs * steps_per_epoch
    num_warmup_steps = max(0, cfg.warmup_epochs) * steps_per_epoch
    scheduler = _get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps, min_lr=cfg.min_lr / max(cfg.lr, 1e-12)
    )
    logger.info(
        "LR scheduler: cosine warmup — warmup_steps=%d, total_steps=%d, peak_lr=%.2e",
        num_warmup_steps,
        num_training_steps,
        cfg.lr,
    )
    evaluator = TemporalSplitEvaluator(ks=list(cfg.eval_ks), device=str(device))

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
        wandb_run = wandb_manager.init_wandb(
            config={
                "epochs": cfg.epochs,
                "batch_size": cfg.batch_size,
                "lr": cfg.lr,
                "weight_decay": cfg.weight_decay,
                "l2_lambda": cfg.l2_lambda,
                "num_neg": cfg.num_neg,
                "amp": cfg.amp,
                "cl_weight": cfg.cl_weight,
            }
        )
        logger.info("W&B enabled — project=%s run=%s", cfg.wandb_project, wandb_run.id)

    start_epoch = 0
    if wandb_manager is not None:
        start_epoch = wandb_manager.load_checkpoint(model, optimizer, scaler, device)

    if start_epoch == 0 and wandb_manager is None:
        latest_ckpt = _find_latest_checkpoint(save_dir)
        if latest_ckpt is not None:
            start_epoch = _load_checkpoint(latest_ckpt, model, optimizer, scaler, device)

    pm = cfg.primary_metric
    best_primary = -1.0
    no_improve = 0
    metrics = {}

    epoch_pbar = tqdm(range(start_epoch, cfg.epochs), desc="epochs", dynamic_ncols=True)
    for epoch in epoch_pbar:
        train_log = train_epoch(
            model,
            sampler,
            loader,
            optimizer,
            loss_fn,
            scaler,
            device,
            num_neg=cfg.num_neg,
            max_grad_norm=cfg.max_grad_norm,
            amp=cfg.amp,
            use_bf16=cfg.use_bf16,
            cl_every_k=cfg.cl_every_k,
            history_ptr=history_ptr,
            history_item=history_item,
            pop_dist=pop_dist,
            scheduler=scheduler,
        )
        train_loss = train_log["train/loss"]
        train_log["train/lr"] = float(optimizer.param_groups[0]["lr"])

        row = f"Epoch {epoch:03d} | " + " | ".join(f"{k}={v:.4f}" for k, v in train_log.items())

        postfix: dict[str, str] = {"loss": f"{train_loss:.4f}"}

        if (epoch + 1) % cfg.eval_every == 0:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            metrics = eval_epoch(
                model,
                sampler,
                eval_user_ids,
                ground_truth,
                exclude_items,
                n_items,
                evaluator,
                device,
                cfg.eval_batch_size,
                use_bf16=cfg.use_bf16,
                subsample=cfg.eval_subsample,
                seed=cfg.eval_seed,
                ref_time=eval_ref_time,
            )
            row += " | " + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())

            primary_val = metrics.get(pm, -1.0)
            postfix[pm.replace("@", "_")] = f"{primary_val:.4f}"
            postfix["best_primary"] = f"{max(best_primary, primary_val):.4f}"

            if primary_val > best_primary:
                best_primary = primary_val
                no_improve = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scaler_state_dict": scaler.state_dict(),
                        "metrics": metrics,
                    },
                    save_dir / "best.pt",
                )
                row += "  <- best"
            else:
                no_improve += 1

        epoch_pbar.set_postfix(postfix)
        _save_checkpoint(save_dir, epoch, model, optimizer, scaler, train_loss, metrics)

        diag = diagnostic_module_stats(model)
        if diag:
            lam_keys = sorted(k for k in diag if "_lambda_" in k)
            zb_keys = sorted(k for k in diag if "_zbeta_norm_" in k)
            lam_summary = " ".join(f"{k.split('/')[-1]}={diag[k]:.3f}" for k in lam_keys[:6])
            zb_summary = " ".join(f"{k.split('/')[-1]}={diag[k]:.3f}" for k in zb_keys[:6])
            row += f" | DIAG λ: {lam_summary} | DIAG z: {zb_summary}"

        if wandb_manager is not None:
            wandb_run.log({**train_log, **metrics, **diag, "epoch": epoch})
            cloud_ok = wandb_manager.save_checkpoint(
                model,
                optimizer,
                epoch,
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
            logger.info(
                "Early stopping at epoch %d. Best %s=%.4f",
                epoch,
                pm,
                best_primary,
            )
            break

    best_path = save_dir / "best.pt"
    if best_path.exists():
        logger.info("Loading best.pt for final FULL-rank evaluation on all val users...")
        best_ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt["model_state_dict"])
        final_val_metrics = eval_epoch(
            model,
            sampler,
            eval_user_ids,
            ground_truth,
            exclude_items,
            n_items,
            evaluator,
            device,
            cfg.eval_batch_size,
            use_bf16=cfg.use_bf16,
            subsample=0,
            ref_time=eval_ref_time,
        )
        logger.info(
            "FINAL VAL full-rank eval on best.pt: %s",
            " | ".join(f"{k}={v:.4f}" for k, v in final_val_metrics.items()),
        )
        with open(save_dir / "final_val_metrics.json", "w") as f:
            json.dump(final_val_metrics, f, indent=2)
        if wandb_run is not None:
            wandb_run.log(
                {f"final/val/{k}": v for k, v in final_val_metrics.items()}
            )

    if wandb_run is not None:
        wandb_run.finish()

    logger.info(
        "Training complete. Best %s (subsample)=%.4f. "
        "Run test eval: python scripts/evaluate.py --checkpoint %s --split test",
        pm,
        best_primary,
        save_dir / "best.pt",
    )

