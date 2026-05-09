#!/usr/bin/env python
"""
Evaluate a BPATMPModel checkpoint on val or test split.

  # Local checkpoint:
  python scripts/evaluate.py --checkpoint checkpoints-v2/best.pt

  # W&B artifact (alias: latest | epoch-025 | ...):
  python scripts/evaluate.py --wandb-artifact latest
  python scripts/evaluate.py --wandb-artifact epoch-025
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path

import torch
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_config(path: str = "config/training.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def download_artifact(alias: str, cfg: dict, dest: str = "/content/checkpoints") -> Path:
    import wandb
    w = cfg["wandb"]
    ref = f"{w['entity']}/{w['project']}/{w['artifact_name']}:{alias}"
    logger.info("Downloading %s", ref)
    dl_dir = Path(wandb.Api().artifact(ref, type="model").download(root=dest))
    pts = list(dl_dir.glob("*.pt")) + list(dl_dir.glob("*.pth"))
    if not pts:
        raise FileNotFoundError(f"No .pt file in {dl_dir}")
    return max(pts, key=lambda p: p.stat().st_mtime)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/training.yaml")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", help="Local .pt file")
    src.add_argument("--wandb-artifact", metavar="ALIAS", help="W&B artifact alias")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- checkpoint ---
    if args.wandb_artifact:
        ckpt_path = download_artifact(args.wandb_artifact, cfg)
    else:
        ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    logger.info("Loaded epoch=%s metrics=%s", ckpt.get("epoch"), ckpt.get("metrics"))

    # --- data ---
    # reuse build_hetero_data from run_training
    import sys; sys.path.insert(0, str(Path(__file__).parents[1]))
    from run_training import build_hetero_data
    hetero, node_counts, behavior_data = build_hetero_data(cfg)

    import numpy as np
    ref_time = float(np.concatenate([behavior_data[b]["ts"] for b in ["view", "cart", "purchase"]]).max())

    data_dir = Path(cfg["data"]["data_dir"])
    with open(data_dir / f"{args.split}_ground_truth.pkl", "rb") as f:
        ground_truth = pickle.load(f)

    mask_path = data_dir / "train_mask_purchase_only.pkl"
    if not mask_path.exists():
        mask_path = data_dir / "train_mask.pkl"
    with open(mask_path, "rb") as f:
        exclude_items = {u: list(v) for u, v in pickle.load(f).items()}

    # --- model ---
    from src.model.bpatmp import BPATMPModel
    from src.graph.neighbor_sampler import BehaviorAwareNeighborSampler, NeighborSamplerConfig

    mc = cfg["model"]
    model = BPATMPModel(
        num_nodes_dict=node_counts,
        embed_dim=mc["embed_dim"],
        n_layers=mc["n_layers"],
        dropout=mc["dropout"],
        n_intents=mc.get("n_intents", 32),
        rank=mc.get("rank", 32),
        use_grad_checkpoint=mc.get("use_grad_checkpoint", False),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    sc = cfg.get("sampler", {})
    sampler = BehaviorAwareNeighborSampler(
        data=hetero,
        config=NeighborSamplerConfig(
            hop1_budget=sc.get("hop1_budget", 10),
            hop2_budget=sc.get("hop2_budget", 5),
            hop1_sample_replace=sc.get("hop1_sample_replace", True),
        ),
        device=device,
    )

    # --- eval ---
    from src.core.evaluator import TemporalSplitEvaluator
    from src.training.trainer import eval_epoch

    ec, tc = cfg.get("evaluation", {}), cfg["training"]
    metrics = eval_epoch(
        model=model,
        sampler=sampler,
        eval_user_ids=torch.tensor(list(ground_truth.keys()), dtype=torch.long),
        ground_truth=ground_truth,
        exclude_items=exclude_items,
        n_items=node_counts["product"],
        evaluator=TemporalSplitEvaluator(ks=ec.get("ks", [10, 20, 50]), device=str(device)),
        device=device,
        batch_size=tc.get("eval_batch_size", 2048),
        use_bf16=tc.get("use_bf16", True),
        subsample=0,
        ref_time=ref_time,
    )

    print(f"\n{'='*45}\n  {args.split} | epoch {ckpt.get('epoch', '?')} | {ckpt_path.name}")
    for k, v in metrics.items():
        print(f"  {k:<12} {v:.4f}")
    print("="*45)

    out = ckpt_path.parent / f"eval_{args.split}.json"
    out.write_text(json.dumps({"split": args.split, "epoch": ckpt.get("epoch"), "metrics": metrics}, indent=2))
    logger.info("Saved → %s", out)


if __name__ == "__main__":
    main()
