#!/usr/bin/env python
"""
Main training script - handles data loading and training from config file.
Usage: python run_training.py --config config/training.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch_geometric.data import HeteroData

from src.core.contracts import BEHAVIOR_TYPES
from src.graph.neighbor_sampler import BehaviorAwareNeighborSampler
from src.model.bpatmp import BPATMPModel
from src.training.trainer import TrainConfig, train

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_hetero_data(cfg: dict) -> tuple[HeteroData, dict]:
    """Build HeteroData from config paths."""
    data_dir = Path(cfg["data"]["data_dir"])
    struct_dir = Path(cfg["data"]["struct_dir"])

    node_counts_path = data_dir / "node_counts.json"
    if node_counts_path.exists():
        with open(node_counts_path) as f:
            node_counts = json.load(f)
        logger.info(f"Loaded node_counts from {node_counts_path}")
    else:
        node_counts = cfg["data"]["node_counts"]
        logger.info("Using node_counts from config")

    n_users = node_counts["user"]
    n_items = node_counts["product"]
    n_categories = node_counts["category"]
    n_brands = node_counts["brand"]

    logger.info(f"Users: {n_users:,}, Items: {n_items:,}, Categories: {n_categories}, Brands: {n_brands:,}")

    def load_npy(prefix, suffix):
        return np.load(data_dir / f"{prefix}_{suffix}.npy")

    behavior_data = {}
    for beh in ["view", "cart", "purchase"]:
        src = load_npy(f"{beh}_train", "src")
        dst = load_npy(f"{beh}_train", "dst")
        ts = load_npy(f"{beh}_train", "ts")
        behavior_data[beh] = {"src": src, "dst": dst, "ts": ts}
        logger.info(f"{beh}: {len(src):,} edges")

    prod_cat = pd.read_parquet(struct_dir / "product_category.parquet")
    prod_brand = pd.read_parquet(struct_dir / "product_brand.parquet")
    logger.info(f"Product-Category edges: {len(prod_cat):,}")
    logger.info(f"Product-Brand edges: {len(prod_brand):,}")

    hetero = HeteroData()

    hetero["user"].x = torch.arange(n_users, dtype=torch.long)
    hetero["user"].num_nodes = n_users
    hetero["product"].x = torch.arange(n_items, dtype=torch.long)
    hetero["product"].num_nodes = n_items
    hetero["category"].x = torch.arange(n_categories, dtype=torch.long)
    hetero["category"].num_nodes = n_categories
    hetero["brand"].x = torch.arange(n_brands, dtype=torch.long)
    hetero["brand"].num_nodes = n_brands

    for beh in ["view", "cart", "purchase"]:
        src = torch.from_numpy(behavior_data[beh]["src"]).long()
        dst = torch.from_numpy(behavior_data[beh]["dst"]).long()
        ts = torch.from_numpy(behavior_data[beh]["ts"]).long()

        hetero[("user", beh, "product")].edge_index = torch.stack([src, dst])
        hetero[("user", beh, "product")].edge_time = ts
        hetero[("product", f"rev_{beh}", "user")].edge_index = torch.stack([dst, src])
        hetero[("product", f"rev_{beh}", "user")].edge_time = ts

    cat_ei = torch.from_numpy(prod_cat[["product_idx", "category_idx"]].values.T.copy()).long()
    brand_ei = torch.from_numpy(prod_brand[["product_idx", "brand_idx"]].values.T.copy()).long()

    hetero[("product", "belongs_to", "category")].edge_index = cat_ei
    hetero[("category", "contains", "product")].edge_index = cat_ei.flip(0)
    hetero[("product", "producedBy", "brand")].edge_index = brand_ei
    hetero[("brand", "brands", "product")].edge_index = brand_ei.flip(0)

    logger.info(f"HeteroData built: {hetero}")
    return hetero, node_counts, behavior_data


def build_train_triplets(behavior_data: dict, max_view: int = -1) -> torch.Tensor:
    """Build train triplets from behavior data."""
    triplets_list = []
    for beh_id, beh in enumerate(BEHAVIOR_TYPES):
        src = behavior_data[beh]["src"]
        dst = behavior_data[beh]["dst"]
        ts = behavior_data[beh]["ts"]
        beh_arr = np.full(len(src), beh_id, dtype=np.int64)
        triplets_list.append(np.stack([src, dst, beh_arr, ts], axis=1))

    train_triplets = torch.from_numpy(np.concatenate(triplets_list, axis=0)).long()

    if max_view > 0:
        view_mask = train_triplets[:, 2] == BEHAVIOR_TYPES.index("view")
        non_view = train_triplets[~view_mask]
        view_only = train_triplets[view_mask]
        if len(view_only) > max_view:
            perm = torch.randperm(len(view_only))[:max_view]
            view_only = view_only[perm]
        train_triplets = torch.cat([non_view, view_only])

    logger.info(f"Train triplets: {len(train_triplets):,}")
    return train_triplets


def main():
    parser = argparse.ArgumentParser(description="Train BPATMP model")
    parser.add_argument("--config", type=str, default="config/training.yaml", help="Path to config file")
    args = parser.parse_args()

    logger.info(f"Loading config from {args.config}")
    cfg = load_config(args.config)

    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    hetero, node_counts, behavior_data = build_hetero_data(cfg)

    data_dir = Path(cfg["data"]["data_dir"])
    with open(data_dir / "val_ground_truth.pkl", "rb") as f:
        val_gt = pickle.load(f)
    with open(data_dir / "train_mask.pkl", "rb") as f:
        train_mask = pickle.load(f)

    exclude_items = {u: list(items) for u, items in train_mask.items()}
    eval_user_ids = torch.tensor(list(val_gt.keys()), dtype=torch.long)
    logger.info(f"Eval users: {len(eval_user_ids):,}")

    max_view = cfg["training"].get("max_view_triplets", -1)
    train_triplets = build_train_triplets(behavior_data, max_view)

    behavior_counts = {beh: int((train_triplets[:, 2] == i).sum()) for i, beh in enumerate(BEHAVIOR_TYPES)}
    logger.info(f"Behavior counts: {behavior_counts}")

    device = torch.device(cfg["training"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Using device: {device}")

    model_cfg = cfg["model"]
    model = BPATMPModel(
        num_nodes_dict=node_counts,
        embed_dim=model_cfg["embed_dim"],
        n_layers=model_cfg["n_layers"],
        dropout=model_cfg["dropout"],
        n_intents=model_cfg.get("n_intents", 32),
        rank=model_cfg.get("rank", 32),
        use_grad_checkpoint=model_cfg.get("use_grad_checkpoint", False),
    )
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    sampler_cfg = cfg.get("sampler", {})
    from src.graph.neighbor_sampler import NeighborSamplerConfig
    sampler_config = NeighborSamplerConfig(
        hop1_budget=sampler_cfg.get("hop1_budget", 10),
        hop2_budget=sampler_cfg.get("hop2_budget", 5),
        hop1_sample_replace=sampler_cfg.get("hop1_sample_replace", True),
    )
    sampler = BehaviorAwareNeighborSampler(
        data=hetero,
        config=sampler_config,
    )

    train_cfg = TrainConfig.from_yaml(cfg)
    eval_ref_time = float(train_triplets[:, 3].max().item())
    logger.info(f"Eval reference time: {eval_ref_time}")

    train(
        model=model,
        sampler=sampler,
        train_triplets=train_triplets,
        eval_user_ids=eval_user_ids,
        ground_truth=val_gt,
        exclude_items=exclude_items,
        n_items=node_counts["product"],
        n_users=node_counts["user"],
        behavior_counts=behavior_counts,
        cfg=train_cfg,
        device=device,
        eval_ref_time=eval_ref_time,
    )


if __name__ == "__main__":
    main()
