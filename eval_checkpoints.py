import os
import json
import pickle
import numpy as np
import torch
import wandb
from pathlib import Path
from torch_geometric.data import HeteroData
import pandas as pd

from src.model.bagnn import BAGNNModel
from src.graph.neighbor_sampler import BehaviorAwareNeighborSampler, NeighborSamplerConfig
from src.core.evaluator import TemporalSplitEvaluator
from src.training.trainer import eval_epoch

ENTITY = "nguyenmaiductrong37-h-c-vi-n-c-ng-ngh-b-u-ch-nh-vi-n-th-ng"
PROJECT = "bagnn-recsys"
ARTIFACT_NAME = "bagnn-checkpoint-v2"
DATA_DIR = "data/processed/temporal"
STRUCT_DIR = "data/processed/temporal/node_mappings"
DOWNLOAD_DIR = "data/checkpoints_eval"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Path(DOWNLOAD_DIR).mkdir(parents=True, exist_ok=True)

with open(f"{DATA_DIR}/node_counts.json", "r") as f:
    NODE_COUNTS = json.load(f)

def build_hetero_graph():
    def load_ei(src_f, dst_f):
        src = np.load(f"{DATA_DIR}/{src_f}")
        dst = np.load(f"{DATA_DIR}/{dst_f}")
        return torch.from_numpy(np.stack([src, dst])).long()

    view_ei = load_ei("view_train_src.npy", "view_train_dst.npy")
    cart_ei = load_ei("cart_train_src.npy", "cart_train_dst.npy")
    purchase_ei = load_ei("purchase_train_src.npy", "purchase_train_dst.npy")

    pb = pd.read_parquet(f"{STRUCT_DIR}/product_brand.parquet")
    pc = pd.read_parquet(f"{STRUCT_DIR}/product_category.parquet")
    brand_ei = torch.from_numpy(pb[["product_idx", "brand_idx"]].values.T.copy()).long()
    category_ei = torch.from_numpy(pc[["product_idx", "category_idx"]].values.T.copy()).long()

    g = HeteroData()
    for ntype, n in NODE_COUNTS.items():
        g[ntype].x = torch.arange(n)
        g[ntype].num_nodes = n

    g[("user", "view", "product")].edge_index = view_ei.contiguous()
    g[("user", "cart", "product")].edge_index = cart_ei.contiguous()
    g[("user",    "purchase",      "product")].edge_index = purchase_ei.contiguous()
    g[("product", "rev_view",      "user")].edge_index    = view_ei.flip(0).contiguous()
    g[("product", "rev_cart",      "user")].edge_index    = cart_ei.flip(0).contiguous()
    g[("product", "rev_purchase",  "user")].edge_index    = purchase_ei.flip(0).contiguous()
    g[("product", "belongs_to",    "category")].edge_index = category_ei.contiguous()
    g[("category","contains",      "product")].edge_index  = category_ei.flip(0).contiguous()
    g[("product", "producedBy",    "brand")].edge_index    = brand_ei.contiguous()
    g[("brand",   "brands",        "product")].edge_index  = brand_ei.flip(0).contiguous()
    return g

hetero = build_hetero_graph()

sampler = BehaviorAwareNeighborSampler(
    data=hetero,
    num_nodes_dict=NODE_COUNTS,
    config=NeighborSamplerConfig(hop1_budget=15, hop2_budget=10),
    device=DEVICE,
)

model = BAGNNModel(
    n_nodes=NODE_COUNTS,
    embed_dim=64,
    n_layers=2,
    rank=16,
    dropout=0.4,
).to(DEVICE)

test_users = torch.from_numpy(np.load(f"{DATA_DIR}/test_user_idx.npy")).long()
test_items = np.load(f"{DATA_DIR}/test_product_idx.npy")
ground_truth = {int(u): int(i) for u, i in zip(test_users.tolist(), test_items.tolist())}

with open(f"{DATA_DIR}/train_mask.pkl", "rb") as f:
    raw_mask = pickle.load(f)
exclude_items = {int(k): list(int(x) for x in v) for k, v in raw_mask.items()}

val_users = np.load(f"{DATA_DIR}/val_user_idx.npy")
val_items = np.load(f"{DATA_DIR}/val_product_idx.npy")
for u, i in zip(val_users, val_items):
    u, i = int(u), int(i)
    if u not in exclude_items:
        exclude_items[u] = []
    exclude_items[u].append(i)

evaluator = TemporalSplitEvaluator(ks=[10, 20, 50], device=str(DEVICE))
api = wandb.Api()

for i in range(11):
    artifact_ref = f"{ENTITY}/{PROJECT}/{ARTIFACT_NAME}:v{i}"
    artifact = api.artifact(artifact_ref, type="model")
    
    # [QUAN TRỌNG] Tạo thư mục con riêng biệt (v0, v1...) để các file ko đè lên nhau
    v_dir = Path(DOWNLOAD_DIR) / f"v{i}"
    dl_dir = Path(artifact.download(root=str(v_dir)))

    pt_files = list(dl_dir.glob("*.pt"))
    ckpt_path = pt_files[0]

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True) # Thêm weights_only để tránh cảnh báo bảo mật
    model.load_state_dict(ckpt["model_state_dict"])

    metrics = eval_epoch(
        model=model,
        sampler=sampler,
        eval_user_ids=test_users,
        ground_truth=ground_truth,
        exclude_items=exclude_items,
        n_items=NODE_COUNTS["product"],
        evaluator=evaluator,
        device=DEVICE,
        batch_size=16384,
    )

    metrics_str = " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
    
    print(f"[{ckpt_path.name}]  {metrics_str}")