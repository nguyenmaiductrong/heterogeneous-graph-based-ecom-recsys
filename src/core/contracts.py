from __future__ import annotations
 
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple
 
import torch
from torch import Tensor

EMBED_DIM: int = 128
NUM_HEADS: int = 4
HEAD_DIM: int = EMBED_DIM // NUM_HEADS  # 32
NUM_GNN_LAYERS: int = 2
LOW_RANK: int = 16  # r for A_phi @ B_beta^T
SVD_RANK: int = 256 # q for randomized SVD
 
NODE_TYPES: List[str] = ["user", "product", "category", "brand"]
BEHAVIOR_TYPES: List[str] = ["view", "cart", "purchase"]
TARGET_BEHAVIOR: str = "purchase"

RELATION_TYPES: List[Tuple[str, str, str]] = [
    ("user",     "view",       "product"),
    ("user",     "cart",       "product"),
    ("user",     "purchase",   "product"),
    ("product",  "rev_view",   "user"),
    ("product",  "rev_cart",   "user"),
    ("product",  "rev_purchase","user"),
    ("product",  "belongs_to", "category"),
    ("category", "contains",   "product"),
    ("product",  "producedBy",  "brand"),
    ("brand",    "brands",     "product"),
]

BEHAVIOR_LOSS_WEIGHTS: Dict[str, float] = {
    "purchase": 1.0,
    "cart":     0.3,
    "view":     0.1,
}

BEHAVIOR_TO_ID: Dict[str, int] = {"view": 0, "cart": 1, "purchase": 2}
ID_TO_BEHAVIOR: Dict[int, str] = {v: k for k, v in BEHAVIOR_TO_ID.items()}

@dataclass
class SampledSubgraph:
    # {node_type: Tensor(num_nodes_of_type, EMBED_DIM)}
    node_features: Dict[str, Tensor]
 
    # {(src_type, edge_type, dst_type): Tensor(2, num_edges)}
    edge_index: Dict[Tuple[str, str, str], Tensor]
 
    # {(src_type, edge_type, dst_type): Tensor(num_edges,)}
    # values in {0=view, 1=cart, 2=purchase}
    edge_behavior_origin: Dict[Tuple[str, str, str], Tensor]
 
    # Indices into node_features["user"] for this batch
    # shape: (batch_size,)
    target_user_indices: Tensor
 
    node_id_map: Dict[str, Tensor]
 
    def validate(self) -> None:
        for ntype, feat in self.node_features.items():
            assert ntype in NODE_TYPES, f"Unknown node type: {ntype}"
            assert feat.dim() == 2 and feat.size(1) == EMBED_DIM, (
                f"node_features[{ntype}] shape {feat.shape}, "
                f"expected (*, {EMBED_DIM})"
            )
        for rel, idx in self.edge_index.items():
            assert idx.dim() == 2 and idx.size(0) == 2, (
                f"edge_index[{rel}] must be (2, E), got {idx.shape}"
            )
            if rel in self.edge_behavior_origin:
                E = idx.size(1)
                orig = self.edge_behavior_origin[rel]
                assert orig.shape == (E,), (
                    f"origin shape {orig.shape} != ({E},)"
                )

@dataclass
class GNNOutput:
    # BEFORE hierarchy gating — per-behavior
    # {behavior: {"user": (B, d), "product": (B, d)}}
    per_behavior_emb: Dict[str, Dict[str, Tensor]]
 
    # AFTER multi-order concat + projection
    final_user_emb: Tensor   # (num_users_in_batch, d)
    final_item_emb: Tensor   # (num_items_in_batch, d)
 
    def validate(self) -> None:
        for beh in BEHAVIOR_TYPES:
            assert beh in self.per_behavior_emb
            for nt in ["user", "product"]:
                e = self.per_behavior_emb[beh][nt]
                assert e.dim() == 2 and e.size(1) == EMBED_DIM
        assert self.final_user_emb.size(1) == EMBED_DIM
        assert self.final_item_emb.size(1) == EMBED_DIM

@dataclass
class SVDFactors:
    """
    Producer: P1 (preprocessing, runs once)
    Consumer: P3 (ContrastiveLearning — SVD-augmented view)
 
    For behavior k:
      US_k = U_k @ S_k   shape: (num_users, SVD_RANK)
      VS_k = V_k @ S_k   shape: (num_items, SVD_RANK)
 
    SVD-view embedding (never materialize full I×J matrix):
      g_user = US_k @ (VS_k^T @ E_item)   # O(q*d)
    """
    US: Dict[str, Tensor]   # {behavior: (num_users, q)}
    VS: Dict[str, Tensor]   # {behavior: (num_items, q)}
 
    def validate(self) -> None:
        for beh in BEHAVIOR_TYPES:
            assert beh in self.US and beh in self.VS
            assert self.US[beh].size(1) == SVD_RANK
            assert self.VS[beh].size(1) == SVD_RANK

@dataclass
class GatedOutput:
    gated_user_emb: Tensor   # (batch_users, d)
    gated_item_emb: Tensor   # (batch_items, d)
    cl_loss: Tensor          # scalar
 
    def validate(self) -> None:
        assert self.gated_user_emb.size(1) == EMBED_DIM
        assert self.gated_item_emb.size(1) == EMBED_DIM
        assert self.cl_loss.dim() == 0

@dataclass
class LossInput:
    """Consolidated input for multi-task BPR + CL loss.
 
    Assembled by: P4 from GatedOutput + neg sampling
    """
    user_emb: Tensor        # (B, d)
    pos_item_emb: Tensor    # (B, d)
    neg_item_emb: Tensor    # (B, num_neg, d)
    behavior_ids: Tensor    # (B,) in {0,1,2}
    cl_loss: Tensor         # scalar from P3
 
    def validate(self) -> None:
        B = self.user_emb.size(0)
        assert self.user_emb.shape == (B, EMBED_DIM)
        assert self.pos_item_emb.shape == (B, EMBED_DIM)
        assert self.neg_item_emb.dim() == 3
        assert self.neg_item_emb.size(2) == EMBED_DIM
        assert self.behavior_ids.shape == (B,)
 
@dataclass
class LossOutput:
    total_loss: Tensor   # scalar — call .backward() on this
    bpr_loss: Tensor     # scalar
    cl_loss: Tensor      # scalar
    reg_loss: Tensor     # scalar
 
    def validate(self) -> None:
        for name in ["total_loss", "bpr_loss", "cl_loss", "reg_loss"]:
            assert getattr(self, name).dim() == 0

@dataclass
class EvalInput:
    user_embeddings: Tensor             # (num_eval_users, d)
    item_embeddings: Tensor             # (num_items, d)
    eval_user_ids: Tensor               # (num_eval_users,)
    ground_truth: Dict[int, int]        # {user_id: target_item_id}
    exclude_items: Dict[int, List[int]] # {user_id: [train_pos_items]}
 
    def validate(self) -> None:
        assert self.user_embeddings.size(1) == EMBED_DIM
        assert self.item_embeddings.size(1) == EMBED_DIM
        assert len(self.ground_truth) == self.eval_user_ids.size(0)

@dataclass
class ServingArtifacts:
    user_embeddings_path: str = "user_embeddings.npy"
    item_embeddings_path: str = "item_embeddings.npy"
    faiss_index_path: str     = "item_index.faiss"
    user_id_map_path: str     = "user_id_map.json"
    item_id_map_path: str     = "item_id_map.json"
    product_meta_path: str    = "products.parquet"

@dataclass
class CrossComboWeightSpec:
    """
    W(phi,beta) = W_base + Delta_phi + Delta_beta + A_phi @ B_beta^T
    """
    w_base:     Tuple[int, int] = (EMBED_DIM, EMBED_DIM)
    delta_phi:  Tuple[int, int] = (EMBED_DIM, EMBED_DIM)  # × |R|
    delta_beta: Tuple[int, int] = (EMBED_DIM, EMBED_DIM)  # × |B|
    a_phi:      Tuple[int, int] = (EMBED_DIM, LOW_RANK)   # × |R|
    b_beta:     Tuple[int, int] = (EMBED_DIM, LOW_RANK)   # × |B|
 
    @property
    def total_params(self) -> int:
        d, r = EMBED_DIM, LOW_RANK
        nR, nB = len(RELATION_TYPES), len(BEHAVIOR_TYPES)
        return d*d + nR*d*d + nB*d*d + nR*d*r + nB*d*r

def _self_test() -> None:
    print("=" * 55)
    print("  contracts.py self-test")
    print("=" * 55)
 
    B, Nu, Ni, Nc, Nb = 64, 200, 500, 30, 20
 
    # 1. SampledSubgraph
    sg = SampledSubgraph(
        node_features={
            "user":     torch.randn(Nu, EMBED_DIM),
            "product":  torch.randn(Ni, EMBED_DIM),
            "category": torch.randn(Nc, EMBED_DIM),
            "brand":    torch.randn(Nb, EMBED_DIM),
        },
        edge_index={
            ("user","view","product"):       torch.randint(0, Nu, (2, 1000)),
            ("user","cart","product"):        torch.randint(0, Nu, (2, 300)),
            ("user","purchase","product"):    torch.randint(0, Nu, (2, 100)),
            ("product","belongs_to","category"): torch.randint(0, Ni, (2, 500)),
            ("product","producedBy","brand"):  torch.randint(0, Ni, (2, 500)),
        },
        edge_behavior_origin={
            ("user","view","product"):       torch.zeros(1000, dtype=torch.long),
            ("user","cart","product"):        torch.ones(300, dtype=torch.long),
            ("user","purchase","product"):    torch.full((100,), 2, dtype=torch.long),
            ("product","belongs_to","category"): torch.randint(0, 3, (500,)),
            ("product","producedBy","brand"):  torch.randint(0, 3, (500,)),
        },
        target_user_indices=torch.randint(0, Nu, (B,)),
        node_id_map={t: torch.arange(n) for t, n in
                     [("user",Nu),("product",Ni),("category",Nc),("brand",Nb)]},
    )
    sg.validate()
    print("  [PASS] SampledSubgraph")
 
    # 2. GNNOutput
    gnn = GNNOutput(
        per_behavior_emb={b: {"user": torch.randn(B, EMBED_DIM),
                              "product": torch.randn(B, EMBED_DIM)}
                          for b in BEHAVIOR_TYPES},
        final_user_emb=torch.randn(B, EMBED_DIM),
        final_item_emb=torch.randn(B, EMBED_DIM),
    )
    gnn.validate()
    print("  [PASS] GNNOutput")
 
    # 3. SVDFactors
    svd = SVDFactors(
        US={b: torch.randn(Nu, SVD_RANK) for b in BEHAVIOR_TYPES},
        VS={b: torch.randn(Ni, SVD_RANK) for b in BEHAVIOR_TYPES},
    )
    svd.validate()
    print("  [PASS] SVDFactors")
 
    # 4. GatedOutput
    gated = GatedOutput(torch.randn(B, EMBED_DIM),
                        torch.randn(B, EMBED_DIM),
                        torch.tensor(0.5))
    gated.validate()
    print("  [PASS] GatedOutput")
 
    # 5. LossInput / LossOutput
    li = LossInput(torch.randn(B, EMBED_DIM),
                   torch.randn(B, EMBED_DIM),
                   torch.randn(B, 4, EMBED_DIM),
                   torch.randint(0, 3, (B,)),
                   torch.tensor(0.5))
    li.validate()
    print("  [PASS] LossInput")
 
    lo = LossOutput(*(torch.tensor(x) for x in [1.2, 0.8, 0.3, 0.1]))
    lo.validate()
    print("  [PASS] LossOutput")
 
    # 6. EvalInput
    Ne = 100
    ei = EvalInput(torch.randn(Ne, EMBED_DIM),
                   torch.randn(Ni, EMBED_DIM),
                   torch.arange(Ne),
                   {i: i % Ni for i in range(Ne)},
                   {i: [i % Ni] for i in range(Ne)})
    ei.validate()
    print("[PASS] EvalInput")
 
    # 7. Param count
    spec = CrossComboWeightSpec()
    print(f"\n  Cross-combo params: {spec.total_params:,}"
          f" ({spec.total_params/1e6:.2f}M)")
 
    print("\n" + "=" * 55)
    print("ALL CONTRACTS PASSED")
    print("=" * 55)
 
 
if __name__ == "__main__":
    _self_test()