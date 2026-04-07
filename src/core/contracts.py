from __future__ import annotations
 
from dataclasses import dataclass, field
import torch
from torch import Tensor

EMBED_DIM: int = 128
NUM_HEADS: int = 4
HEAD_DIM: int = EMBED_DIM // NUM_HEADS  # 32
NUM_GNN_LAYERS: int = 2
LOW_RANK: int = 16  # r for A_phi @ B_beta^T
SVD_RANK: int = 256 # q for randomized SVD
 
NODE_TYPES: list[str] = ["user", "product", "category", "brand"]
BEHAVIOR_TYPES: list[str] = ["view", "cart", "purchase"]
TARGET_BEHAVIOR: str = "purchase"

RELATION_TYPES: list[tuple[str, str, str]] = [
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

BEHAVIOR_LOSS_WEIGHTS: dict[str, float] = {
    "purchase": 1.0,
    "cart":     0.3,
    "view":     0.1,
}

BEHAVIOR_TO_ID: dict[str, int] = {"view": 0, "cart": 1, "purchase": 2}
ID_TO_BEHAVIOR: dict[int, str] = {v: k for k, v in BEHAVIOR_TO_ID.items()}

@dataclass
class SampledSubgraph:
    # {node_type: Tensor(num_nodes_of_type, EMBED_DIM)}
    node_features: dict[str, Tensor]

    # {(src_type, edge_type, dst_type): Tensor(2, num_edges)}
    edge_index: dict[tuple[str, str, str], Tensor]

    # {(src_type, edge_type, dst_type): Tensor(num_edges,)}
    # values in {0=view, 1=cart, 2=purchase}
    edge_behavior_origin: dict[tuple[str, str, str], Tensor]

    # Indices into node_features["user"] for this batch
    # shape: (batch_size,)
    target_user_indices: Tensor

    node_id_map: dict[str, Tensor]
 
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
    per_behavior_emb: dict[str, dict[str, Tensor]]
 
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
    US: dict[str, Tensor]   # {behavior: (num_users, q)}
    VS: dict[str, Tensor]   # {behavior: (num_items, q)}
 
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
    ground_truth: dict[int, int]        # {user_id: target_item_id}
    exclude_items: dict[int, list[int]] # {user_id: [train_pos_items]}
 
    def validate(self, total_num_users: int | None = None) -> None:
        """Validate EvalInput tensor shapes and index bounds.

        Parameters
        ----------
        total_num_users:
            The authoritative total user count from ``node_counts.json``
            (i.e. ``SplitResult.num_users``).  Pass this to enable a strict
            bounds check on ``eval_user_ids``, which holds *global* user
            indices in ``[0, total_num_users)`` — NOT positions into
            ``user_embeddings`` (which has size ``N_eval_users``).
            When omitted the global-ID bounds check is skipped.
        """
        n_eval_users = self.user_embeddings.size(0)
        num_items    = self.item_embeddings.size(0)

        assert self.user_embeddings.size(1) == EMBED_DIM, (
            f"user_embeddings dim1={self.user_embeddings.size(1)}, expected {EMBED_DIM}"
        )
        assert self.item_embeddings.size(1) == EMBED_DIM, (
            f"item_embeddings dim1={self.item_embeddings.size(1)}, expected {EMBED_DIM}"
        )

        # user_embeddings is ordered by POSITION in eval_user_ids, not by
        # global user index.  Check the positional pairing, not a value bound.
        assert n_eval_users == self.eval_user_ids.size(0), (
            f"user_embeddings has {n_eval_users} rows but "
            f"eval_user_ids has {self.eval_user_ids.size(0)} entries — "
            "they must have the same length (embeddings are ordered by eval_user_ids position)"
        )
        assert len(self.ground_truth) == self.eval_user_ids.size(0), (
            f"ground_truth has {len(self.ground_truth)} entries but "
            f"eval_user_ids has {self.eval_user_ids.size(0)}"
        )

        # eval_user_ids holds GLOBAL user indices; check against total vocab size
        # when the authoritative count is provided.
        if total_num_users is not None:
            uid_max = int(self.eval_user_ids.max().item())
            assert uid_max < total_num_users, (
                f"eval_user_ids contains global index {uid_max} >= "
                f"total_num_users {total_num_users}.  "
                "This is a vocabulary mismatch between the saved mapping and "
                "the node counts passed to the model."
            )

        # Item bounds: ground_truth values are global item indices into item_embeddings
        # (item_embeddings covers ALL items, unlike user_embeddings which covers only eval users).
        gt_items = list(self.ground_truth.values())
        item_min = min(gt_items)
        item_max = max(gt_items)
        assert item_min >= 0, (
            f"ground_truth contains negative item index {item_min}"
        )
        assert item_max < num_items, (
            f"ground_truth contains item index {item_max} >= "
            f"item_embeddings.size(0) {num_items}.  "
            "This will cause a CUDA OOB during evaluator scoring."
        )

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
    w_base:     tuple[int, int] = (EMBED_DIM, EMBED_DIM)
    delta_phi:  tuple[int, int] = (EMBED_DIM, EMBED_DIM)  # × |R|
    delta_beta: tuple[int, int] = (EMBED_DIM, EMBED_DIM)  # × |B|
    a_phi:      tuple[int, int] = (EMBED_DIM, LOW_RANK)   # × |R|
    b_beta:     tuple[int, int] = (EMBED_DIM, LOW_RANK)   # × |B|
 
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