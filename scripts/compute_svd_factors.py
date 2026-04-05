"""
compute_svd_factors.py — Offline precomputation of SVD factors for contrastive views.

Pipeline position: P1 (runs once after data preprocessing, before training).

Reads:
    graph_dir/{behavior}.npz   — scipy CSR adjacency (user × item), built by run_pipeline
    stats_dir/node_summary.json — num_users, num_products (for shape validation)

Writes:
    out_dir/svd_factors.pt     — SVDFactors dataclass  {US, VS} per behavior
    out_dir/svd_meta.json      — rank, shapes, nnz, timing

SVD convention  (matches SVDFactors in contracts.py):
    A_k  ≈  US_k @ VS_k^T
    US_k = U_k * S_k   shape (num_users, q)
    VS_k = V_k * S_k   shape (num_items, q)

Memory note:
    Randomized SVD (sklearn) never materialises A in dense form.
    Intermediate [q×d] projection in ContrastiveLearning costs ≤ 128 KB.

Usage:
    python scripts/compute_svd_factors.py \
        --graph_dir  /mnt/data/rees46/processed/graph \
        --stats_dir  /mnt/data/rees46/processed/statistics \
        --out_dir    /mnt/data/rees46/processed/svd \
        [--rank 256] [--n_iter 4] [--behaviors view,cart,purchase]
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.utils.extmath import randomized_svd

# project contracts
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.contracts import BEHAVIOR_TYPES, SVD_RANK, SVDFactors

def load_adj(graph_dir: str, behavior: str) -> sp.csr_matrix:
    path = os.path.join(graph_dir, f"{behavior}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Adjacency matrix not found: {path}\n"
            f"Run run_pipeline first to generate it."
        )
    A = sp.load_npz(path).astype(np.float32)
    A = A.tocsr()
    return A


def compute_svd_for_behavior(
    A: sp.csr_matrix,
    rank: int,
    n_iter: int,
    behavior: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns US  [num_users, rank]  and  VS  [num_items, rank]  as float32 Tensors.

    randomized_svd draws random projections — deterministic given random_state.
    Singular values come out in descending order (already correct for sklearn).
    """
    print(f"  [{behavior}] shape={A.shape}, nnz={A.nnz:,}  → SVD(k={rank}, n_iter={n_iter})")
    t0 = time.perf_counter()

    U, S, Vt = randomized_svd(
        A,
        n_components=rank,
        n_iter=n_iter,
        random_state=42,
    )
    # U  : (num_users, rank), float64 from sklearn
    # S  : (rank,)
    # Vt : (rank, num_items)

    # Absorb singular values into both sides so that A ≈ US @ VS^T
    sqrt_S = np.sqrt(S).astype(np.float32)        # (rank,)
    US = (U * sqrt_S).astype(np.float32)           # (num_users, rank)
    VS = (Vt.T * sqrt_S).astype(np.float32)        # (num_items, rank)

    elapsed = time.perf_counter() - t0
    print(f"  [{behavior}] done in {elapsed:.1f}s  "
          f"US={US.shape}  VS={VS.shape}  "
          f"S_max={S[0]:.2f}  S_min={S[-1]:.4f}")

    return torch.from_numpy(US), torch.from_numpy(VS)


def validate_shapes(
    US: torch.Tensor, VS: torch.Tensor,
    num_users: int, num_items: int, rank: int, behavior: str,
) -> None:
    assert US.shape == (num_users, rank), (
        f"[{behavior}] US shape mismatch: got {tuple(US.shape)}, "
        f"expected ({num_users}, {rank})"
    )
    assert VS.shape == (num_items, rank), (
        f"[{behavior}] VS shape mismatch: got {tuple(VS.shape)}, "
        f"expected ({num_items}, {rank})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute and save offline SVD factors for contrastive learning."
    )
    parser.add_argument(
        "--graph_dir", required=True,
        help="Directory containing {behavior}.npz CSR matrices.",
    )
    parser.add_argument(
        "--stats_dir", default=None,
        help="Directory containing node_summary.json (for shape validation). "
             "Skip validation if not provided.",
    )
    parser.add_argument(
        "--out_dir", required=True,
        help="Output directory — will create if not exists.",
    )
    parser.add_argument(
        "--rank", type=int, default=SVD_RANK,
        help=f"SVD rank q (default: {SVD_RANK} from contracts.SVD_RANK).",
    )
    parser.add_argument(
        "--n_iter", type=int, default=4,
        help="Number of power iterations for randomized SVD (higher → more accurate).",
    )
    parser.add_argument(
        "--behaviors", default=",".join(BEHAVIOR_TYPES),
        help=f"Comma-separated behaviors to compute (default: {','.join(BEHAVIOR_TYPES)}).",
    )
    args = parser.parse_args()

    behaviors: list[str] = [b.strip() for b in args.behaviors.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)

    # optional shape validation via node_summary.json
    num_users: int | None = None
    num_items: int | None = None
    if args.stats_dir:
        summary_path = os.path.join(args.stats_dir, "node_summary.json")
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                summary = json.load(f)
            num_users = summary.get("num_users")
            num_items = summary.get("num_products")
            print(f"Node summary: num_users={num_users:,}  num_products={num_items:,}")
        else:
            print(f"[warn] node_summary.json not found at {summary_path} — skipping validation")

    # compute SVD per behavior
    US_dict: dict[str, torch.Tensor] = {}
    VS_dict: dict[str, torch.Tensor] = {}
    meta: dict = {"rank": args.rank, "n_iter": args.n_iter, "factors": {}}

    total_t0 = time.perf_counter()
    for beh in behaviors:
        print(f"\n{beh}")
        A = load_adj(args.graph_dir, beh)

        US, VS = compute_svd_for_behavior(A, rank=args.rank, n_iter=args.n_iter, behavior=beh)

        if num_users is not None and num_items is not None:
            validate_shapes(US, VS, num_users, num_items, args.rank, beh)

        US_dict[beh] = US
        VS_dict[beh] = VS

        meta["factors"][beh] = {
            "adj_shape": list(A.shape),
            "adj_nnz": int(A.nnz),
            "US_shape": list(US.shape),
            "VS_shape": list(VS.shape),
        }
        del A  # free sparse matrix immediately

    total_elapsed = time.perf_counter() - total_t0
    print(f"\nTotal SVD time: {total_elapsed:.1f}s")

    # assemble SVDFactors and save .pt
    svd_factors = SVDFactors(US=US_dict, VS=VS_dict)
    svd_factors.validate()  # contracts check

    out_pt = os.path.join(args.out_dir, "svd_factors.pt")
    torch.save(svd_factors, out_pt)
    size_mb = os.path.getsize(out_pt) / 1024 / 1024
    print(f"\nSaved: {out_pt}  ({size_mb:.1f} MB)")

    # save metadata json
    meta["total_elapsed_s"] = round(total_elapsed, 2)
    meta["output_file"] = out_pt
    out_meta = os.path.join(args.out_dir, "svd_meta.json")
    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved: {out_meta}")

    print("\nDone.")


if __name__ == "__main__":
    main()
