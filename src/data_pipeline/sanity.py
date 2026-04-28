from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import torch
    from torch_geometric.data import HeteroData

logger = logging.getLogger(__name__)


def sanity_check_heterodata(
    hetero_data: "HeteroData",
    train_triplets: "torch.Tensor",
    test_triplets: pd.DataFrame | dict[int, int] | dict[int, list[int]],
    *,
    num_nodes_dict: dict[str, int] | None = None,
    check_leakage: bool = True,
    verbose: bool = True,
) -> None:
    import torch
    _log = logger.info if verbose else lambda *a, **k: None

    _log("sanity_check_heterodata: starting ...")
    num_nodes: dict[str, int] = {}
    for ntype in hetero_data.node_types:
        if num_nodes_dict is not None and ntype in num_nodes_dict:
            authoritative = num_nodes_dict[ntype]
            stored = int(hetero_data[ntype].num_nodes)
            assert stored == authoritative, (
                f"hetero_data['{ntype}'].num_nodes = {stored} but "
                f"num_nodes_dict['{ntype}'] = {authoritative}."
            )
            num_nodes[ntype] = authoritative
        else:
            num_nodes[ntype] = int(hetero_data[ntype].num_nodes)
        _log("  node type %-12s : num_nodes = %d", ntype, num_nodes[ntype])

    _log("CHECK 1: edge_index dtype, contiguity, and bounds ...")

    for edge_type in hetero_data.edge_types:
        src_type, _, dst_type = edge_type
        ei = hetero_data[edge_type].edge_index

        assert ei.dtype == torch.int64, (
            f"edge_index {edge_type}: dtype is {ei.dtype}, expected torch.int64."
        )
        assert ei.is_contiguous(), (
            f"edge_index {edge_type}: tensor is not contiguous."
        )

        if ei.numel() == 0:
            continue

        assert ei.shape[0] == 2, (
            f"edge_index {edge_type}: expected shape (2, E), got {ei.shape}"
        )

        src_idx = ei[0]
        dst_idx = ei[1]

        assert int(src_idx.min().item()) >= 0
        assert int(dst_idx.min().item()) >= 0
        if src_type in num_nodes:
            assert int(src_idx.max().item()) < num_nodes[src_type], (
                f"edge_index {edge_type}: src out of range"
            )
        if dst_type in num_nodes:
            assert int(dst_idx.max().item()) < num_nodes[dst_type], (
                f"edge_index {edge_type}: dst out of range"
            )

    _log("  PASS: %d edge types within bounds.", len(hetero_data.edge_types))

    _log("CHECK 2: train_triplets bipartite ranges ...")

    assert train_triplets.ndim == 2 and train_triplets.size(1) >= 2

    train_user_idx = train_triplets[:, 0]
    train_item_idx = train_triplets[:, 1]

    n_users = num_nodes.get("user",    0)
    n_items = num_nodes.get("product", 0)

    if train_user_idx.numel() > 0:
        assert int(train_user_idx.min().item()) >= 0
        assert int(train_user_idx.max().item()) < n_users
    if train_item_idx.numel() > 0:
        assert int(train_item_idx.min().item()) >= 0
        assert int(train_item_idx.max().item()) < n_items

    _log("  PASS: train_triplets within [0,%d) x [0,%d).", n_users, n_items)

    _log("CHECK 3: eval pairs in vocab ...")

    if isinstance(test_triplets, dict):
        keys = list(test_triplets.keys())
        if keys and isinstance(test_triplets[keys[0]], (list, tuple, set, np.ndarray)):
            users, items = [], []
            for u, it_list in test_triplets.items():
                for it in it_list:
                    users.append(int(u))
                    items.append(int(it))
            test_user_arr = np.asarray(users, dtype=np.int64)
            test_item_arr = np.asarray(items, dtype=np.int64)
        else:
            test_user_arr = np.fromiter(test_triplets.keys(),   dtype=np.int64)
            test_item_arr = np.fromiter(test_triplets.values(), dtype=np.int64)
    elif isinstance(test_triplets, pd.DataFrame):
        test_user_arr = test_triplets["user_idx"].to_numpy(dtype=np.int64)
        test_item_arr = test_triplets["item_idx"].to_numpy(dtype=np.int64)
    else:
        raise TypeError(f"test_triplets must be dict or DataFrame, got {type(test_triplets)}")

    if test_user_arr.size > 0:
        assert int(test_user_arr.min()) >= 0
        assert int(test_user_arr.max()) < n_users, (
            f"eval user_idx max {int(test_user_arr.max())} >= n_users {n_users}"
        )
        assert int(test_item_arr.min()) >= 0
        assert int(test_item_arr.max()) < n_items, (
            f"eval item_idx max {int(test_item_arr.max())} >= n_items {n_items}"
        )

    _log("  PASS: eval pairs within vocab ranges.")

    if check_leakage and test_user_arr.size > 0:
        _log("CHECK 4: no train/test pair leakage ...")
        train_pairs = set(zip(train_user_idx.tolist(), train_item_idx.tolist()))
        test_pairs = set(zip(test_user_arr.tolist(), test_item_arr.tolist()))
        leaked = train_pairs & test_pairs
        assert not leaked, f"Leakage: {len(leaked)} pairs in both train and eval"
        _log("  PASS: zero leaking pairs.")

    _log("sanity_check_heterodata: ALL CHECKS PASSED.")


def sanity_check_temporal_artifacts(
    behavior_artifacts: dict[str, dict[str, np.ndarray]],
    val_ts: np.ndarray,
    test_ts: np.ndarray,
    train_cutoff_ts: int,
    val_cutoff_ts: int,
) -> None:
    """Check edge timestamp lengths/dtypes and train/val/test interval bounds."""
    logger.info("sanity_check_temporal_artifacts: starting ...")

    for beh, arrs in behavior_artifacts.items():
        src, dst, ts = arrs["src"], arrs["dst"], arrs["ts"]
        assert src.dtype == np.int64, f"{beh}_src dtype {src.dtype}"
        assert dst.dtype == np.int64, f"{beh}_dst dtype {dst.dtype}"
        assert ts.dtype  == np.int64, f"{beh}_ts dtype {ts.dtype}"
        assert len(src) == len(dst) == len(ts), (
            f"{beh}: length mismatch src={len(src)} dst={len(dst)} ts={len(ts)}"
        )
        if len(ts) > 0:
            ts_max = int(ts.max())
            assert ts_max < train_cutoff_ts, (
                f"{beh}_train_ts.max() = {ts_max} >= train_cutoff_ts = {train_cutoff_ts} "
                "(future leakage in train edges)"
            )

    assert val_ts.dtype  == np.int64
    assert test_ts.dtype == np.int64

    if val_ts.size > 0:
        assert int(val_ts.min()) >= train_cutoff_ts, (
            f"val_ts.min() = {int(val_ts.min())} < train_cutoff_ts = {train_cutoff_ts}"
        )
        assert int(val_ts.max()) <  val_cutoff_ts, (
            f"val_ts.max() = {int(val_ts.max())} >= val_cutoff_ts = {val_cutoff_ts}"
        )
    if test_ts.size > 0:
        assert int(test_ts.min()) >= val_cutoff_ts, (
            f"test_ts.min() = {int(test_ts.min())} < val_cutoff_ts = {val_cutoff_ts}"
        )

    logger.info("  PASS: temporal artifacts respect split intervals.")


def sanity_check_ground_truth(
    gt: dict[int, list[int]],
    parquet_df: pd.DataFrame,
    *,
    split_name: str = "test",
) -> None:
    """Ensure multi-positive ground truth was not collapsed."""
    logger.info("sanity_check_ground_truth(%s): starting ...", split_name)

    pq_pairs = set(
        (int(u), int(i)) for u, i in zip(parquet_df["user_idx"], parquet_df["item_idx"])
    )
    gt_pairs = set(
        (int(u), int(i)) for u, items in gt.items() for i in items
    )

    assert pq_pairs == gt_pairs, (
        f"{split_name}: ground truth pkl ({len(gt_pairs)} pairs) does not match "
        f"parquet ({len(pq_pairs)} pairs); difference={len(pq_pairs ^ gt_pairs)}"
    )

    n_total = sum(len(v) for v in gt.values())
    assert n_total == len(gt_pairs), (
        f"{split_name}: ground truth contains duplicate (user,item) pairs"
    )

    logger.info(
        "  PASS: %s ground truth has %d positives across %d users.",
        split_name, len(gt_pairs), len(gt),
    )


def sanity_check_eval_mask(
    primary_mask: dict[int, list[int]],
    gt: dict[int, list[int]],
    *,
    split_name: str = "test",
) -> None:
    """For every eval positive (u, i), assert i is not in the primary mask."""
    logger.info("sanity_check_eval_mask(%s): starting ...", split_name)

    n_violations = 0
    examples = []
    for u, items in gt.items():
        masked = primary_mask.get(int(u), [])
        if not masked:
            continue
        masked_set = set(int(x) for x in masked)
        for i in items:
            if int(i) in masked_set:
                n_violations += 1
                if len(examples) < 5:
                    examples.append((int(u), int(i)))

    assert n_violations == 0, (
        f"{split_name}: {n_violations} eval positive(s) appear in the primary mask, "
        f"e.g. {examples}. Repeated train purchases must be dropped from eval."
    )
    logger.info("  PASS: %s positives are disjoint from primary mask.", split_name)
