from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

logger = logging.getLogger(__name__)


def sanity_check_heterodata(
    hetero_data: HeteroData,
    train_triplets: torch.Tensor,
    test_triplets: pd.DataFrame | dict[int, int],
    *,
    num_nodes_dict: dict[str, int] | None = None,
    check_leakage: bool = True,
    verbose: bool = True,
) -> None:
    _log = logger.info if verbose else lambda *a, **k: None

    _log("sanity_check_heterodata: starting ...")
    num_nodes: dict[str, int] = {}
    for ntype in hetero_data.node_types:
        if num_nodes_dict is not None and ntype in num_nodes_dict:
            authoritative = num_nodes_dict[ntype]
            stored = int(hetero_data[ntype].num_nodes)
            assert stored == authoritative, (
                f"hetero_data['{ntype}'].num_nodes = {stored} but "
                f"num_nodes_dict['{ntype}'] = {authoritative}.  "
                "The HeteroData node count disagrees with the vocabulary — "
                "one of them is wrong, and the embedding table will be mis-sized."
            )
            num_nodes[ntype] = authoritative
        else:
            n = int(hetero_data[ntype].num_nodes)
            if num_nodes_dict is None:
                logger.debug(
                    "node type '%s': using PyG-inferred num_nodes=%d; pass "
                    "num_nodes_dict for a stricter cold-start check.", ntype, n,
                )
            num_nodes[ntype] = n
        _log("  node type %-12s : num_nodes = %d", ntype, num_nodes[ntype])

    _log("CHECK 1: edge_index dtype, contiguity, and bounds ...")

    for edge_type in hetero_data.edge_types:
        src_type, _, dst_type = edge_type
        ei = hetero_data[edge_type].edge_index

        assert ei.dtype == torch.int64, (
            f"edge_index {edge_type}: dtype is {ei.dtype}, expected torch.int64.  "
            "Non-int64 indices cause incorrect results with nn.Embedding and "
            "scatter_add_ on some CUDA versions."
        )

        assert ei.is_contiguous(), (
            f"edge_index {edge_type}: tensor is not contiguous (likely produced by "
            ".flip(0) without a subsequent .contiguous() call).  PyG's scatter_add_ "
            "will allocate a hidden copy on every forward pass, spiking VRAM."
        )

        if ei.numel() == 0:
            continue

        assert ei.shape[0] == 2, (
            f"edge_index {edge_type}: expected shape (2, E), got {ei.shape}"
        )

        src_idx = ei[0]
        dst_idx = ei[1]

        assert int(src_idx.min().item()) >= 0, (
            f"edge_index {edge_type}: negative source index "
            f"{int(src_idx.min().item())}"
        )
        assert int(dst_idx.min().item()) >= 0, (
            f"edge_index {edge_type}: negative destination index "
            f"{int(dst_idx.min().item())}"
        )

        if src_type in num_nodes:
            n_src = num_nodes[src_type]
            src_max = int(src_idx.max().item())
            assert src_max < n_src, (
                f"edge_index {edge_type}: source index {src_max} >= "
                f"num_nodes['{src_type}'] = {n_src}.  "
                "The GNN embedding lookup will throw a CUDA OOB on this edge."
            )

        if dst_type in num_nodes:
            n_dst = num_nodes[dst_type]
            dst_max = int(dst_idx.max().item())
            assert dst_max < n_dst, (
                f"edge_index {edge_type}: destination index {dst_max} >= "
                f"num_nodes['{dst_type}'] = {n_dst}.  "
                "The GNN embedding lookup will throw a CUDA OOB on this edge."
            )

    _log("  PASS: all %d edge types are int64, contiguous, and within bounds.",
         len(hetero_data.edge_types))

    _log("CHECK 2: bipartite user/item index spaces in train_triplets ...")

    assert train_triplets.ndim == 2 and train_triplets.size(1) >= 2, (
        f"train_triplets must have shape (N, >=2), got {tuple(train_triplets.shape)}"
    )

    train_user_idx = train_triplets[:, 0]
    train_item_idx = train_triplets[:, 1]

    n_users   = num_nodes.get("user",    0)
    n_items   = num_nodes.get("product", 0)

    assert int(train_user_idx.min().item()) >= 0, (
        "train_triplets: negative user index detected"
    )
    assert int(train_user_idx.max().item()) < n_users, (
        f"train_triplets: max user_idx {int(train_user_idx.max().item())} "
        f">= num_nodes['user'] = {n_users}"
    )

    assert int(train_item_idx.min().item()) >= 0, (
        "train_triplets: negative item index detected"
    )
    assert int(train_item_idx.max().item()) < n_items, (
        f"train_triplets: max item_idx {int(train_item_idx.max().item())} "
        f">= num_nodes['product'] = {n_items}"
    )

    _log("  PASS: train_triplets user ∈ [0, %d), item ∈ [0, %d).", n_users, n_items)

    _log("CHECK 3: test_triplets bounds ...")

    if isinstance(test_triplets, dict):
        test_user_arr = np.fromiter(test_triplets.keys(),   dtype=np.int64)
        test_item_arr = np.fromiter(test_triplets.values(), dtype=np.int64)
    elif isinstance(test_triplets, pd.DataFrame):
        test_user_arr = test_triplets["user_idx"].to_numpy(dtype=np.int64)
        test_item_arr = test_triplets["item_idx"].to_numpy(dtype=np.int64)
    else:
        raise TypeError(
            f"test_triplets must be a DataFrame or dict, got {type(test_triplets)}"
        )

    assert int(test_user_arr.min()) >= 0, (
        "test_triplets: negative user index detected"
    )
    assert int(test_user_arr.max()) < n_users, (
        f"test_triplets: max user_idx {int(test_user_arr.max())} "
        f">= num_nodes['user'] = {n_users}.  Cold-start user reached eval."
    )

    assert int(test_item_arr.min()) >= 0, (
        "test_triplets: negative item index detected"
    )
    assert int(test_item_arr.max()) < n_items, (
        f"test_triplets: max item_idx {int(test_item_arr.max())} "
        f">= num_nodes['product'] = {n_items}.  "
        "This index will cause a CUDA OOB when the evaluator does "
        "item_embs[cand_t] during scoring."
    )

    train_user_set = set(train_user_idx.tolist())
    test_user_set  = set(test_user_arr.tolist())
    unseen_users   = test_user_set - train_user_set
    assert not unseen_users, (
        f"test_triplets contains {len(unseen_users)} user(s) with no training "
        f"interactions (e.g. {sorted(unseen_users)[:5]}).  "
        "Their embeddings are random noise; drop them before evaluation."
    )

    train_item_set = set(train_item_idx.tolist())
    test_item_set  = set(test_item_arr.tolist())
    cold_start_items = test_item_set - train_item_set
    if cold_start_items:
        _log(
            "  INFO: %d cold-start item(s) in test set (e.g. %s) — "
            "embedded via structural graph (category/brand), not random noise.",
            len(cold_start_items), sorted(cold_start_items)[:5],
        )

    _log(
        "  PASS: test user ∈ [0, %d), item ∈ [0, %d); "
        "%d cold-start item(s) handled via structural graph.",
        n_users, n_items, len(cold_start_items),
    )

    if check_leakage:
        _log("CHECK 4: no train/test leakage on (user_idx, item_idx) pairs ...")

        train_pairs = set(
            zip(train_user_idx.tolist(), train_item_idx.tolist())
        )
        test_pairs = set(zip(test_user_arr.tolist(), test_item_arr.tolist()))

        leaked = train_pairs & test_pairs
        assert not leaked, (
            f"Data leakage: {len(leaked)} (user, item) pair(s) appear in both "
            f"train purchases and the test set (e.g. {list(leaked)[:3]}).  "
            "The temporal split should have excluded the test interaction from train."
        )

        _log("  PASS: zero leaking (user, item) pairs between train and test.")

    _log(
        "sanity_check_heterodata: ALL CHECKS PASSED.  "
        "Safe to initialise model and start training."
    )
