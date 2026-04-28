from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json_atomic(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2, default=str)
    os.replace(tmp, path)


def _git_commit() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return out or None
    except Exception:
        return None


def write_split_manifest(
    path: str,
    *,
    behaviors: list[str],
    target_behavior: str,
    protocol_name: str,
    timezone_str: str,
    train_end: str,
    val_end: str,
    train_cutoff_ts: int,
    val_cutoff_ts: int,
    candidate_set: str,
    ground_truth: str,
    primary_mask_behaviors: list[str],
    transductive_item_vocab: bool,
    allow_transductive_item_metadata: bool,
    extra: dict[str, Any] | None = None,
) -> None:
    obj: dict[str, Any] = {
        "dataset":                            "REES46 / ecommerce behavior data",
        "behaviors":                          list(behaviors),
        "target_behavior":                    target_behavior,
        "protocol_name":                      protocol_name,
        "timezone":                           timezone_str,
        "train_end":                          train_end,
        "val_end":                            val_end,
        "train_cutoff_ts":                    int(train_cutoff_ts),
        "val_cutoff_ts":                      int(val_cutoff_ts),
        "filter_train_only":                  True,
        "candidate_set":                      candidate_set,
        "ground_truth":                       ground_truth,
        "primary_mask_behaviors":             list(primary_mask_behaviors),
        "transductive_item_vocab":            bool(transductive_item_vocab),
        "allow_transductive_item_metadata":   bool(allow_transductive_item_metadata),
        "generated_at_utc":                   datetime.now(timezone.utc).isoformat(),
        "git_commit":                         _git_commit(),
    }
    if extra:
        obj.update(extra)
    write_json_atomic(obj, path)
    logger.info("split manifest written: %s", path)


def build_artifacts_manifest(roots: dict[str, str]) -> dict[str, Any]:
    """Walk each root directory and hash every file inside."""
    files: list[dict[str, Any]] = []
    for label, root in roots.items():
        if not os.path.isdir(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for fn in sorted(filenames):
                fp = os.path.join(dirpath, fn)
                try:
                    stat = os.stat(fp)
                    files.append({
                        "label":   label,
                        "path":    os.path.relpath(fp, start=root),
                        "abspath": fp,
                        "size":    stat.st_size,
                        "sha256":  file_sha256(fp),
                    })
                except OSError as e:
                    logger.warning("manifest: failed to hash %s (%s)", fp, e)
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit":       _git_commit(),
        "files":            files,
    }


def write_artifacts_manifest(path: str, roots: dict[str, str]) -> None:
    obj = build_artifacts_manifest(roots)
    write_json_atomic(obj, path)
    logger.info("artifacts manifest written: %s (%d files)", path, len(obj["files"]))


def write_data_card(
    json_path: str,
    md_path: str,
    *,
    raw_rows: int | None,
    cleaned_rows: int | None,
    behavior_counts_overall: dict[str, int],
    behavior_counts_by_split: dict[str, dict[str, int]],
    num_users: int,
    num_items: int,
    num_categories: int,
    num_brands: int,
    train_purchase_pairs: int,
    val_positives: int,
    test_positives: int,
    val_users: int,
    test_users: int,
    cold_users_dropped: dict[str, int],
    cold_items_dropped: dict[str, int],
    repeated_purchase_dropped: dict[str, int],
    metadata_missingness: dict[str, int],
    edge_artifact_counts: dict[str, int],
    mask_stats: dict[str, dict[str, int | float]],
    leakage_check: dict[str, str],
    config: dict[str, Any],
    train_cutoff_ts: int,
    val_cutoff_ts: int,
    train_end: str,
    val_end: str,
    protocol_name: str,
) -> None:
    card_obj: dict[str, Any] = {
        "protocol_name":              protocol_name,
        "train_end":                  train_end,
        "val_end":                    val_end,
        "train_cutoff_ts":            int(train_cutoff_ts),
        "val_cutoff_ts":              int(val_cutoff_ts),
        "raw_rows":                   raw_rows,
        "cleaned_rows":               cleaned_rows,
        "behavior_counts_overall":    behavior_counts_overall,
        "behavior_counts_by_split":   behavior_counts_by_split,
        "num_users_train_vocab":      int(num_users),
        "num_items_train_vocab":      int(num_items),
        "num_categories":             int(num_categories),
        "num_brands":                 int(num_brands),
        "train_purchase_pairs":       int(train_purchase_pairs),
        "val_positives":              int(val_positives),
        "test_positives":             int(test_positives),
        "val_users":                  int(val_users),
        "test_users":                 int(test_users),
        "cold_users_dropped":         cold_users_dropped,
        "cold_items_dropped":         cold_items_dropped,
        "repeated_purchase_dropped":  repeated_purchase_dropped,
        "metadata_missingness":       metadata_missingness,
        "edge_artifact_counts":       edge_artifact_counts,
        "mask_stats":                 mask_stats,
        "leakage_check":              leakage_check,
        "config":                     config,
        "git_commit":                 _git_commit(),
        "generated_at_utc":           datetime.now(timezone.utc).isoformat(),
    }
    write_json_atomic(card_obj, json_path)

    md_lines = [
        f"# Data Card  ({protocol_name})",
        "",
        f"- Protocol: {protocol_name}",
        f"- Train end: {train_end}  (ts < {train_cutoff_ts})",
        f"- Val end:   {val_end}    (ts < {val_cutoff_ts})",
        f"- Generated at: {card_obj['generated_at_utc']} UTC",
        f"- Git commit: {card_obj['git_commit']}",
        "",
        "## Row counts",
        f"- raw rows: {raw_rows}",
        f"- cleaned rows: {cleaned_rows}",
        "",
        "## Behavior counts overall",
    ]
    for b, n in behavior_counts_overall.items():
        md_lines.append(f"- {b}: {n}")

    md_lines += ["", "## Behavior counts by split"]
    for split_name, counts in behavior_counts_by_split.items():
        md_lines.append(f"- {split_name}: " + ", ".join(f"{k}={v}" for k, v in counts.items()))

    md_lines += [
        "",
        "## Vocabulary",
        f"- users (train vocab): {num_users}",
        f"- items (train vocab): {num_items}",
        f"- categories: {num_categories}",
        f"- brands: {num_brands}",
        "",
        "## Train / eval positives",
        f"- train purchase pairs: {train_purchase_pairs}",
        f"- val positives: {val_positives}  (across {val_users} users)",
        f"- test positives: {test_positives}  (across {test_users} users)",
        "",
        "## Cold-start drops",
        f"- val cold users dropped: {cold_users_dropped.get('val', 0)}",
        f"- test cold users dropped: {cold_users_dropped.get('test', 0)}",
        f"- val cold items dropped: {cold_items_dropped.get('val', 0)}",
        f"- test cold items dropped: {cold_items_dropped.get('test', 0)}",
        "",
        "## Repeated train purchase drops (primary new-purchase)",
        f"- val: {repeated_purchase_dropped.get('val', 0)}",
        f"- test: {repeated_purchase_dropped.get('test', 0)}",
        "",
        "## Metadata missingness",
    ]
    for k, v in metadata_missingness.items():
        md_lines.append(f"- {k}: {v}")

    md_lines += ["", "## Edge artifact counts"]
    for k, v in edge_artifact_counts.items():
        md_lines.append(f"- {k}: {v}")

    md_lines += ["", "## Mask stats"]
    for k, v in mask_stats.items():
        md_lines.append(f"- {k}: {v}")

    md_lines += ["", "## Leakage check"]
    for k, v in leakage_check.items():
        md_lines.append(f"- {k}: {v}")

    os.makedirs(os.path.dirname(md_path), exist_ok=True)
    with open(md_path, "w") as fh:
        fh.write("\n".join(md_lines) + "\n")
    logger.info("data card written: %s and %s", json_path, md_path)


def write_baseline_contract(
    path: str,
    *,
    protocol_name: str,
    primary_metric: str,
    metrics: list[str],
    primary_mask_behaviors: list[str],
    candidate_set: str,
    ground_truth: str,
) -> None:
    lines = [
        "# Baseline Contract",
        "",
        f"All baselines (GraphSAGE, CRGCN, KHGT, MixRec, BPATMP++) must use the same:",
        "- `split_manifest.json` from this directory.",
        "- candidate item set (warm train items only).",
        "- `train_mask_purchase_only.pkl` for primary masking.",
        "- `val_ground_truth.parquet` / `test_ground_truth.parquet` (multi-positive).",
        "- `user2idx.json`, `item2idx.json`, `behavior2idx.json` mappings.",
        "- full-ranking evaluator with consistent metric implementation.",
        "",
        f"## Protocol: {protocol_name}",
        f"- primary mask behaviors: {primary_mask_behaviors}",
        f"- candidate set: {candidate_set}",
        f"- ground truth: {ground_truth}",
        "",
        "## Metrics",
        f"- primary: {primary_metric}",
        f"- all: {', '.join(metrics)}",
        "",
        "## Invalid baseline runs",
        "- sampled-negative evaluation while others use full-ranking",
        "- different user/item filtering",
        "- masking train view/cart in primary",
        "- last-purchase-per-user when primary expects all purchases",
        "- using full data to build vocab in primary protocol",
        "- using val/test interactions in the train graph",
        "- copying paper numbers from other datasets as REES46 results",
        "",
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    logger.info("baseline contract written: %s", path)
