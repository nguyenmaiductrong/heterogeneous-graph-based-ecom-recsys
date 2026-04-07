from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["DataSplitter", "SplitResult"]


@dataclass
class SplitResult:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    user2idx: dict
    item2idx: dict
    num_users: int
    num_items: int
    stats: dict = field(default_factory=dict)
    # Populated by temporal_split_by_dates. prepare_data.py uses this as the
    # global auxiliary cutoff: any aux row with timestamp >= train_end_ts is
    # dropped from the training graph to prevent future-signal leakage.
    train_end_ts: int | None = None

    def summary(self) -> str:
        lines = [
            f"num_users              : {self.num_users:>10,}",
            f"num_items              : {self.num_items:>10,}",
            f"train rows             : {len(self.train):>10,}",
            f"val   rows             : {len(self.val):>10,}",
            f"test  rows             : {len(self.test):>10,}",
        ]
        if self.train_end_ts is not None:
            lines.append(f"train_end_ts           : {self.train_end_ts:>10,}")
        for k, v in self.stats.items():
            lines.append(f"{k:<30s}: {v}")
        return "\n".join(lines)


class DataSplitter:
    """Global Temporal Split with cold-start item support.

    Only ``temporal_split_by_dates`` is supported — the LOO and ratio-based
    protocols have been removed.

    Cold-start items (first purchased after the training cutoff) are included
    in the global item vocabulary so BAGNN can represent them via the structural
    (category / brand) graph. Excluding them would cause out-of-index errors
    during evaluation when the scorer accesses ``item_embs[cand_t]``.

    Cold-start users (never active during the training window) are dropped from
    val / test because they have no purchase graph edges and their GNN embeddings
    would be random noise.

    Parameters
    ----------
    df:
        Interaction DataFrame. Must contain ``user_id``, ``item_id``,
        and ``timestamp``. ``event_type`` is optional.
    """

    _REQUIRED = {"user_id", "item_id", "timestamp"}

    def __init__(self, df: pd.DataFrame) -> None:
        missing = self._REQUIRED - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame is missing required columns: {missing}")
        self._df = df.copy()
        self._has_event_type = "event_type" in df.columns

    # ------------------------------------------------------------------
    # Public split method
    # ------------------------------------------------------------------

    def temporal_split_by_dates(
        self,
        train_end: str,
        val_end: str,
    ) -> SplitResult:
        """Date-based global chronological split using explicit ISO 8601 cutoffs.

        Dates are interpreted as inclusive end-of-day boundaries in UTC:

        - Training   : timestamp <  midnight of (train_end + 1 day)
        - Validation : timestamp in [midnight of (train_end+1 day),
                                     midnight of (val_end+1 day))
        - Test        : timestamp >= midnight of (val_end + 1 day)

        This matches the REES46 convention defined in ``spark_config.yaml``:
        ``split.train_end`` / ``split.val_end``.

        Cold-start items (only purchased after train_end) receive valid contiguous
        indices and BAGNN embeds them via category / brand structural edges.
        Cold-start users (not in the training window) are dropped from val / test.
        """
        train_end_ts = int(
            (pd.Timestamp(train_end, tz="UTC") + pd.Timedelta(days=1)).timestamp()
        )
        val_end_ts = int(
            (pd.Timestamp(val_end, tz="UTC") + pd.Timedelta(days=1)).timestamp()
        )

        if train_end_ts >= val_end_ts:
            raise ValueError(
                f"train_end ({train_end!r}) must be strictly before val_end ({val_end!r})."
            )

        df = self._df.copy()
        n = len(df)

        train_raw = df.loc[df["timestamp"] <  train_end_ts].copy()
        val_raw   = df.loc[
            (df["timestamp"] >= train_end_ts) & (df["timestamp"] < val_end_ts)
        ].copy()
        test_raw  = df.loc[df["timestamp"] >= val_end_ts].copy()

        logger.info(
            "Temporal date split — train_end=%s (ts<%d)  val_end=%s (ts<%d)",
            train_end, train_end_ts, val_end, val_end_ts,
        )
        logger.info(
            "  raw rows: train=%d  val=%d  test=%d  (total=%d)",
            len(train_raw), len(val_raw), len(test_raw), n,
        )

        stats = {
            "split_protocol":  "temporal_dates",
            "train_end_date":  train_end,
            "val_end_date":    val_end,
            "train_cutoff_ts": train_end_ts,
            "val_cutoff_ts":   val_end_ts,
            "train_pct":       f"{len(train_raw) / max(n, 1) * 100:.1f}%",
            "val_pct":         f"{len(val_raw)   / max(n, 1) * 100:.1f}%",
            "test_pct":        f"{len(test_raw)  / max(n, 1) * 100:.1f}%",
        }
        return self._finalize(
            train_raw, val_raw, test_raw, stats, train_end_ts=train_end_ts
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_mappings(self, train_df: pd.DataFrame) -> tuple[dict, dict]:
        """Build contiguous index mappings for users and items.

        user2idx — training users only.
            Cold-start users (absent from training) are excluded; they have no
            purchase graph edges and produce random-noise GNN embeddings.

        item2idx — ALL items across every split (self._df).
            Cold-start items (only purchased after the training cutoff) receive
            valid contiguous indices. BAGNN embeds them via the structural graph
            (category / brand edges). Excluding them would cause out-of-index
            CUDA errors when the evaluator accesses ``item_embs[cand_t]``.
        """
        user2idx = {u: i for i, u in enumerate(sorted(train_df["user_id"].unique()))}
        item2idx = {it: i for i, it in enumerate(sorted(self._df["item_id"].unique()))}
        return user2idx, item2idx

    def _apply_mapping(
        self,
        df: pd.DataFrame,
        user2idx: dict,
        item2idx: dict,
        drop_cold_users: bool,
    ) -> pd.DataFrame:
        out = df.copy()

        if drop_cold_users:
            mask = out["user_id"].isin(user2idx)
            n_dropped = int((~mask).sum())
            if n_dropped > 0:
                logger.warning(
                    "Dropped %d rows for cold-start users (not in train vocabulary).",
                    n_dropped,
                )
            out = out.loc[mask].copy()

        out["user_idx"] = out["user_id"].map(user2idx)
        out["item_idx"] = out["item_id"].map(item2idx)

        # NaN after .map() is a hard bug:
        # - user NaN: impossible — cold-start users were filtered above, and
        #   training users are always in user2idx by construction.
        # - item NaN: impossible — item2idx is built from self._df (all data),
        #   so every item in any split is guaranteed a valid mapping.
        if not out.empty and (out["user_idx"].isna().any() or out["item_idx"].isna().any()):
            raise RuntimeError(
                "NaN in mapped indices after vocabulary lookup — this is a bug."
            )

        out["user_idx"] = out["user_idx"].astype(np.int64)
        out["item_idx"] = out["item_idx"].astype(np.int64)

        keep = ["user_idx", "item_idx", "timestamp"]
        if self._has_event_type:
            keep.append("event_type")
        return out[keep].reset_index(drop=True)

    def _finalize(
        self,
        train_raw: pd.DataFrame,
        val_raw: pd.DataFrame,
        test_raw: pd.DataFrame,
        stats: dict,
        train_end_ts: int | None = None,
    ) -> SplitResult:
        user2idx, item2idx = self._build_mappings(train_raw)

        train = self._apply_mapping(train_raw, user2idx, item2idx, drop_cold_users=False)

        val_n_in = len(val_raw)
        val = self._apply_mapping(val_raw, user2idx, item2idx, drop_cold_users=True)

        test_n_in = len(test_raw)
        test = self._apply_mapping(test_raw, user2idx, item2idx, drop_cold_users=True)

        num_users = len(user2idx)
        num_items = len(item2idx)

        # Hard bounds check — any failure here indicates a bug in this module.
        for name, split_df in (("train", train), ("val", val), ("test", test)):
            if split_df.empty:
                continue
            if split_df["user_idx"].max() >= num_users:
                raise RuntimeError(
                    f"{name} user_idx {split_df['user_idx'].max()} >= num_users {num_users}"
                )
            if split_df["item_idx"].max() >= num_items:
                raise RuntimeError(
                    f"{name} item_idx {split_df['item_idx'].max()} >= num_items {num_items}"
                )
            if split_df["user_idx"].min() < 0 or split_df["item_idx"].min() < 0:
                raise RuntimeError(f"{name} contains a negative index.")

        # De-leakage: remove from train any (user_idx, item_idx) pair that also
        # appears in test. A user can legitimately purchase the same item in both
        # the training and test windows; keeping it in the training purchase graph
        # inflates eval metrics (the GNN already has that edge), so we drop it.
        n_train_leaked = 0
        if not test.empty:
            test_keys = test[["user_idx", "item_idx"]].drop_duplicates().assign(_leak=True)
            merged = train[["user_idx", "item_idx"]].merge(
                test_keys, on=["user_idx", "item_idx"], how="left"
            )
            leak_mask = merged["_leak"].notna().to_numpy()
            n_train_leaked = int(leak_mask.sum())
            if n_train_leaked > 0:
                logger.warning(
                    "De-leaked %d train purchase rows whose (user_idx, item_idx) "
                    "pair also appears in the test set (temporal train/test overlap).",
                    n_train_leaked,
                )
                train = train.loc[~leak_mask].reset_index(drop=True)

        # After de-leakage some users may have lost ALL their training interactions
        # (i.e. their only purchase pair was the same as their test pair and got
        # removed above). Their GNN embeddings would be random noise, so drop
        # them from val/test now — before artifacts are written to disk.
        if not train.empty:
            train_user_set = set(train["user_idx"].unique().tolist())
            n_val_before = len(val)
            n_test_before = len(test)
            if not val.empty:
                val = val.loc[val["user_idx"].isin(train_user_set)].reset_index(drop=True)
            if not test.empty:
                test = test.loc[test["user_idx"].isin(train_user_set)].reset_index(drop=True)
            n_val_dropped  = n_val_before  - len(val)
            n_test_dropped = n_test_before - len(test)
            if n_val_dropped > 0 or n_test_dropped > 0:
                logger.warning(
                    "Post-de-leakage: dropped %d val and %d test rows for users "
                    "who lost all training interactions after de-leakage.",
                    n_val_dropped, n_test_dropped,
                )

        # Count cold-start items retained in val/test (in vocab but not in training
        # purchases). These are handled by BAGNN via structural graph edges.
        train_item_set = set(train["item_idx"].unique().tolist()) if not train.empty else set()
        val_cold_items  = int((~val["item_idx"].isin(train_item_set)).sum())  if not val.empty  else 0
        test_cold_items = int((~test["item_idx"].isin(train_item_set)).sum()) if not test.empty else 0

        if val_cold_items > 0 or test_cold_items > 0:
            logger.info(
                "Cold-start items retained: val=%d  test=%d  "
                "(embeddings via structural graph).",
                val_cold_items, test_cold_items,
            )

        stats.update({
            "val_cold_users_dropped":   val_n_in - len(val),
            "test_cold_users_dropped":  test_n_in - len(test),
            "val_cold_items_retained":  val_cold_items,
            "test_cold_items_retained": test_cold_items,
            "val_rows_final":           len(val),
            "test_rows_final":          len(test),
            "train_leaked_pairs_removed": n_train_leaked,
        })

        return SplitResult(
            train=train,
            val=val,
            test=test,
            user2idx=user2idx,
            item2idx=item2idx,
            num_users=num_users,
            num_items=num_items,
            stats=stats,
            train_end_ts=train_end_ts,
        )
