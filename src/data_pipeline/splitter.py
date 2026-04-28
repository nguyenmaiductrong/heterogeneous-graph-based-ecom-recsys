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
    train_end_ts: int | None = None
    val_end_ts: int | None = None
    protocol_name: str = "warm_new_purchase_full_ranking"
    candidate_item_idx: np.ndarray | None = None

    def summary(self) -> str:
        lines = [
            f"protocol               : {self.protocol_name}",
            f"num_users              : {self.num_users:>10,}",
            f"num_items              : {self.num_items:>10,}",
            f"train rows             : {len(self.train):>10,}",
            f"val   rows             : {len(self.val):>10,}",
            f"test  rows             : {len(self.test):>10,}",
        ]
        if self.train_end_ts is not None:
            lines.append(f"train_end_ts           : {self.train_end_ts:>10,}")
        if self.val_end_ts is not None:
            lines.append(f"val_end_ts             : {self.val_end_ts:>10,}")
        for k, v in self.stats.items():
            lines.append(f"{k:<30s}: {v}")
        return "\n".join(lines)


class DataSplitter:
    """Warm-new-purchase temporal split.

    Primary protocol:
      - user2idx, item2idx are built from the train window only.
      - val/test rows with cold users or cold items are dropped.
      - val/test (user, item) pairs that already appear in train purchases are
        dropped from the eval ground truth (new-purchase target).
      - Train rows are never mutated using val/test membership.
      - Multi-positive ground truth is preserved.
    """

    _REQUIRED = {"user_id", "item_id", "timestamp"}

    def __init__(self, df: pd.DataFrame) -> None:
        missing = self._REQUIRED - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame is missing required columns: {missing}")
        self._df = df.copy()
        self._has_event_type = "event_type" in df.columns

    def temporal_split_by_dates(
        self,
        train_end: str,
        val_end: str,
        *,
        transductive_item_vocab: bool = False,
        drop_repeated_train_purchases_from_eval: bool = True,
        protocol_name: str = "warm_new_purchase_full_ranking",
    ) -> SplitResult:
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
            "split_protocol":           protocol_name,
            "train_end_date":           train_end,
            "val_end_date":             val_end,
            "train_cutoff_ts":          train_end_ts,
            "val_cutoff_ts":            val_end_ts,
            "transductive_item_vocab":  transductive_item_vocab,
            "raw_rows_train":           len(train_raw),
            "raw_rows_val":             len(val_raw),
            "raw_rows_test":            len(test_raw),
            "train_pct":                f"{len(train_raw) / max(n, 1) * 100:.1f}%",
            "val_pct":                  f"{len(val_raw)   / max(n, 1) * 100:.1f}%",
            "test_pct":                 f"{len(test_raw)  / max(n, 1) * 100:.1f}%",
        }
        return self._finalize(
            train_raw, val_raw, test_raw, stats,
            train_end_ts=train_end_ts,
            val_end_ts=val_end_ts,
            transductive_item_vocab=transductive_item_vocab,
            drop_repeated_train_purchases_from_eval=drop_repeated_train_purchases_from_eval,
            protocol_name=protocol_name,
        )

    def _build_mappings(
        self,
        train_df: pd.DataFrame,
        all_df: pd.DataFrame | None = None,
        transductive_item_vocab: bool = False,
    ) -> tuple[dict, dict]:
        user2idx = {u: i for i, u in enumerate(sorted(train_df["user_id"].unique()))}
        if transductive_item_vocab and all_df is not None:
            item2idx = {it: i for i, it in enumerate(sorted(all_df["item_id"].unique()))}
        else:
            item2idx = {it: i for i, it in enumerate(sorted(train_df["item_id"].unique()))}
        return user2idx, item2idx

    def _apply_mapping(
        self,
        df: pd.DataFrame,
        user2idx: dict,
        item2idx: dict,
        drop_unknown: bool,
    ) -> tuple[pd.DataFrame, int, int]:
        out = df.copy()
        cold_users_dropped = 0
        cold_items_dropped = 0

        if drop_unknown:
            user_mask = out["user_id"].isin(user2idx)
            cold_users_dropped = int((~user_mask).sum())
            out = out.loc[user_mask].copy()

            item_mask = out["item_id"].isin(item2idx)
            cold_items_dropped = int((~item_mask).sum())
            out = out.loc[item_mask].copy()

        out["user_idx"] = out["user_id"].map(user2idx)
        out["item_idx"] = out["item_id"].map(item2idx)

        if not out.empty and (out["user_idx"].isna().any() or out["item_idx"].isna().any()):
            raise RuntimeError(
                "NaN in mapped indices after vocabulary lookup — this is a bug."
            )

        if not out.empty:
            out["user_idx"] = out["user_idx"].astype(np.int64)
            out["item_idx"] = out["item_idx"].astype(np.int64)
        else:
            out["user_idx"] = pd.Series([], dtype=np.int64)
            out["item_idx"] = pd.Series([], dtype=np.int64)

        keep = ["user_idx", "item_idx", "timestamp"]
        if self._has_event_type:
            keep.append("event_type")
        return out[keep].reset_index(drop=True), cold_users_dropped, cold_items_dropped

    def _finalize(
        self,
        train_raw: pd.DataFrame,
        val_raw: pd.DataFrame,
        test_raw: pd.DataFrame,
        stats: dict,
        *,
        train_end_ts: int,
        val_end_ts: int,
        transductive_item_vocab: bool,
        drop_repeated_train_purchases_from_eval: bool,
        protocol_name: str,
    ) -> SplitResult:
        all_df = self._df if transductive_item_vocab else None
        user2idx, item2idx = self._build_mappings(
            train_raw, all_df=all_df,
            transductive_item_vocab=transductive_item_vocab,
        )

        train, _, _ = self._apply_mapping(train_raw, user2idx, item2idx, drop_unknown=False)

        val_n_in = len(val_raw)
        val, val_cold_users, val_cold_items = self._apply_mapping(
            val_raw, user2idx, item2idx, drop_unknown=True
        )

        test_n_in = len(test_raw)
        test, test_cold_users, test_cold_items = self._apply_mapping(
            test_raw, user2idx, item2idx, drop_unknown=True
        )

        num_users = len(user2idx)
        num_items = len(item2idx)

        for name, split_df in (("train", train), ("val", val), ("test", test)):
            if split_df.empty:
                continue
            if int(split_df["user_idx"].max()) >= num_users:
                raise RuntimeError(
                    f"{name} user_idx out of range: max={int(split_df['user_idx'].max())} num_users={num_users}"
                )
            if int(split_df["item_idx"].max()) >= num_items:
                raise RuntimeError(
                    f"{name} item_idx out of range: max={int(split_df['item_idx'].max())} num_items={num_items}"
                )
            if int(split_df["user_idx"].min()) < 0 or int(split_df["item_idx"].min()) < 0:
                raise RuntimeError(f"{name} contains a negative index.")

        val_repeated_dropped = 0
        test_repeated_dropped = 0
        if drop_repeated_train_purchases_from_eval and not train.empty:
            train_pairs = set(
                zip(train["user_idx"].tolist(), train["item_idx"].tolist())
            )
            if not val.empty:
                val_keys = list(zip(val["user_idx"].tolist(), val["item_idx"].tolist()))
                val_keep = np.array([k not in train_pairs for k in val_keys])
                val_repeated_dropped = int((~val_keep).sum())
                val = val.loc[val_keep].reset_index(drop=True)
            if not test.empty:
                test_keys = list(zip(test["user_idx"].tolist(), test["item_idx"].tolist()))
                test_keep = np.array([k not in train_pairs for k in test_keys])
                test_repeated_dropped = int((~test_keep).sum())
                test = test.loc[test_keep].reset_index(drop=True)
            if val_repeated_dropped or test_repeated_dropped:
                logger.info(
                    "Dropped repeated train-purchase pairs from eval: val=%d test=%d",
                    val_repeated_dropped, test_repeated_dropped,
                )

        train_item_set = set(train["item_idx"].unique().tolist()) if not train.empty else set()
        val_warm_cold_items  = int((~val["item_idx"].isin(train_item_set)).sum())  if not val.empty  else 0
        test_warm_cold_items = int((~test["item_idx"].isin(train_item_set)).sum()) if not test.empty else 0

        candidate_item_idx = np.arange(num_items, dtype=np.int64)

        stats.update({
            "val_cold_users_dropped":            val_cold_users,
            "val_cold_items_dropped":            val_cold_items,
            "test_cold_users_dropped":           test_cold_users,
            "test_cold_items_dropped":           test_cold_items,
            "val_repeated_train_purchase_dropped":  val_repeated_dropped,
            "test_repeated_train_purchase_dropped": test_repeated_dropped,
            "val_rows_in":                       val_n_in,
            "test_rows_in":                      test_n_in,
            "val_rows_final":                    len(val),
            "test_rows_final":                   len(test),
            "val_items_unseen_in_train":         val_warm_cold_items,
            "test_items_unseen_in_train":        test_warm_cold_items,
            "num_users_train_vocab":             num_users,
            "num_items_train_vocab":             num_items,
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
            val_end_ts=val_end_ts,
            protocol_name=protocol_name,
            candidate_item_idx=candidate_item_idx,
        )
