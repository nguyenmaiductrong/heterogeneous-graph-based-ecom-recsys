from __future__ import annotations

import os
import pickle

import numpy as np
import pandas as pd
import pytest

from src.data_pipeline.extract import _VALID_BEHAVIORS, clean
from src.data_pipeline.splitter import DataSplitter
from src.data_pipeline.transform import (
    build_train_mask,
    filter_by_train_only_counts,
    map_all_train_events,
    map_auxiliary,
)
from src.data_pipeline.load import _build_ground_truth, save_eval_split
from src.data_pipeline.sanity import (
    sanity_check_eval_mask,
    sanity_check_ground_truth,
    sanity_check_temporal_artifacts,
)


TRAIN_END = "2020-02-29"
VAL_END   = "2020-03-31"
TRAIN_CUTOFF_TS = int((pd.Timestamp(TRAIN_END, tz="UTC") + pd.Timedelta(days=1)).timestamp())
VAL_CUTOFF_TS   = int((pd.Timestamp(VAL_END,   tz="UTC") + pd.Timedelta(days=1)).timestamp())


def _ts(date_str: str) -> int:
    return int(pd.Timestamp(date_str, tz="UTC").timestamp())


def _synthetic_purchase_df() -> pd.DataFrame:
    rows = [
        # train window
        {"user_id": 1, "item_id": 100, "timestamp": _ts("2020-01-15"), "event_type": "purchase"},
        {"user_id": 1, "item_id": 101, "timestamp": _ts("2020-01-16"), "event_type": "purchase"},
        {"user_id": 1, "item_id": 100, "timestamp": _ts("2020-01-17"), "event_type": "purchase"},
        {"user_id": 1, "item_id": 103, "timestamp": _ts("2020-01-25"), "event_type": "purchase"},
        {"user_id": 2, "item_id": 102, "timestamp": _ts("2020-01-18"), "event_type": "purchase"},
        {"user_id": 2, "item_id": 100, "timestamp": _ts("2020-01-19"), "event_type": "purchase"},
        {"user_id": 3, "item_id": 101, "timestamp": _ts("2020-02-10"), "event_type": "purchase"},
        # val window
        {"user_id": 1, "item_id": 102, "timestamp": _ts("2020-03-05"), "event_type": "purchase"},
        {"user_id": 1, "item_id": 100, "timestamp": _ts("2020-03-06"), "event_type": "purchase"},
        {"user_id": 4, "item_id": 100, "timestamp": _ts("2020-03-07"), "event_type": "purchase"},
        # test window
        {"user_id": 2, "item_id": 999, "timestamp": _ts("2020-04-05"), "event_type": "purchase"},
        {"user_id": 2, "item_id": 100, "timestamp": _ts("2020-04-06"), "event_type": "purchase"},
        {"user_id": 2, "item_id": 101, "timestamp": _ts("2020-04-07"), "event_type": "purchase"},
        {"user_id": 2, "item_id": 103, "timestamp": _ts("2020-04-09"), "event_type": "purchase"},
        {"user_id": 3, "item_id": 102, "timestamp": _ts("2020-04-08"), "event_type": "purchase"},
    ]
    return pd.DataFrame(rows)


def test_extract_valid_behaviors_constant():
    assert _VALID_BEHAVIORS == ("view", "cart", "purchase")
    assert "remove_from_cart" not in _VALID_BEHAVIORS


def test_split_train_only_user_vocab_excludes_cold_user():
    df = _synthetic_purchase_df()
    split = DataSplitter(df).temporal_split_by_dates(
        train_end=TRAIN_END, val_end=VAL_END,
    )
    assert 4 not in split.user2idx, (
        "user 4 has no train interactions and must be excluded from primary vocab"
    )
    assert 1 in split.user2idx
    assert 2 in split.user2idx
    assert 3 in split.user2idx


def test_split_train_only_item_vocab_excludes_future_only():
    df = _synthetic_purchase_df()
    split = DataSplitter(df).temporal_split_by_dates(
        train_end=TRAIN_END, val_end=VAL_END,
    )
    assert 999 not in split.item2idx
    assert 100 in split.item2idx
    assert 101 in split.item2idx
    assert 102 in split.item2idx


def test_split_drops_repeated_train_purchases_from_eval():
    df = _synthetic_purchase_df()
    split = DataSplitter(df).temporal_split_by_dates(
        train_end=TRAIN_END, val_end=VAL_END,
    )
    val_pairs  = set(zip(split.val["user_idx"].tolist(),  split.val["item_idx"].tolist()))
    test_pairs = set(zip(split.test["user_idx"].tolist(), split.test["item_idx"].tolist()))
    train_pairs = set(zip(split.train["user_idx"].tolist(), split.train["item_idx"].tolist()))

    assert val_pairs.isdisjoint(train_pairs)
    assert test_pairs.isdisjoint(train_pairs)
    assert split.stats["val_repeated_train_purchase_dropped"]  >= 1
    assert split.stats["test_repeated_train_purchase_dropped"] >= 1


def test_split_does_not_mutate_train_using_test():
    df = _synthetic_purchase_df()
    split = DataSplitter(df).temporal_split_by_dates(
        train_end=TRAIN_END, val_end=VAL_END,
    )
    train_pairs = set(zip(split.train["user_idx"].tolist(), split.train["item_idx"].tolist()))
    u1 = split.user2idx[1]
    i100 = split.item2idx[100]
    assert (u1, i100) in train_pairs
    u2 = split.user2idx[2]
    assert (u2, i100) in train_pairs


def test_split_multi_positive_ground_truth_preserved():
    df = _synthetic_purchase_df()
    split = DataSplitter(df).temporal_split_by_dates(
        train_end=TRAIN_END, val_end=VAL_END,
    )
    u2 = split.user2idx[2]
    test_for_u2 = split.test.loc[split.test["user_idx"] == u2]
    assert len(test_for_u2) == 2, (
        "u2 keeps test purchases for items 101 and 103 after dropping cold 999 and repeated (2,100)"
    )
    gt = _build_ground_truth(split.test)
    assert sorted(gt[u2]) == sorted([split.item2idx[101], split.item2idx[103]])


def test_save_eval_split_writes_timestamps_and_multi_positive(tmp_path):
    df = _synthetic_purchase_df()
    split = DataSplitter(df).temporal_split_by_dates(
        train_end=TRAIN_END, val_end=VAL_END,
    )
    data_dir  = tmp_path / "data"
    graph_dir = tmp_path / "graph"
    data_dir.mkdir()
    graph_dir.mkdir()

    test_gt = save_eval_split(split.test, str(data_dir), str(graph_dir), "test")

    test_ts = np.load(data_dir / "test_timestamp.npy")
    test_user = np.load(data_dir / "test_user_idx.npy")
    test_item = np.load(data_dir / "test_product_idx.npy")
    assert test_ts.dtype == np.int64
    assert len(test_ts) == len(test_user) == len(test_item) == len(split.test)

    pq_df = pd.read_parquet(graph_dir / "test_ground_truth.parquet")
    assert len(pq_df) == len(split.test)

    sanity_check_ground_truth(test_gt, pq_df, split_name="test")


def test_temporal_artifacts_check_catches_future_train_edge():
    behavior_artifacts = {
        "view":     {"src": np.array([0], np.int64), "dst": np.array([0], np.int64), "ts": np.array([1], np.int64)},
        "cart":     {"src": np.array([0], np.int64), "dst": np.array([0], np.int64), "ts": np.array([1], np.int64)},
        "purchase": {"src": np.array([0], np.int64), "dst": np.array([0], np.int64),
                     "ts": np.array([TRAIN_CUTOFF_TS + 1], np.int64)},
    }
    val_ts  = np.array([TRAIN_CUTOFF_TS + 100], np.int64)
    test_ts = np.array([VAL_CUTOFF_TS   + 100], np.int64)
    with pytest.raises(AssertionError):
        sanity_check_temporal_artifacts(
            behavior_artifacts, val_ts, test_ts,
            TRAIN_CUTOFF_TS, VAL_CUTOFF_TS,
        )


def test_temporal_artifacts_check_passes_for_well_formed_arrays():
    behavior_artifacts = {
        "view":     {"src": np.array([0, 1], np.int64), "dst": np.array([0, 1], np.int64),
                     "ts": np.array([10, 20], np.int64)},
        "cart":     {"src": np.array([0],    np.int64), "dst": np.array([0],    np.int64),
                     "ts": np.array([30],    np.int64)},
        "purchase": {"src": np.array([0],    np.int64), "dst": np.array([0],    np.int64),
                     "ts": np.array([40],    np.int64)},
    }
    val_ts  = np.array([TRAIN_CUTOFF_TS + 100], np.int64)
    test_ts = np.array([VAL_CUTOFF_TS   + 100], np.int64)
    sanity_check_temporal_artifacts(
        behavior_artifacts, val_ts, test_ts,
        TRAIN_CUTOFF_TS, VAL_CUTOFF_TS,
    )


def test_eval_mask_check_rejects_positive_in_primary_mask():
    primary_mask = {1: [10]}
    gt           = {1: [10]}
    with pytest.raises(AssertionError):
        sanity_check_eval_mask(primary_mask, gt, split_name="test")


def test_clean_filters_to_3_behaviors(spark):
    rows = [
        ("u1", "i1", "view",            pd.Timestamp("2020-01-01 00:00:00", tz="UTC").to_pydatetime(), 1.0, "elec.phone", "apple", "sess1"),
        ("u2", "i2", "cart",            pd.Timestamp("2020-01-02 00:00:00", tz="UTC").to_pydatetime(), 2.0, "elec.phone", "apple", "sess2"),
        ("u3", "i3", "purchase",        pd.Timestamp("2020-01-03 00:00:00", tz="UTC").to_pydatetime(), 3.0, "elec.phone", "apple", "sess3"),
        ("u4", "i4", "remove_from_cart", pd.Timestamp("2020-01-04 00:00:00", tz="UTC").to_pydatetime(), 4.0, "elec.phone", "apple", "sess4"),
    ]
    cols = ["user_id", "product_id", "event_type", "event_time", "price", "category_code", "brand", "user_session"]
    sdf = spark.createDataFrame(rows, cols)
    sdf = sdf.withColumn("user_id",    sdf["user_id"]).withColumn(
        "product_id", sdf["product_id"])

    cfg = {
        "filter": {
            "unknown_brand": "__UNKNOWN_BRAND__",
            "unknown_category": "__UNKNOWN_CATEGORY__",
            "unknown_session": "__UNKNOWN_SESSION__",
            "category_level": "top",
        },
    }
    out = clean(sdf, cfg)
    types = {r["event_type"] for r in out.select("event_type").distinct().collect()}
    assert types == {"view", "cart", "purchase"}
    assert "user_session" in out.columns


def test_train_mask_primary_does_not_mask_view_or_cart_only(spark):
    purchase_train = pd.DataFrame({
        "user_idx":  [0, 1],
        "item_idx":  [10, 20],
        "timestamp": [1, 2],
    })
    aux_rows = [
        (0, 30, "view", 100),
        (0, 40, "cart", 101),
        (1, 50, "view", 102),
    ]
    aux_spark = spark.createDataFrame(
        aux_rows, ["user_idx", "item_idx", "event_type", "timestamp"]
    )

    primary = build_train_mask(
        purchase_train, aux_spark, [0, 1], spark,
        mask_behaviors=("purchase",),
    )
    seen_all = build_train_mask(
        purchase_train, aux_spark, [0, 1], spark,
        mask_behaviors=("view", "cart", "purchase"),
    )

    assert sorted(primary[0]) == [10]
    assert sorted(primary[1]) == [20]
    assert sorted(seen_all[0]) == [10, 30, 40]
    assert sorted(seen_all[1]) == [20, 50]


def test_filter_by_train_only_counts_uses_only_train_window(spark):
    rows = [
        (1, 100, "purchase", _ts("2020-01-01")),
        (1, 100, "purchase", _ts("2020-01-02")),
        (1, 100, "purchase", _ts("2020-01-03")),
        (2, 100, "purchase", _ts("2020-04-01")),
        (2, 100, "purchase", _ts("2020-04-02")),
        (2, 100, "purchase", _ts("2020-04-03")),
        (3, 200, "purchase", _ts("2020-01-01")),
    ]
    sdf = spark.createDataFrame(
        rows, ["user_id", "product_id", "event_type", "timestamp"]
    )
    out = filter_by_train_only_counts(
        sdf,
        target_behavior="purchase",
        train_cutoff_ts=TRAIN_CUTOFF_TS,
        min_user_purchases=2,
        min_item_purchases=2,
        rounds=2,
    )
    user_ids = {r["user_id"] for r in out.select("user_id").distinct().collect()}
    assert 1 in user_ids
    assert 2 not in user_ids
    assert 3 not in user_ids


def test_map_all_train_events_drops_post_cutoff_and_assigns_behavior_id(spark):
    df = _synthetic_purchase_df()
    purchase_df = df.copy()
    purchase_df["product_id"] = purchase_df["item_id"]
    purchase_df = purchase_df.drop(columns=["item_id"])
    sdf = spark.createDataFrame(
        purchase_df[["user_id", "product_id", "event_type", "timestamp"]]
    )

    split = DataSplitter(df).temporal_split_by_dates(
        train_end=TRAIN_END, val_end=VAL_END,
    )

    events = map_all_train_events(
        sdf, spark, split, global_cutoff=split.train_end_ts,
        behavior_ids={"view": 0, "cart": 1, "purchase": 2},
    )
    pdf = events.toPandas()
    assert (pdf["timestamp"] < split.train_end_ts).all()
    assert set(pdf["behavior_id"].unique()).issubset({0, 1, 2})
    assert set(pdf["behavior"].unique()) == {"purchase"}
