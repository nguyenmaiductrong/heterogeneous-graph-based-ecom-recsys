from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timezone
from typing import Any

import torch
import torch.nn.functional as F

from .model_inference import load_embedding_model


BEHAVIOR_WEIGHTS = {
    "view": 1.0,
    "cart": 3.0,
    "purchase": 5.0,
}


def _parse_time(value: str) -> datetime:
    try:
        normalized = value.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return datetime.now(timezone.utc)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _product_payload(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "productIdx": row["product_idx"],
        "name": row["name"],
        "brand": row["brand"],
        "category": row["category"],
        "price": row["price"],
        "popularityScore": row["popularity_score"],
        "image": row["image"],
        "active": bool(row["active"]),
    }


def recommend(
    conn: sqlite3.Connection,
    user_id: str,
    *,
    top_k: int = 5,
    mask_purchased: bool = True,
    behavior_weights: dict[str, float] | None = None,
    recency_decay: float = 0.055,
    category_weight: float = 1.9,
    brand_weight: float = 1.35,
    popularity_weight: float = 0.85,
    ref_time: str | None = None,
) -> dict[str, Any]:
    model = load_embedding_model()
    weights = {**BEHAVIOR_WEIGHTS, **(behavior_weights or {})}
    ref_dt = _parse_time(ref_time) if ref_time else None
    products = conn.execute(
        """
        SELECT * FROM products
        WHERE active = 1
        ORDER BY product_idx
        """
    ).fetchall()
    events = conn.execute(
        """
        SELECT e.*, p.name, p.brand, p.category
        FROM events e
        JOIN products p ON p.id = e.product_id
        WHERE e.user_id = ?
        ORDER BY e.timestamp DESC
        LIMIT 40
        """,
        (user_id,),
    ).fetchall()
    if ref_dt is not None:
        events = [event for event in events if _parse_time(event["timestamp"]) <= ref_dt]

    now = ref_dt or datetime.now(timezone.utc)
    if events and ref_dt is None:
        now = max(_parse_time(event["timestamp"]) for event in events)

    purchased = {event["product_id"] for event in events if event["behavior"] == "purchase"}
    ranked: list[dict[str, Any]] = []

    valid_products = [row for row in products if 0 <= int(row["product_idx"]) < model.n_products]
    candidate_indices = [int(row["product_idx"]) for row in valid_products]
    candidate_emb = model.product_vectors(candidate_indices) if candidate_indices else torch.empty(0, model.product.size(1))

    user_vec = model.user_vector(user_id)
    session_vec = None
    if events:
        weighted_vecs = []
        total_weight = 0.0
        for event in events:
            product_idx = conn.execute(
                "SELECT product_idx FROM products WHERE id = ?",
                (event["product_id"],),
            ).fetchone()
            if product_idx is None:
                continue
            idx = int(product_idx["product_idx"])
            if not (0 <= idx < model.n_products):
                continue
            event_time = _parse_time(event["timestamp"])
            age_hours = max(0.0, (now - event_time).total_seconds() / 3600.0)
            decay = math.exp(-recency_decay * age_hours)
            weight = weights.get(event["behavior"], 0.4) * decay
            if weight <= 0:
                continue
            weighted_vecs.append(model.product_vector(idx) * weight)
            total_weight += weight
        if weighted_vecs and total_weight > 0:
            session_vec = F.normalize(torch.stack(weighted_vecs).sum(dim=0) / total_weight, dim=0)

    if user_vec is not None and session_vec is not None:
        query_vec = F.normalize(user_vec * 0.7 + session_vec * 0.3, dim=0)
        query_source = "checkpoint_user_plus_session"
    elif user_vec is not None:
        query_vec = user_vec
        query_source = "checkpoint_user_embedding"
    elif session_vec is not None:
        query_vec = session_vec
        query_source = "checkpoint_item_session_embedding"
    else:
        query_vec = None
        query_source = "popularity_fallback_no_model_input"

    model_scores: dict[str, float] = {}
    raw_score_min = None
    raw_score_max = None
    if query_vec is not None and len(valid_products):
        scores = candidate_emb @ query_vec
        raw_score_min = float(scores.min().item())
        raw_score_max = float(scores.max().item())
        model_scores = {
            row["id"]: float(score)
            for row, score in zip(valid_products, scores.tolist())
        }

    for product in valid_products:
        if mask_purchased and product["id"] in purchased:
            continue

        raw_model_score = model_scores.get(product["id"], 0.0)
        model_score = (raw_model_score + 1.0) * 3.0
        contribution = {
            "model": model_score,
            "behavior": 0.0,
            "recency": 0.0,
            "category": 0.0,
            "brand": 0.0,
            "popularity": (float(product["popularity_score"]) / 100.0) * popularity_weight,
        }
        traces = []

        for event in events:
            event_time = _parse_time(event["timestamp"])
            age_hours = max(0.0, (now - event_time).total_seconds() / 3600.0)
            decay = math.exp(-recency_decay * age_hours)
            base = weights.get(event["behavior"], 0.4) * decay
            same_product = event["product_id"] == product["id"]
            same_category = event["category"] == product["category"]
            same_brand = event["brand"] == product["brand"]

            if not (same_product or same_category or same_brand):
                continue

            if same_product:
                contribution["behavior"] += base * 0.9
            contribution["recency"] += base * (0.18 if same_product else 0.08)
            if same_category:
                contribution["category"] += decay * category_weight
            if same_brand:
                contribution["brand"] += decay * brand_weight

            if len(traces) < 6:
                relation = []
                if same_product:
                    relation.append("cung san pham")
                if same_category:
                    relation.append("cung danh muc")
                if same_brand:
                    relation.append("cung thuong hieu")
                traces.append(
                    {
                        "event": {
                            "id": event["id"],
                            "userId": event["user_id"],
                            "productId": event["product_id"],
                            "behavior": event["behavior"],
                            "timestamp": event["timestamp"],
                            "sessionId": event["session_id"],
                        },
                        "origin": {
                            "id": event["product_id"],
                            "name": event["name"],
                            "brand": event["brand"],
                            "category": event["category"],
                        },
                        "ageHours": age_hours,
                        "relation": ", ".join(relation),
                    }
                )

        score = sum(contribution.values())
        ranked.append(
            {
                "product": _product_payload(product),
                "score": score,
                "contribution": contribution,
                "traces": traces,
                "noRelation": bool(events and not traces),
            }
        )

    ranked.sort(key=lambda item: item["score"], reverse=True)
    result = {
        "ranked": ranked[: max(1, min(int(top_k), 50))],
        "warnings": [],
        "source": {
            "label": "Backend SQLite",
            "events": [
                {
                    "id": event["id"],
                    "userId": event["user_id"],
                    "productId": event["product_id"],
                    "behavior": event["behavior"],
                    "timestamp": event["timestamp"],
                    "sessionId": event["session_id"],
                }
                for event in events
            ],
            "refTime": now.isoformat(),
            "querySource": query_source,
            "checkpoint": str(model.checkpoint_path),
            "checkpointEpoch": model.epoch,
            "pipeline": [
                {
                    "name": "Load checkpoint",
                    "detail": f"{model.checkpoint_path} | epoch={model.epoch}",
                },
                {
                    "name": "Read embedding tables",
                    "detail": f"user={model.n_users} x {model.user.size(1)}, product={model.n_products} x {model.product.size(1)}",
                },
                {
                    "name": "Build query embedding",
                    "detail": query_source.replace("_", " "),
                },
                {
                    "name": "Score candidates",
                    "detail": (
                        f"{len(valid_products)} active demo products by cosine similarity"
                        if raw_score_min is None
                        else f"{len(valid_products)} active demo products | raw cosine {raw_score_min:.4f}..{raw_score_max:.4f}"
                    ),
                },
                {
                    "name": "Apply business filters",
                    "detail": f"maskPurchased={mask_purchased}, hiddenPurchased={len(purchased) if mask_purchased else 0}",
                },
                {
                    "name": "Blend visible controls",
                    "detail": (
                        f"behavior={weights}, recency={recency_decay}, "
                        f"category={category_weight}, brand={brand_weight}, popularity={popularity_weight}"
                    ),
                },
                {
                    "name": "Sort top-K",
                    "detail": f"returning {max(1, min(int(top_k), 50))} of {len(ranked)} scored products",
                },
            ],
        },
    }
    if not events:
        result["warnings"].append("User chua co hanh vi; ket qua dang dua vao do pho bien.")
    if query_vec is None:
        result["warnings"].append("Khong co user embedding hoac session embedding; fallback ve popularity.")
    if mask_purchased and purchased:
        result["warnings"].append(f"Da an {len(purchased)} san pham user da mua.")

    conn.execute(
        """
        INSERT INTO recommendation_logs(user_id, top_k, payload_json)
        VALUES (?, ?, ?)
        """,
        (user_id, top_k, json.dumps(result, ensure_ascii=False)),
    )
    conn.commit()
    return result
