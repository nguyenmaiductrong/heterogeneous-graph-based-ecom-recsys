from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .db import connect, db_path, init_db
from .model_inference import DEFAULT_CHECKPOINT, load_embedding_model, model_available
from .recommender import recommend


WEB_DIR = Path(__file__).resolve().parents[1] / "web"

app = FastAPI(title="BPATMP Demo Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class UserIn(BaseModel):
    id: Optional[str] = None
    name: str = Field(min_length=1, max_length=80)
    notes: str = ""
    active: bool = True


class EventIn(BaseModel):
    id: Optional[str] = None
    userId: str
    productId: str
    behavior: str
    timestamp: Optional[str] = None
    sessionId: str = ""
    source: str = "ui"


class RecommendationIn(BaseModel):
    userId: str
    topK: int = 30
    maskPurchased: bool = True
    weights: Dict[str, float] = Field(default_factory=lambda: {"view": 1.0, "cart": 3.0, "purchase": 5.0})
    recencyDecay: float = 0.055
    categoryWeight: float = 1.9
    brandWeight: float = 1.35
    popularityWeight: float = 0.85
    refTime: Optional[str] = None


def _clamp(value: float, low: float, high: float, fallback: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return min(high, max(low, number))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _user(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "name": row["name"],
        "notes": row["notes"],
        "active": bool(row["active"]),
    }


def _product(row: Any) -> dict[str, Any]:
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


def _event(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "userId": row["user_id"],
        "productId": row["product_id"],
        "behavior": row["behavior"],
        "timestamp": row["timestamp"],
        "sessionId": row["session_id"],
    }


def _bootstrap_payload() -> dict[str, Any]:
    with connect() as conn:
        init_db(conn)
        users = [_user(row) for row in conn.execute("SELECT * FROM users ORDER BY id")]
        products = [_product(row) for row in conn.execute("SELECT * FROM products ORDER BY product_idx")]
        rows = conn.execute(
            """
            SELECT * FROM events
            ORDER BY user_id, timestamp
            """
        ).fetchall()

    sessions_by_user: dict[str, dict[str, Any]] = {}
    for row in rows:
        user_id = row["user_id"]
        session_id = row["session_id"] or f"seed-{user_id}"
        key = f"{user_id}:{session_id}"
        if key not in sessions_by_user:
            sessions_by_user[key] = {
                "id": session_id,
                "label": f"Log backend {user_id}",
                "userId": user_id,
                "createdAt": row["timestamp"],
                "events": [],
            }
        sessions_by_user[key]["events"].append(_event(row))

    return {
        "products": products,
        "users": users,
        "eventSessions": list(sessions_by_user.values()),
        "simulatorPresets": [
            {
                "id": "preset-default",
                "name": "Can bang",
                "weights": {"view": 1, "cart": 3, "purchase": 5},
                "recencyDecay": 0.055,
                "categoryWeight": 1.9,
                "brandWeight": 1.35,
                "popularityWeight": 0.85,
                "topK": 30,
                "maskPurchased": True,
            }
        ],
        "activePresetId": "preset-default",
        "signedInUserId": None,
        "draftSession": None,
    }


@app.get("/api/health")
def health() -> dict[str, Any]:
    path = db_path()
    with connect(path) as conn:
        init_db(conn)
        users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        products = conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]
        events = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    loaded = False
    model_info: dict[str, Any] = {
        "checkpoint": str(DEFAULT_CHECKPOINT),
        "available": model_available(),
    }
    if model_info["available"]:
        try:
            model = load_embedding_model()
            loaded = True
            model_info.update(
                {
                    "checkpoint": str(model.checkpoint_path),
                    "epoch": model.epoch,
                    "users": model.n_users,
                    "products": model.n_products,
                    "metrics": model.metrics,
                }
            )
        except Exception as exc:
            model_info["error"] = str(exc)
    return {
        "ok": True,
        "dbPath": str(path),
        "users": users,
        "products": products,
        "events": events,
        "modelLoaded": loaded,
        "engine": "bpatmp-checkpoint-embedding",
        "model": model_info,
    }


@app.get("/api/bootstrap")
def bootstrap() -> dict[str, Any]:
    return _bootstrap_payload()


@app.get("/api/users")
def users() -> list[dict[str, Any]]:
    with connect() as conn:
        init_db(conn)
        return [_user(row) for row in conn.execute("SELECT * FROM users ORDER BY id")]


@app.post("/api/users")
def create_user(payload: UserIn) -> dict[str, Any]:
    user_id = payload.id or f"user-{uuid.uuid4()}"
    with connect() as conn:
        init_db(conn)
        conn.execute(
            """
            INSERT INTO users(id, name, notes, active)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name=excluded.name,
                notes=excluded.notes,
                active=excluded.active
            """,
            (user_id, payload.name, payload.notes, int(payload.active)),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return _user(row)


@app.get("/api/products")
def products() -> list[dict[str, Any]]:
    with connect() as conn:
        init_db(conn)
        return [_product(row) for row in conn.execute("SELECT * FROM products ORDER BY product_idx")]


@app.post("/api/events")
def create_event(payload: EventIn) -> dict[str, Any]:
    if payload.behavior not in {"view", "cart", "purchase"}:
        raise HTTPException(status_code=400, detail="behavior must be view, cart, or purchase")
    event_id = payload.id or f"event-{uuid.uuid4()}"
    timestamp = payload.timestamp or _now()
    with connect() as conn:
        init_db(conn)
        if not conn.execute("SELECT 1 FROM users WHERE id = ?", (payload.userId,)).fetchone():
            conn.execute(
                """
                INSERT INTO users(id, name, notes, active)
                VALUES (?, ?, ?, 1)
                """,
                (payload.userId, payload.userId, "Created automatically from an event."),
            )
        if not conn.execute("SELECT 1 FROM products WHERE id = ?", (payload.productId,)).fetchone():
            raise HTTPException(status_code=404, detail="product not found")
        conn.execute(
            """
            INSERT OR IGNORE INTO events(id, user_id, product_id, behavior, timestamp, session_id, source)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (event_id, payload.userId, payload.productId, payload.behavior, timestamp, payload.sessionId, payload.source),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
    return _event(row)


@app.post("/api/recommendations")
def recommendations(payload: RecommendationIn) -> dict[str, Any]:
    with connect() as conn:
        init_db(conn)
        if not conn.execute("SELECT 1 FROM users WHERE id = ?", (payload.userId,)).fetchone():
            raise HTTPException(status_code=404, detail="user not found")
        return recommend(
            conn,
            payload.userId,
            top_k=int(_clamp(payload.topK, 1, 50, 30)),
            mask_purchased=payload.maskPurchased,
            behavior_weights={
                "view": _clamp(payload.weights.get("view", 1.0), 0, 6, 1.0),
                "cart": _clamp(payload.weights.get("cart", 3.0), 0, 6, 3.0),
                "purchase": _clamp(payload.weights.get("purchase", 5.0), 0, 6, 5.0),
            },
            recency_decay=_clamp(payload.recencyDecay, 0, 0.2, 0.055),
            category_weight=_clamp(payload.categoryWeight, 0, 4, 1.9),
            brand_weight=_clamp(payload.brandWeight, 0, 4, 1.35),
            popularity_weight=_clamp(payload.popularityWeight, 0, 2, 0.85),
            ref_time=payload.refTime,
        )


if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
