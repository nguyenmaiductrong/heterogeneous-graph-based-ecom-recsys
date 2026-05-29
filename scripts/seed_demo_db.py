#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backend.db import DEFAULT_DB_PATH, connect, init_db


CATEGORIES = [
    "Audio",
    "Computer accessories",
    "Home office",
    "Mobile devices",
    "Wearables",
    "Camera",
    "Gaming",
    "Smart home",
]
BRANDS = [
    "Sony",
    "Logitech",
    "Apple",
    "Samsung",
    "LG",
    "Keychron",
    "Fitbit",
    "Ergo",
    "Anker",
    "Dell",
    "Asus",
    "Xiaomi",
]
IMAGES = ["headset", "keyboard", "monitor", "mouse", "speaker", "chair", "watch", "tablet", "camera"]


def stable_int(value: object) -> int:
    digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def product_payload(product_idx: int, popularity_rank: int, total: int) -> tuple:
    h = stable_int(product_idx)
    category = CATEGORIES[h % len(CATEGORIES)]
    brand = BRANDS[(h // 7) % len(BRANDS)]
    image = IMAGES[(h // 13) % len(IMAGES)]
    price = 199000 + (h % 180) * 100000
    popularity = max(8, round(95 - popularity_rank * (75 / max(total, 1)), 2))
    return (
        f"p-{product_idx}",
        int(product_idx),
        f"{category} product {product_idx}",
        brand,
        category,
        int(price),
        float(popularity),
        image,
        1,
    )


def reset_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        DELETE FROM recommendation_logs;
        DELETE FROM events;
        DELETE FROM products;
        DELETE FROM users;
        """
    )
    conn.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed a small SQLite DB for the web recommender demo.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--users", type=int, default=30)
    parser.add_argument("--products", type=int, default=300)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    user_idx = np.load(data_dir / "test_user_idx.npy")
    product_idx = np.load(data_dir / "test_product_idx.npy")
    timestamps = np.load(data_dir / "test_timestamp.npy")

    user_counts = Counter(map(int, user_idx.tolist()))
    product_counts = Counter(map(int, product_idx.tolist()))
    selected_users = [u for u, _ in user_counts.most_common(args.users)]

    selected_products = []
    seen = set()
    for u, p in zip(user_idx.tolist(), product_idx.tolist()):
        if len(selected_products) >= args.products:
            break
        if int(u) in selected_users and int(p) not in seen:
            selected_products.append(int(p))
            seen.add(int(p))
    for p, _ in product_counts.most_common():
        if len(selected_products) >= args.products:
            break
        if int(p) not in seen:
            selected_products.append(int(p))
            seen.add(int(p))

    selected_product_set = set(selected_products)
    events_by_user: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for u, p, ts in zip(user_idx.tolist(), product_idx.tolist(), timestamps.tolist()):
        u_int = int(u)
        p_int = int(p)
        if u_int in selected_users and p_int in selected_product_set:
            events_by_user[u_int].append((p_int, int(ts)))

    db_file = Path(args.db)
    with connect(db_file) as conn:
        init_db(conn)
        if args.reset:
            reset_tables(conn)

        conn.executemany(
            """
            INSERT OR REPLACE INTO users(id, name, notes, active)
            VALUES (?, ?, ?, ?)
            """,
            [
                (
                    f"u-{u}",
                    f"Demo user {u}",
                    "Seeded from REES46 test interactions.",
                    1,
                )
                for u in selected_users
            ],
        )

        ranked_products = {p: rank for rank, p in enumerate(selected_products)}
        conn.executemany(
            """
            INSERT OR REPLACE INTO products(
                id, product_idx, name, brand, category, price, popularity_score, image, active
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                product_payload(p, ranked_products[p], len(selected_products))
                for p in selected_products
            ],
        )

        behavior_cycle = ["view", "view", "cart", "purchase"]
        event_rows = []
        for user in selected_users:
            rows = sorted(events_by_user[user], key=lambda item: item[1])[-18:]
            for pos, (product, ts) in enumerate(rows):
                behavior = behavior_cycle[(stable_int(f"{user}:{product}:{pos}") + pos) % len(behavior_cycle)]
                event_rows.append(
                    (
                        f"seed-{user}-{product}-{ts}-{pos}",
                        f"u-{user}",
                        f"p-{product}",
                        behavior,
                        datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                        f"seed-session-{user}",
                        "seed",
                    )
                )
        conn.executemany(
            """
            INSERT OR IGNORE INTO events(id, user_id, product_id, behavior, timestamp, session_id, source)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            event_rows,
        )
        conn.commit()

        n_users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        n_products = conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]
        n_events = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]

    print(f"Seeded {db_file}")
    print(f"users={n_users} products={n_products} events={n_events}")


if __name__ == "__main__":
    main()
