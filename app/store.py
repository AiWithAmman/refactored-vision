"""Tiny SQLite memory: what Lumen has seen and where."""

import os
import sqlite3
import time
import uuid
from datetime import datetime
from typing import List

DB_PATH = os.getenv("LUMEN_DB", "data/lumen.db")
IMG_DIR = os.path.join(os.path.dirname(DB_PATH) or ".", "images")


def _conn() -> sqlite3.Connection:
    parent = os.path.dirname(DB_PATH)
    if parent:
        os.makedirs(parent, exist_ok=True)
    os.makedirs(IMG_DIR, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            label TEXT,
            place TEXT,
            image_path TEXT,
            created_at REAL
        )
        """
    )
    return conn


def add_item(label: str, place: str, image: bytes) -> int:
    path = os.path.join(IMG_DIR, f"{uuid.uuid4().hex}.jpg")
    with open(path, "wb") as f:
        f.write(image)

    with _conn() as conn:
        cur = conn.execute(
            "INSERT INTO items (label, place, image_path, created_at) VALUES (?, ?, ?, ?)",
            (label, place, path, time.time()),
        )
        return cur.lastrowid


def list_items() -> List[dict]:
    """Newest first. 'remembered' is a readable timestamp the model can reason over."""
    with _conn() as conn:
        rows = conn.execute("SELECT * FROM items ORDER BY created_at DESC").fetchall()

    return [
        {
            "id": r["id"],
            "label": r["label"],
            "place": r["place"],
            "remembered": datetime.fromtimestamp(r["created_at"]).strftime("%Y-%m-%d %H:%M"),
        }
        for r in rows
    ]
