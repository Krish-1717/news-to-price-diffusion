"""
data/news_cache.py -- SQLite-backed news cache with TTL and WAL mode
Day 10 Commit 2: Thread-safe cache, context manager, TTL-based expiry.
"""
from __future__ import annotations
import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional


_DDL = """
CREATE TABLE IF NOT EXISTS raw_news (
    id         TEXT PRIMARY KEY,
    ticker     TEXT NOT NULL,
    headline   TEXT NOT NULL,
    body       TEXT,
    source     TEXT,
    published  REAL NOT NULL,
    fetched_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_raw_news_ticker ON raw_news(ticker);
CREATE INDEX IF NOT EXISTS idx_raw_news_published ON raw_news(published);

CREATE TABLE IF NOT EXISTS sentiment_scores (
    news_id    TEXT NOT NULL REFERENCES raw_news(id) ON DELETE CASCADE,
    model      TEXT NOT NULL,
    score      REAL NOT NULL,
    confidence REAL,
    scored_at  REAL NOT NULL,
    PRIMARY KEY (news_id, model)
);
"""


@dataclass
class NewsItem:
    id: str
    ticker: str
    headline: str
    published: float
    body: Optional[str] = None
    source: Optional[str] = None
    fetched_at: float = 0.0

    def age_hours(self) -> float:
        return (time.time() - self.published) / 3600.0


@dataclass
class SentimentScore:
    news_id: str
    model: str
    score: float
    confidence: Optional[float] = None
    scored_at: float = 0.0


class NewsCache:
    """SQLite-backed news cache with WAL mode and TTL eviction."""

    def __init__(self, db_path: str | Path, ttl_hours: float = 72.0):
        self.db_path = Path(db_path)
        self.ttl_hours = ttl_hours
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.executescript(_DDL)

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ââ Write âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

    def upsert_news(self, items: List[NewsItem]) -> int:
        """Insert or replace news items. Returns count written."""
        if not items:
            return 0
        with self.transaction() as conn:
            conn.executemany(
                """INSERT OR REPLACE INTO raw_news
                   (id, ticker, headline, body, source, published, fetched_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                [
                    (i.id, i.ticker, i.headline, i.body, i.source,
                     i.published, i.fetched_at or time.time())
                    for i in items
                ],
            )
        return len(items)

    def upsert_scores(self, scores: List[SentimentScore]) -> int:
        if not scores:
            return 0
        with self.transaction() as conn:
            conn.executemany(
                """INSERT OR REPLACE INTO sentiment_scores
                   (news_id, model, score, confidence, scored_at)
                   VALUES (?, ?, ?, ?, ?)""",
                [
                    (s.news_id, s.model, s.score, s.confidence,
                     s.scored_at or time.time())
                    for s in scores
                ],
            )
        return len(scores)

    # ââ Read ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

    def get_news(self, ticker: str, since: float = 0.0,
                 limit: int = 500) -> List[NewsItem]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM raw_news
                   WHERE ticker=? AND published>=?
                   ORDER BY published DESC LIMIT ?""",
                (ticker, since, limit),
            ).fetchall()
        return [NewsItem(**dict(r)) for r in rows]

    def get_scores(self, news_ids: List[str],
                   model: Optional[str] = None) -> Dict[str, SentimentScore]:
        if not news_ids:
            return {}
        ph = ",".join("?" * len(news_ids))
        params: list = list(news_ids)
        sql = f"SELECT * FROM sentiment_scores WHERE news_id IN ({ph})"
        if model:
            sql += " AND model=?"
            params.append(model)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return {r["news_id"]: SentimentScore(**dict(r)) for r in rows}

    # ââ Maintenance âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

    def evict_expired(self) -> int:
        cutoff = time.time() - self.ttl_hours * 3600
        with self.transaction() as conn:
            cur = conn.execute(
                "DELETE FROM raw_news WHERE fetched_at < ?", (cutoff,)
            )
        return cur.rowcount

    def stats(self) -> dict:
        with self._connect() as conn:
            n_news = conn.execute("SELECT COUNT(*) FROM raw_news").fetchone()[0]
            n_scores = conn.execute(
                "SELECT COUNT(*) FROM sentiment_scores"
            ).fetchone()[0]
            oldest = conn.execute(
                "SELECT MIN(published) FROM raw_news"
            ).fetchone()[0]
        return {"n_news": n_news, "n_scores": n_scores, "oldest_published": oldest}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass
