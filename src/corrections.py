"""
Correction feedback system for object detection.

Stores user corrections and applies them as post-processing rules
to improve detection accuracy over time without retraining the model.

Uses SQLite for persistent storage of corrections, rules, and statistics.
"""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from src.utils import ROOT

logger = __import__("logging").getLogger(__name__)

# Default corrections database path
CORRECTIONS_DB = ROOT / "data" / "corrections.db"


@dataclass
class Correction:
    """A single user correction."""
    id: Optional[int] = None
    video_id: str = ""
    frame_number: int = 0
    timestamp: float = 0.0
    original_class: str = ""
    corrected_class: str = ""
    bbox: Optional[list[float]] = None  # [x1, y1, x2, y2]
    original_confidence: float = 0.0
    action: str = "relabel"  # relabel, delete, add, ignore_class
    notes: str = ""
    created_at: float = 0.0

    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()


@dataclass
class CorrectionRule:
    """An auto-applied rule derived from corrections."""
    id: Optional[int] = None
    pattern_class: str = ""  # Original class to match
    target_class: str = ""   # What to relabel it as
    confidence_threshold: Optional[float] = None  # Only apply if confidence < this
    video_pattern: Optional[str] = None  # Apply only to videos matching this pattern
    enabled: bool = True
    usage_count: int = 0
    created_at: float = 0.0

    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()


class CorrectionStore:
    """SQLite-backed store for corrections and rules."""

    def __init__(self, db_path: Path = CORRECTIONS_DB):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Create tables if they don't exist."""
        conn = self._get_conn()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS corrections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id TEXT NOT NULL,
                    frame_number INTEGER NOT NULL,
                    timestamp REAL NOT NULL,
                    original_class TEXT NOT NULL,
                    corrected_class TEXT NOT NULL,
                    bbox TEXT,
                    original_confidence REAL NOT NULL,
                    action TEXT NOT NULL DEFAULT 'relabel',
                    notes TEXT DEFAULT '',
                    created_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS rules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_class TEXT NOT NULL,
                    target_class TEXT NOT NULL,
                    confidence_threshold REAL,
                    video_pattern TEXT,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    usage_count INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS class_stats (
                    class_name TEXT PRIMARY KEY,
                    total_detections INTEGER NOT NULL DEFAULT 0,
                    total_corrections INTEGER NOT NULL DEFAULT 0,
                    avg_confidence REAL NOT NULL DEFAULT 0.0,
                    last_updated REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_corrections_video ON corrections(video_id);
                CREATE INDEX IF NOT EXISTS idx_corrections_class ON corrections(original_class);
                CREATE INDEX IF NOT EXISTS idx_rules_pattern ON rules(pattern_class);
            """)
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Corrections
    # ------------------------------------------------------------------

    def add_correction(self, correction: Correction) -> int:
        """Add a correction and return its ID."""
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                """INSERT INTO corrections
                   (video_id, frame_number, timestamp, original_class, corrected_class,
                    bbox, original_confidence, action, notes, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    correction.video_id,
                    correction.frame_number,
                    correction.timestamp,
                    correction.original_class,
                    correction.corrected_class,
                    json.dumps(correction.bbox) if correction.bbox else None,
                    correction.original_confidence,
                    correction.action,
                    correction.notes,
                    correction.created_at,
                ),
            )
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def get_corrections(
        self,
        video_id: Optional[str] = None,
        original_class: Optional[str] = None,
        action: Optional[str] = None,
        limit: int = 100,
    ) -> list[Correction]:
        """Query corrections with optional filters."""
        conn = self._get_conn()
        try:
            query = "SELECT * FROM corrections WHERE 1=1"
            params: list = []

            if video_id:
                query += " AND video_id = ?"
                params.append(video_id)
            if original_class:
                query += " AND original_class = ?"
                params.append(original_class)
            if action:
                query += " AND action = ?"
                params.append(action)

            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)

            rows = conn.execute(query, params).fetchall()
            return [self._row_to_correction(r) for r in rows]
        finally:
            conn.close()

    def delete_correction(self, correction_id: int):
        """Delete a correction by ID."""
        conn = self._get_conn()
        try:
            conn.execute("DELETE FROM corrections WHERE id = ?", (correction_id,))
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Rules
    # ------------------------------------------------------------------

    def add_rule(self, rule: CorrectionRule) -> int:
        """Add an auto-apply rule."""
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                """INSERT INTO rules
                   (pattern_class, target_class, confidence_threshold, video_pattern,
                    enabled, usage_count, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    rule.pattern_class,
                    rule.target_class,
                    rule.confidence_threshold,
                    rule.video_pattern,
                    1 if rule.enabled else 0,
                    rule.usage_count,
                    rule.created_at,
                ),
            )
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def get_rules(self, enabled_only: bool = True) -> list[CorrectionRule]:
        """Get all rules, optionally only enabled ones."""
        conn = self._get_conn()
        try:
            query = "SELECT * FROM rules"
            if enabled_only:
                query += " WHERE enabled = 1"
            query += " ORDER BY pattern_class"

            rows = conn.execute(query).fetchall()
            return [self._row_to_rule(r) for r in rows]
        finally:
            conn.close()

    def toggle_rule(self, rule_id: int, enabled: bool):
        """Enable or disable a rule."""
        conn = self._get_conn()
        try:
            conn.execute(
                "UPDATE rules SET enabled = ? WHERE id = ?",
                (1 if enabled else 0, rule_id),
            )
            conn.commit()
        finally:
            conn.close()

    def delete_rule(self, rule_id: int):
        """Delete a rule."""
        conn = self._get_conn()
        try:
            conn.execute("DELETE FROM rules WHERE id = ?", (rule_id,))
            conn.commit()
        finally:
            conn.close()

    def increment_rule_usage(self, rule_id: int):
        """Increment the usage counter for a rule."""
        conn = self._get_conn()
        try:
            conn.execute(
                "UPDATE rules SET usage_count = usage_count + 1 WHERE id = ?",
                (rule_id,),
            )
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Class Statistics
    # ------------------------------------------------------------------

    def update_class_stats(self, class_name: str, confidence: float, was_corrected: bool = False):
        """Update statistics for a detected class."""
        conn = self._get_conn()
        try:
            conn.execute(
                """INSERT INTO class_stats (class_name, total_detections, total_corrections, avg_confidence, last_updated)
                   VALUES (?, 1, ?, ?, ?)
                   ON CONFLICT(class_name) DO UPDATE SET
                       total_detections = total_detections + 1,
                       total_corrections = total_corrections + ?,
                       avg_confidence = (avg_confidence * (total_detections - 1) + ?) / total_detections,
                       last_updated = ?""",
                (
                    class_name,
                    1 if was_corrected else 0,
                    confidence,
                    time.time(),
                    1 if was_corrected else 0,
                    confidence,
                    time.time(),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def get_class_stats(self) -> dict[str, dict]:
        """Get statistics for all classes."""
        conn = self._get_conn()
        try:
            rows = conn.execute("SELECT * FROM class_stats ORDER BY total_detections DESC").fetchall()
            return {
                r["class_name"]: {
                    "total_detections": r["total_detections"],
                    "total_corrections": r["total_corrections"],
                    "avg_confidence": round(r["avg_confidence"], 3),
                    "correction_rate": round(r["total_corrections"] / max(1, r["total_detections"]), 3),
                }
                for r in rows
            }
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Auto-rule generation
    # ------------------------------------------------------------------

    def generate_rules_from_corrections(self, min_occurrences: int = 3):
        """
        Automatically generate rules from repeated corrections.
        If the same correction happens min_occurrences times, create a rule.
        """
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """SELECT original_class, corrected_class, COUNT(*) as count,
                          AVG(original_confidence) as avg_conf
                   FROM corrections
                   WHERE action = 'relabel'
                   GROUP BY original_class, corrected_class
                   HAVING count >= ?""",
                (min_occurrences,),
            ).fetchall()

            for row in rows:
                # Check if rule already exists
                existing = conn.execute(
                    "SELECT id FROM rules WHERE pattern_class = ? AND target_class = ?",
                    (row["original_class"], row["corrected_class"]),
                ).fetchone()

                if not existing:
                    conn.execute(
                        """INSERT INTO rules
                           (pattern_class, target_class, confidence_threshold, enabled, usage_count, created_at)
                           VALUES (?, ?, ?, 1, 0, ?)""",
                        (
                            row["original_class"],
                            row["corrected_class"],
                            round(row["avg_conf"] + 0.1, 2),  # Slightly above avg confidence
                            time.time(),
                        ),
                    )

            conn.commit()
            return len(rows)
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _row_to_correction(self, row: sqlite3.Row) -> Correction:
        bbox = json.loads(row["bbox"]) if row["bbox"] else None
        return Correction(
            id=row["id"],
            video_id=row["video_id"],
            frame_number=row["frame_number"],
            timestamp=row["timestamp"],
            original_class=row["original_class"],
            corrected_class=row["corrected_class"],
            bbox=bbox,
            original_confidence=row["original_confidence"],
            action=row["action"],
            notes=row["notes"],
            created_at=row["created_at"],
        )

    def _row_to_rule(self, row: sqlite3.Row) -> CorrectionRule:
        return CorrectionRule(
            id=row["id"],
            pattern_class=row["pattern_class"],
            target_class=row["target_class"],
            confidence_threshold=row["confidence_threshold"],
            video_pattern=row["video_pattern"],
            enabled=bool(row["enabled"]),
            usage_count=row["usage_count"],
            created_at=row["created_at"],
        )

    def export_corrections(self, output_path: Path):
        """Export all corrections to JSON."""
        corrections = self.get_corrections(limit=10000)
        rules = self.get_rules(enabled_only=False)
        stats = self.get_class_stats()

        data = {
            "corrections": [asdict(c) for c in corrections],
            "rules": [asdict(r) for r in rules],
            "class_stats": stats,
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        return output_path


# Module-level singleton
_store: Optional[CorrectionStore] = None


def get_store(db_path: Path = CORRECTIONS_DB) -> CorrectionStore:
    """Get or create the correction store singleton."""
    global _store
    if _store is None or _store.db_path != db_path:
        _store = CorrectionStore(db_path)
    return _store
