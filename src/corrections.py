"""
Correction feedback system for object detection.

Stores user corrections and rules in JSON for easy inspection and portability.
A one-time migration from legacy SQLite is supported when corrections.json
is missing and corrections.db is present.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from src.utils import DATA_DIR

logger = logging.getLogger(__name__)

CORRECTIONS_FILE = DATA_DIR / "corrections.json"
LEGACY_SQLITE_FILE = DATA_DIR / "corrections.db"


@dataclass
class Correction:
    """A single user correction."""

    id: Optional[int] = None
    video_id: str = ""
    frame_number: int = 0
    timestamp: float = 0.0
    original_class: str = ""
    corrected_class: str = ""
    bbox: Optional[list[float]] = None
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
    pattern_class: str = ""
    target_class: str = ""
    confidence_threshold: Optional[float] = None
    video_pattern: Optional[str] = None
    enabled: bool = True
    usage_count: int = 0
    created_at: float = 0.0

    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()


class CorrectionStore:
    """JSON-backed store for corrections and rules."""

    def __init__(self, db_path: Path = CORRECTIONS_FILE):
        # Keep db_path attr for backward compatibility with existing callers.
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._ensure_store()

    def _empty_payload(self) -> dict:
        return {
            "version": 1,
            "updated_at": time.time(),
            "next_ids": {"correction": 1, "rule": 1},
            "corrections": [],
            "rules": [],
            "class_stats": {},
        }

    def _ensure_store(self):
        if self.db_path.exists():
            return

        # One-time best-effort migration from legacy SQLite file.
        if LEGACY_SQLITE_FILE.exists():
            try:
                payload = self._migrate_from_legacy_sqlite(LEGACY_SQLITE_FILE)
                self._save_payload(payload)
                logger.info("Migrated corrections from %s to %s", LEGACY_SQLITE_FILE, self.db_path)
                return
            except Exception as e:
                logger.warning("Legacy corrections migration failed, creating fresh JSON store: %s", e)

        self._save_payload(self._empty_payload())

    def _load_payload(self) -> dict:
        with self._lock:
            if not self.db_path.exists():
                return self._empty_payload()

            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception as e:
                logger.error("Failed to read corrections JSON, using empty payload: %s", e)
                return self._empty_payload()

            if not isinstance(payload, dict):
                return self._empty_payload()

            payload.setdefault("version", 1)
            payload.setdefault("updated_at", time.time())
            payload.setdefault("next_ids", {"correction": 1, "rule": 1})
            payload.setdefault("corrections", [])
            payload.setdefault("rules", [])
            payload.setdefault("class_stats", {})
            payload["next_ids"].setdefault("correction", 1)
            payload["next_ids"].setdefault("rule", 1)
            return payload

    def _save_payload(self, payload: dict):
        with self._lock:
            payload["updated_at"] = time.time()
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(prefix=f"{self.db_path.name}.", dir=str(self.db_path.parent))
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)
                os.replace(tmp_path, self.db_path)
            finally:
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass

    def _migrate_from_legacy_sqlite(self, sqlite_path: Path) -> dict:
        import sqlite3

        payload = self._empty_payload()

        conn = sqlite3.connect(str(sqlite_path))
        conn.row_factory = sqlite3.Row
        try:
            correction_rows = conn.execute(
                """SELECT id, video_id, frame_number, timestamp, original_class, corrected_class,
                          bbox, original_confidence, action, notes, created_at
                   FROM corrections
                   ORDER BY created_at ASC"""
            ).fetchall()
            rule_rows = conn.execute(
                """SELECT id, pattern_class, target_class, confidence_threshold, video_pattern,
                          enabled, usage_count, created_at
                   FROM rules
                   ORDER BY created_at ASC"""
            ).fetchall()
            stats_rows = conn.execute(
                """SELECT class_name, total_detections, total_corrections, avg_confidence, last_updated
                   FROM class_stats"""
            ).fetchall()
        finally:
            conn.close()

        max_c_id = 0
        max_r_id = 0

        for row in correction_rows:
            cid = int(row["id"])
            max_c_id = max(max_c_id, cid)
            bbox = None
            if row["bbox"]:
                try:
                    bbox = json.loads(row["bbox"])
                except Exception:
                    bbox = None

            payload["corrections"].append(
                {
                    "id": cid,
                    "video_id": row["video_id"],
                    "frame_number": int(row["frame_number"]),
                    "timestamp": float(row["timestamp"]),
                    "original_class": row["original_class"],
                    "corrected_class": row["corrected_class"],
                    "bbox": bbox,
                    "original_confidence": float(row["original_confidence"]),
                    "action": row["action"],
                    "notes": row["notes"] or "",
                    "created_at": float(row["created_at"]),
                }
            )

        for row in rule_rows:
            rid = int(row["id"])
            max_r_id = max(max_r_id, rid)
            payload["rules"].append(
                {
                    "id": rid,
                    "pattern_class": row["pattern_class"],
                    "target_class": row["target_class"],
                    "confidence_threshold": (
                        float(row["confidence_threshold"]) if row["confidence_threshold"] is not None else None
                    ),
                    "video_pattern": row["video_pattern"],
                    "enabled": bool(row["enabled"]),
                    "usage_count": int(row["usage_count"]),
                    "created_at": float(row["created_at"]),
                }
            )

        for row in stats_rows:
            class_name = row["class_name"]
            total_detections = int(row["total_detections"])
            total_corrections = int(row["total_corrections"])
            payload["class_stats"][class_name] = {
                "total_detections": total_detections,
                "total_corrections": total_corrections,
                "avg_confidence": round(float(row["avg_confidence"]), 3),
                "correction_rate": round(total_corrections / max(1, total_detections), 3),
                "last_updated": float(row["last_updated"]),
            }

        payload["next_ids"] = {"correction": max_c_id + 1, "rule": max_r_id + 1}
        return payload

    # ------------------------------------------------------------------
    # Corrections
    # ------------------------------------------------------------------

    def add_correction(self, correction: Correction) -> int:
        payload = self._load_payload()

        correction_id = int(correction.id or payload["next_ids"]["correction"])
        payload["next_ids"]["correction"] = max(payload["next_ids"]["correction"], correction_id + 1)

        record = asdict(correction)
        record["id"] = correction_id
        payload["corrections"].append(record)

        self._save_payload(payload)
        return correction_id

    def get_corrections(
        self,
        video_id: Optional[str] = None,
        original_class: Optional[str] = None,
        action: Optional[str] = None,
        limit: int = 100,
    ) -> list[Correction]:
        payload = self._load_payload()
        items = payload.get("corrections", [])

        filtered = []
        for item in items:
            if video_id and item.get("video_id") != video_id:
                continue
            if original_class and item.get("original_class") != original_class:
                continue
            if action and item.get("action") != action:
                continue
            filtered.append(item)

        filtered.sort(key=lambda x: float(x.get("created_at", 0.0)), reverse=True)
        if limit > 0:
            filtered = filtered[:limit]

        return [Correction(**item) for item in filtered]

    def delete_correction(self, correction_id: int):
        payload = self._load_payload()
        before = len(payload["corrections"])
        payload["corrections"] = [c for c in payload["corrections"] if int(c.get("id", -1)) != correction_id]
        if len(payload["corrections"]) != before:
            self._save_payload(payload)

    # ------------------------------------------------------------------
    # Rules
    # ------------------------------------------------------------------

    def add_rule(self, rule: CorrectionRule) -> int:
        payload = self._load_payload()

        rule_id = int(rule.id or payload["next_ids"]["rule"])
        payload["next_ids"]["rule"] = max(payload["next_ids"]["rule"], rule_id + 1)

        record = asdict(rule)
        record["id"] = rule_id
        payload["rules"].append(record)

        self._save_payload(payload)
        return rule_id

    def get_rules(self, enabled_only: bool = True) -> list[CorrectionRule]:
        payload = self._load_payload()
        rules = payload.get("rules", [])

        if enabled_only:
            rules = [r for r in rules if bool(r.get("enabled", True))]

        rules = sorted(rules, key=lambda x: str(x.get("pattern_class", "")))
        return [CorrectionRule(**r) for r in rules]

    def toggle_rule(self, rule_id: int, enabled: bool):
        payload = self._load_payload()
        changed = False
        for rule in payload["rules"]:
            if int(rule.get("id", -1)) == rule_id:
                rule["enabled"] = bool(enabled)
                changed = True
                break

        if changed:
            self._save_payload(payload)

    def delete_rule(self, rule_id: int):
        payload = self._load_payload()
        before = len(payload["rules"])
        payload["rules"] = [r for r in payload["rules"] if int(r.get("id", -1)) != rule_id]
        if len(payload["rules"]) != before:
            self._save_payload(payload)

    def increment_rule_usage(self, rule_id: int):
        payload = self._load_payload()
        changed = False
        for rule in payload["rules"]:
            if int(rule.get("id", -1)) == rule_id:
                rule["usage_count"] = int(rule.get("usage_count", 0)) + 1
                changed = True
                break

        if changed:
            self._save_payload(payload)

    # ------------------------------------------------------------------
    # Class Statistics
    # ------------------------------------------------------------------

    def update_class_stats(self, class_name: str, confidence: float, was_corrected: bool = False):
        payload = self._load_payload()
        stats = payload.setdefault("class_stats", {})

        current = stats.get(
            class_name,
            {
                "total_detections": 0,
                "total_corrections": 0,
                "avg_confidence": 0.0,
                "correction_rate": 0.0,
                "last_updated": time.time(),
            },
        )

        total_detections = int(current.get("total_detections", 0)) + 1
        total_corrections = int(current.get("total_corrections", 0)) + (1 if was_corrected else 0)
        previous_avg = float(current.get("avg_confidence", 0.0))
        avg_confidence = ((previous_avg * (total_detections - 1)) + float(confidence)) / total_detections

        stats[class_name] = {
            "total_detections": total_detections,
            "total_corrections": total_corrections,
            "avg_confidence": round(avg_confidence, 3),
            "correction_rate": round(total_corrections / max(1, total_detections), 3),
            "last_updated": time.time(),
        }

        self._save_payload(payload)

    def get_class_stats(self) -> dict[str, dict]:
        payload = self._load_payload()
        return dict(payload.get("class_stats", {}))

    # ------------------------------------------------------------------
    # Auto-rule generation
    # ------------------------------------------------------------------

    def generate_rules_from_corrections(self, min_occurrences: int = 3):
        payload = self._load_payload()

        relabels = [c for c in payload["corrections"] if c.get("action") == "relabel"]
        pair_stats: dict[tuple[str, str], dict[str, float]] = {}
        for c in relabels:
            key = (str(c.get("original_class", "")), str(c.get("corrected_class", "")))
            stats = pair_stats.setdefault(key, {"count": 0, "confidence_sum": 0.0})
            stats["count"] += 1
            stats["confidence_sum"] += float(c.get("original_confidence", 0.0))

        existing_pairs = {
            (str(r.get("pattern_class", "")), str(r.get("target_class", "")))
            for r in payload.get("rules", [])
        }

        generated = 0
        for (pattern_class, target_class), stats in pair_stats.items():
            count = int(stats["count"])
            if count < min_occurrences:
                continue
            if (pattern_class, target_class) in existing_pairs:
                continue

            avg_conf = stats["confidence_sum"] / max(1, count)
            new_rule = CorrectionRule(
                id=payload["next_ids"]["rule"],
                pattern_class=pattern_class,
                target_class=target_class,
                confidence_threshold=round(avg_conf + 0.1, 2),
                enabled=True,
                usage_count=0,
            )
            payload["next_ids"]["rule"] += 1
            payload["rules"].append(asdict(new_rule))
            existing_pairs.add((pattern_class, target_class))
            generated += 1

        if generated > 0:
            self._save_payload(payload)

        return generated

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_corrections(self, output_path: Path):
        payload = self._load_payload()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return output_path


_store: Optional[CorrectionStore] = None


def get_store(db_path: Path = CORRECTIONS_FILE) -> CorrectionStore:
    """Get or create the correction store singleton."""
    global _store
    if _store is None or _store.db_path != db_path:
        _store = CorrectionStore(db_path)
    return _store
