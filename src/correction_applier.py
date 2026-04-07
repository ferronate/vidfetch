"""
Correction applier — applies user corrections and auto-rules to detection results.

This module sits between the raw detector output and the final results,
applying relabeling, deletion, and confidence adjustments based on
stored correction rules.
"""
from __future__ import annotations

import fnmatch
import logging
from typing import List, Dict, Any, Optional

from src.corrections import CorrectionRule, get_store

logger = logging.getLogger(__name__)


def apply_corrections(
    detections: List[Dict[str, Any]],
    video_id: str = "",
    rules: Optional[List[CorrectionRule]] = None,
) -> List[Dict[str, Any]]:
    """
    Apply correction rules to a list of detections.

    Args:
        detections: List of detection dicts with 'class', 'confidence', 'bbox', etc.
        video_id: Video identifier for video-specific rules.
        rules: Optional list of rules to apply. If None, loads from store.

    Returns:
        Filtered and corrected list of detections.
    """
    if rules is None:
        rules = get_store().get_rules(enabled_only=True)

    if not rules:
        return detections

    corrected: List[Dict[str, Any]] = []
    rules_applied = 0

    for det in detections:
        original_class = det.get("class", "")
        confidence = det.get("confidence", 0.0)

        # Find matching rule
        matched_rule = _find_matching_rule(rules, original_class, confidence, video_id)

        if matched_rule:
            if matched_rule.target_class == "__DELETE__":
                # Rule says to delete this detection
                logger.debug(
                    f"Deleted detection: {original_class} (conf={confidence:.3f}) "
                    f"[rule #{matched_rule.id}]"
                )
                get_store().increment_rule_usage(matched_rule.id)
                rules_applied += 1
                continue
            elif matched_rule.target_class == "__IGNORE__":
                # Keep but mark as ignored (low priority)
                det["corrected"] = True
                det["original_class"] = original_class
                det["correction_rule_id"] = matched_rule.id
                corrected.append(det)
                get_store().increment_rule_usage(matched_rule.id)
                rules_applied += 1
            else:
                # Relabel
                det["class"] = matched_rule.target_class
                det["corrected"] = True
                det["original_class"] = original_class
                det["correction_rule_id"] = matched_rule.id
                corrected.append(det)
                logger.debug(
                    f"Relabeled: {original_class} → {matched_rule.target_class} "
                    f"(conf={confidence:.3f}) [rule #{matched_rule.id}]"
                )
                get_store().increment_rule_usage(matched_rule.id)
                rules_applied += 1
        else:
            corrected.append(det)

    if rules_applied > 0:
        logger.info(f"Applied {rules_applied} correction(s) to detections")

    return corrected


def apply_corrections_to_timeline(
    timeline: List[Dict[str, Any]],
    video_id: str = "",
    rules: Optional[List[CorrectionRule]] = None,
) -> List[Dict[str, Any]]:
    """
    Apply correction rules to an entire detection timeline.

    Args:
        timeline: List of timeline entries with 't' and 'objects' keys.
        video_id: Video identifier.
        rules: Optional rules list.

    Returns:
        Corrected timeline.
    """
    corrected_timeline = []

    for entry in timeline:
        objects = entry.get("objects", [])
        corrected_objects = apply_corrections(objects, video_id=video_id, rules=rules)

        if corrected_objects:
            corrected_timeline.append({
                "t": entry["t"],
                "objects": corrected_objects,
            })

    return corrected_timeline


def _find_matching_rule(
    rules: List[CorrectionRule],
    class_name: str,
    confidence: float,
    video_id: str = "",
) -> Optional[CorrectionRule]:
    """Find the first matching rule for a detection."""
    for rule in rules:
        if not rule.enabled:
            continue

        # Check class match
        if rule.pattern_class != class_name:
            # Support wildcard patterns like "dog*"
            if not fnmatch.fnmatch(class_name, rule.pattern_class):
                continue

        # Check confidence threshold
        if rule.confidence_threshold is not None:
            if confidence >= rule.confidence_threshold:
                continue  # Only apply rule if confidence is below threshold

        # Check video pattern
        if rule.video_pattern:
            if not fnmatch.fnmatch(video_id, rule.video_pattern):
                continue

        return rule

    return None


def get_correction_summary() -> Dict[str, Any]:
    """Get a summary of all corrections and their impact."""
    store = get_store()
    stats = store.get_class_stats()
    rules = store.get_rules(enabled_only=False)
    corrections = store.get_corrections(limit=1000)

    return {
        "total_corrections": len(corrections),
        "total_rules": len(rules),
        "active_rules": sum(1 for r in rules if r.enabled),
        "class_stats": stats,
        "recent_corrections": [
            {
                "id": c.id,
                "video": c.video_id,
                "original": c.original_class,
                "corrected": c.corrected_class,
                "action": c.action,
                "confidence": c.original_confidence,
            }
            for c in corrections[:20]
        ],
    }
