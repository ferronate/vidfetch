"""
Runtime configuration for vidfetch.

Reads environment-driven settings with sensible defaults.
CPU-specific tuning now lives in ``cpu_profile.py``; this module provides
the remaining global knobs (paths, env overrides).
"""
from __future__ import annotations

import os
from pathlib import Path

from .utils import ROOT, DATA_DIR, INDEX_DIR, MODELS_DIR, INDEX_FILE  # re-export


def _get_env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).lower() in ("1", "true", "yes", "on")


def _get_env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return int(v)
    except Exception:
        return default


# Model / runtime toggles
CPU_ONLY: bool = _get_env_bool("VF_CPU_ONLY", True)
FORCE_YOLO_CPU: bool = _get_env_bool("VF_FORCE_YOLO_CPU", True)

# ONNX Runtime thread settings (fallback; prefer CPUProfile.num_threads)
ORT_INTRA_THREADS: int = _get_env_int("VF_ORT_INTRA_THREADS", max(1, (os.cpu_count() or 1) // 2))
ORT_INTER_THREADS: int = _get_env_int("VF_ORT_INTER_THREADS", 1)

# Inference sizing and batching (fallback; prefer CPUProfile values)
BATCH_SIZE: int = _get_env_int("VF_BATCH_SIZE", 4)
INPUT_SIZE: int = _get_env_int("VF_INPUT_SIZE", 640)

# Process pool sizing
MAX_WORKERS: int = _get_env_int("VF_MAX_WORKERS", max(1, (os.cpu_count() or 2) - 1))


__all__ = [
    "ROOT",
    "DATA_DIR",
    "INDEX_DIR",
    "MODELS_DIR",
    "INDEX_FILE",
    "CPU_ONLY",
    "FORCE_YOLO_CPU",
    "ORT_INTRA_THREADS",
    "ORT_INTER_THREADS",
    "BATCH_SIZE",
    "INPUT_SIZE",
    "MAX_WORKERS",
]
