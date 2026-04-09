"""
CPU detection and hardware-aware profile selection for vidfetch.

Detects the host CPU capabilities (core count, instruction set extensions)
and returns a CPUProfile that tells the rest of the system which model,
input resolution, batch size, and thread count to use.

Usage:
    from src.cpu_profile import get_cpu_profile
    profile = get_cpu_profile()
    print(profile.tier, profile.model_key, profile.input_size)
"""
from __future__ import annotations

import logging
import multiprocessing
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.utils import ROOT, MODELS_DIR

logger = logging.getLogger(__name__)

# Repo root (one level above src/)
_ROOT = ROOT


@dataclass(frozen=True)
class CPUProfile:
    """Hardware-aware settings bundle.

    Every tunable that should change based on the host machine lives here.
    The rest of the codebase reads from this instead of making its own
    guesses about what model/size/threads to use.
    """

    tier: str  # "high", "medium", "low"
    cpu_brand: str  # human-readable CPU name
    cores: int

    # Model selection
    model_key: str  # key into MODEL_REGISTRY
    model_path: Path  # resolved absolute path to model file

    # Inference tuning
    input_size: int  # square input side length (pixels)
    batch_size: int  # frames per inference batch
    sample_fps: float  # base sampling rate for video processing
    num_threads: int  # intra-op threads for ONNX / torch


# ---------------------------------------------------------------------------
# Model registry — single source of truth for every model the project knows
# about. Paths are relative to the repo root and resolved at profile-build
# time so the rest of the code never has to guess.
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    "object365": {
        "path": MODELS_DIR / "yolo11n.pt",  # Updated to available model
        "classes": 80,
        "type": "yolo",
        "description": "YOLO11-nano trained on COCO (80 classes)",
    },
    "coco-nano": {
        "path": MODELS_DIR / "yolov8n.pt",
        "classes": 80,
        "type": "yolo",
        "description": "YOLOv8-nano trained on COCO (80 classes)",
    },
    "coco-small": {
        "path": MODELS_DIR / "yolov8s.pt",
        "classes": 80,
        "type": "yolo",
        "description": "YOLOv8-small trained on COCO (80 classes)",
    },
    "world-small": {
        "path": MODELS_DIR / "yolov8s-world.pt",
        "classes": "open",
        "type": "yolo-world",
        "description": "YOLOv8-small World (open-vocabulary)",
    },
    "coco-oiv7-nano": {
        "path": MODELS_DIR / "yolov8n-oiv7.pt",
        "classes": 600,
        "type": "yolo",
        "description": "YOLOv8-nano trained on Open Images V7",
    },
    "onnx-nano": {
        "path": MODELS_DIR / "yolov8n.onnx",
        "classes": 80,
        "type": "onnx",
        "description": "YOLOv8-nano ONNX (quantised / CPU-optimised)",
    },
    "onnx-nano-quant": {
        "path": MODELS_DIR / "yolov8n.quant.onnx",
        "classes": 80,
        "type": "onnx",
        "description": "YOLOv8-nano ONNX INT8 quantised",
    },
}


def _resolve_model_path(relative: str | Path) -> Path:
    """Resolve a model path relative to the repo root."""
    path = Path(relative)
    return path if path.is_absolute() else _ROOT / path


def resolve_model(key: str) -> Path:
    """Return the absolute path for a registry key. Raises KeyError if unknown."""
    entry = MODEL_REGISTRY[key]
    return _resolve_model_path(entry["path"])


def _scan_available_models() -> dict[str, Path]:
    """Scan the models/ directory and return available models keyed by type."""
    models_dir = MODELS_DIR
    if not models_dir.exists():
        return {}
    
    available: dict[str, Path] = {}
    for f in models_dir.iterdir():
        if f.suffix == '.pt':
            if 'world' in f.name.lower():
                available['yolo-world'] = f
            elif 'oiv7' in f.name.lower():
                available['yolo-oiv7'] = f
            else:
                available['yolo'] = f  # Standard YOLO
        elif f.suffix == '.onnx':
            available['onnx'] = f
    
    return available


def _select_best_available_model(cpu_tier: str) -> tuple[str, Path, str, int]:
    """
    Dynamically select the best available model from the models/ directory.
    
    Returns:
        (model_key, model_path, model_type, num_classes)
    """
    available = _scan_available_models()
    
    if not available:
        # Fallback to registry defaults
        entry = MODEL_REGISTRY["coco-nano"]
        return "coco-nano", _resolve_model_path(entry["path"]), "yolo", 80
    
    # Priority selection based on CPU tier and available models
    if cpu_tier == "high":
        # Prefer broader fixed-class coverage first for indexing/review stability.
        # Priority: OIV7 (600 classes) > YOLO-World (open vocab) > Standard YOLO > ONNX
        if 'yolo-oiv7' in available:
            return "coco-oiv7-nano", available['yolo-oiv7'], "yolo", 600
        if 'yolo-world' in available:
            return "world-small", available['yolo-world'], "yolo-world", "open"
        if 'yolo' in available:
            return "object365", available['yolo'], "yolo", 80
        if 'onnx' in available:
            return "onnx-nano", available['onnx'], "onnx", 80
    
    elif cpu_tier == "medium":
        # Prefer broader classes when available while keeping runtime practical.
        # Priority: OIV7 (600 classes) > YOLO-World > Standard YOLO > ONNX
        if 'yolo-oiv7' in available:
            return "coco-oiv7-nano", available['yolo-oiv7'], "yolo", 600
        if 'yolo-world' in available:
            return "world-small", available['yolo-world'], "yolo-world", "open"
        if 'yolo' in available:
            return "object365", available['yolo'], "yolo", 80
        if 'onnx' in available:
            return "onnx-nano", available['onnx'], "onnx", 80
    
    else:  # low tier
        # Prefer: ONNX (fastest) > Standard YOLO
        if 'onnx' in available:
            return "onnx-nano", available['onnx'], "onnx", 80
        if 'yolo' in available:
            return "coco-nano", available['yolo'], "yolo", 80
        if 'yolo-world' in available:
            return "world-small", available['yolo-world'], "yolo-world", "open"
    
    # Ultimate fallback
    entry = MODEL_REGISTRY["coco-nano"]
    return "coco-nano", _resolve_model_path(entry["path"]), "yolo", 80


# ---------------------------------------------------------------------------
# CPU detection
# ---------------------------------------------------------------------------

def detect_cpu_tier() -> tuple[str, str, int, list[str]]:
    """Detect CPU tier based on core count and instruction set extensions.

    Returns:
        (tier, brand_string, core_count, flags_list)

    Tier definitions:
        high   — 8+ cores AND AVX2 (or AVX-512)
        medium — 4+ cores AND at least SSE4
        low    — everything else
    """
    cores = multiprocessing.cpu_count() or 2
    brand = "unknown"
    flags: list[str] = []

    try:
        import cpuinfo

        info = cpuinfo.get_cpu_info()
        flags = info.get("flags", [])
        brand = info.get("brand_raw", "unknown")
    except ImportError:
        logger.warning("py-cpuinfo not installed — falling back to core-count heuristic")
    except Exception as exc:
        logger.warning("CPU detection failed (%s) — falling back to core-count heuristic", exc)

    has_avx2 = "avx2" in flags
    has_avx512 = "avx512f" in flags
    has_sse4 = "sse4_1" in flags or "sse4_2" in flags

    if has_avx512 or (has_avx2 and cores >= 8):
        tier = "high"
    elif has_avx2 or (has_sse4 and cores >= 4):
        tier = "medium"
    else:
        tier = "low"

    logger.info("CPU detected: %s (%d cores, tier=%s)", brand, cores, tier)
    return tier, brand, cores, flags


# ---------------------------------------------------------------------------
# Profile builder
# ---------------------------------------------------------------------------

# Cached profile so detection only runs once per process.
_cached_profile: Optional[CPUProfile] = None


def get_cpu_profile(*, force: bool = False) -> CPUProfile:
    """Return a CPUProfile for the current machine.

    The result is cached after the first call. Pass ``force=True`` to
    re-detect (useful for tests).
    """
    global _cached_profile
    if _cached_profile is not None and not force:
        return _cached_profile

    tier, brand, cores, _flags = detect_cpu_tier()

    # Dynamically select best available model
    model_key, model_path, model_type, num_classes = _select_best_available_model(tier)

    # ---- HIGH tier (8+ cores, AVX2/AVX-512) ----
    if tier == "high":
        input_size = 640
        batch_size = max(4, cores // 2)
        sample_fps = 2.0
        num_threads = max(2, cores // 2)

    # ---- MEDIUM tier (4+ cores, SSE4) ----
    elif tier == "medium":
        input_size = 640
        batch_size = max(2, cores // 2)
        sample_fps = 1.0
        num_threads = max(2, cores // 2)

    # ---- LOW tier ----
    else:
        input_size = 416
        batch_size = 1
        sample_fps = 0.5
        num_threads = max(1, cores)

    profile = CPUProfile(
        tier=tier,
        cpu_brand=brand,
        cores=cores,
        model_key=model_key,
        model_path=model_path,
        input_size=input_size,
        batch_size=batch_size,
        sample_fps=sample_fps,
        num_threads=num_threads,
    )

    logger.info(
        "CPU profile: tier=%s, model=%s (%s), input=%dpx, batch=%d, threads=%d",
        profile.tier,
        profile.model_key,
        model_type,
        profile.input_size,
        profile.batch_size,
        profile.num_threads,
    )

    _cached_profile = profile
    return profile
