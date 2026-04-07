"""
Inference service: manages a ProcessPoolExecutor of worker processes that run
inference via `src.inference_worker`.

Public API:
- init_pool(max_workers=None)
- warm_workers(...)
- submit_job(...)
- get_job_status(job_id)
- get_job_result(job_id)
- shutdown()

This module keeps a simple in-memory job registry. Jobs are executed in worker
processes via `inference_worker.run_detection`.
"""
from __future__ import annotations

import os
import time
import uuid
import json
import tempfile
import threading
import concurrent.futures
from pathlib import Path
from typing import Optional, Dict, Any, List

from src import inference_worker
from src import config
from src.utils import INDEX_FILE
import logging

logger = logging.getLogger(__name__)

_INDEX_FILE = INDEX_FILE

_POOL: Optional[concurrent.futures.ProcessPoolExecutor] = None
_POOL_LOCK = threading.Lock()

_JOBS: Dict[str, Dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()
_INDEX_LOCK = threading.Lock()


def init_pool(max_workers: Optional[int] = None) -> concurrent.futures.ProcessPoolExecutor:
    """Lazily create a shared ProcessPoolExecutor for inference workers."""
    global _POOL
    with _POOL_LOCK:
        if _POOL is None:
            if max_workers is None:
                # Read configured max workers if available
                try:
                    max_workers = int(getattr(config, "MAX_WORKERS", max(1, (os.cpu_count() or 2) - 1)))
                except Exception:
                    max_workers = max(1, (os.cpu_count() or 2) - 1)
            logger.info(f"Initializing ProcessPoolExecutor with {max_workers} workers")
            _POOL = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)
        return _POOL


def warm_workers(
    detector_type: str = "auto",
    confidence: float = 0.10,
    sample_fps: float = 0.5,
    enable_tracking: bool = True,
    enable_adaptive_sampling: bool = True,
    enable_temporal_aggregation: bool = True,
    timeout: float = 60.0,
    batch_size: int = 1,
) -> int:
    """Warm up each worker by initializing a detector inside the worker process.

    Returns the number of workers that successfully warmed within `timeout` seconds.
    """
    pool = init_pool()
    # Try to determine number of workers
    workers = getattr(pool, "_max_workers", None) or (os.cpu_count() or 1)

    futures: List[concurrent.futures.Future] = []
    for _ in range(workers):
        # Submit the prewarm helper which forces model/session load inside each worker
        futures.append(
            pool.submit(
                inference_worker.prewarm_detector,
                detector_type,
                confidence,
                sample_fps,
                enable_tracking,
                enable_adaptive_sampling,
                enable_temporal_aggregation,
            )
        )

    done, not_done = concurrent.futures.wait(futures, timeout=timeout)
    success = len(done)
    logger.info(f"Warmed {success}/{workers} workers")
    return success


def _atomic_save_index(index_path: Path, data: Dict[str, Any]):
    index_path.parent.mkdir(parents=True, exist_ok=True)
    # Use tempfile to write then rename
    fd, tmp_path = tempfile.mkstemp(prefix=index_path.name, dir=str(index_path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, index_path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def _load_index() -> Dict[str, Any]:
    if not _INDEX_FILE.exists():
        return {}
    try:
        with open(_INDEX_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def load_index_snapshot() -> Dict[str, Any]:
    """Load index while holding the shared index lock."""
    with _INDEX_LOCK:
        return _load_index()


def overwrite_index(data: Dict[str, Any]):
    """Overwrite index atomically while holding the shared index lock."""
    with _INDEX_LOCK:
        _atomic_save_index(_INDEX_FILE, data)


def upsert_index_entry(video_id: str, entry: Dict[str, Any]):
    """Insert or update one video entry atomically in the index."""
    with _INDEX_LOCK:
        idx = _load_index()
        idx[video_id] = entry
        _atomic_save_index(_INDEX_FILE, idx)


def submit_job(
    video_path: str,
    detector_type: str = "auto",
    confidence: float = 0.10,
    sample_fps: float = 0.5,
    prompt: Optional[str] = None,
    enable_tracking: bool = True,
    enable_adaptive_sampling: bool = True,
    enable_temporal_aggregation: bool = True,
    batch_size: int = 1,
    save_to_index: bool = False,
) -> str:
    """Submit a detection job to the pool and return a job id.

    If `save_to_index` is True, the result will be saved to the JSON index file
    when the job completes.
    """
    pool = init_pool()

    job_id = uuid.uuid4().hex
    video_id = Path(video_path).stem

    job = {
        "id": job_id,
        "video_id": video_id,
        "status": "submitted",
        "created_at": time.time(),
        "started_at": None,
        "finished_at": None,
        "result": None,
        "error": None,
        "future": None,
    }

    with _JOBS_LOCK:
        _JOBS[job_id] = job

    future = pool.submit(
        inference_worker.run_detection,
        str(video_path),
        detector_type,
        float(confidence),
        float(sample_fps),
        prompt,
        enable_tracking,
        enable_adaptive_sampling,
        enable_temporal_aggregation,
        int(batch_size),
    )

    def _done_callback(fut: concurrent.futures.Future):
        try:
            res = fut.result()
            with _JOBS_LOCK:
                j = _JOBS.get(job_id)
                if j:
                    j["status"] = "done"
                    j["finished_at"] = time.time()
                    j["result"] = res

            if save_to_index:
                try:
                    upsert_index_entry(j["video_id"], res)
                except Exception as e:
                    logger.exception(f"Failed to save index for {j['video_id']}: {e}")

        except Exception as e:
            logger.exception(f"Job {job_id} failed: {e}")
            with _JOBS_LOCK:
                j = _JOBS.get(job_id)
                if j:
                    j["status"] = "failed"
                    j["finished_at"] = time.time()
                    j["error"] = str(e)

            if save_to_index:
                try:
                    upsert_index_entry(
                        j["video_id"],
                        {"error": str(e), "classes": [], "timeline": []},
                    )
                except Exception:
                    pass

    future.add_done_callback(_done_callback)

    with _JOBS_LOCK:
        _JOBS[job_id]["future"] = future
        _JOBS[job_id]["status"] = "queued"
        _JOBS[job_id]["started_at"] = time.time()

    return job_id


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _JOBS_LOCK:
        j = _JOBS.get(job_id)
        if not j:
            return None
        # Return shallow copy without future object
        return {k: v for k, v in j.items() if k != "future"}


def get_job_result(job_id: str) -> Optional[Dict[str, Any]]:
    with _JOBS_LOCK:
        j = _JOBS.get(job_id)
        if not j:
            return None
        if j.get("status") == "done":
            return j.get("result")
        return None


def shutdown(wait: bool = True):
    global _POOL
    with _POOL_LOCK:
        if _POOL is not None:
            _POOL.shutdown(wait=wait)
            _POOL = None


# Optional: expose a helper to list jobs
def list_jobs() -> List[Dict[str, Any]]:
    with _JOBS_LOCK:
        return [{k: v for k, v in j.items() if k != "future"} for j in _JOBS.values()]
