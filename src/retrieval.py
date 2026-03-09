"""
Query interface: load index, extract query feature, return ranked results.
"""
from pathlib import Path
import time
from .index import VideoIndex
from .extract import video_to_feature


def load_index(index_dir: str | Path) -> VideoIndex:
    return VideoIndex.load(index_dir)


def query(
    index: VideoIndex,
    query_video_path: str | Path,
    k: int = 5,
    fps: float = 1.0,
    max_frames: int = 50,
) -> tuple[list[tuple[str, str, float]], float]:
    """
    Run retrieval for a query video. Returns (results, retrieval_time_seconds).
    results = [(video_id, path, distance), ...]
    """
    t0 = time.perf_counter()
    feat = video_to_feature(query_video_path, fps=fps, max_frames=max_frames)
    results = index.search(feat, k=k)
    elapsed = time.perf_counter() - t0
    return results, elapsed
