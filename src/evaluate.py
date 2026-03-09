"""
Simple evaluation: precision@k, recall@k, and mean retrieval time.
Expects a list of (query_path, set of relevant_video_ids) as ground truth.
"""
from pathlib import Path
from .index import VideoIndex
from .retrieval import query


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Precision@k = (relevant in top-k) / k."""
    top_k = retrieved_ids[:k]
    hits = sum(1 for vid in top_k if vid in relevant_ids)
    return hits / k if k else 0.0


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Recall@k = (relevant in top-k) / |relevant|."""
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for vid in top_k if vid in relevant_ids)
    return hits / len(relevant_ids)


def evaluate(
    index: VideoIndex,
    ground_truth: list[tuple[str, set[str]]],
    k: int = 5,
) -> dict:
    """
    ground_truth: list of (query_video_path, set of relevant video IDs).
    Returns dict with mean_precision_at_k, mean_recall_at_k, mean_retrieval_time_sec.
    """
    precisions = []
    recalls = []
    times = []
    for query_path, relevant in ground_truth:
        results, elapsed = query(index, query_path, k=k)
        retrieved_ids = [r[0] for r in results]
        precisions.append(precision_at_k(retrieved_ids, relevant, k))
        recalls.append(recall_at_k(retrieved_ids, relevant, k))
        times.append(elapsed)
    n = len(ground_truth)
    return {
        "mean_precision_at_k": sum(precisions) / n if n else 0.0,
        "mean_recall_at_k": sum(recalls) / n if n else 0.0,
        "mean_retrieval_time_sec": sum(times) / n if n else 0.0,
        "k": k,
    }
