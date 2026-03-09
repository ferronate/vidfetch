"""
Run evaluation: for each video in the index, use it as query and treat itself as relevant.
Reports mean precision@k, recall@k, and retrieval time.
Usage: python -m scripts.run_evaluation [--index-dir index_store] [--k 5]
"""
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.index import VideoIndex
from src.evaluate import evaluate


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval (self-query)")
    parser.add_argument("--index-dir", default="index_store", help="Index directory")
    parser.add_argument("--k", type=int, default=5, help="k for P@k and R@k")
    args = parser.parse_args()

    index_path = Path(args.index_dir)
    if not (index_path / "features.npy").exists():
        print(f"Error: index not found in {index_path}. Run build_index first.")
        sys.exit(1)

    index = VideoIndex.load(args.index_dir)
    # Ground truth: each video is relevant to itself (by id)
    ground_truth = [(path, {vid_id}) for vid_id, path in zip(index.ids, index.paths)]
    # Use path for query
    metrics = evaluate(index, ground_truth, k=args.k)
    print("Evaluation (query each video, relevant = itself):")
    print(f"  Mean Precision@{metrics['k']}: {metrics['mean_precision_at_k']:.4f}")
    print(f"  Mean Recall@{metrics['k']}:    {metrics['mean_recall_at_k']:.4f}")
    print(f"  Mean retrieval time:          {metrics['mean_retrieval_time_sec']:.4f} s")


if __name__ == "__main__":
    main()
