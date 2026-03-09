"""
Query the index with a video file. Prints top-k similar videos.
Usage: python -m scripts.query <query_video> [--index-dir index_store] [--k 5]
"""
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval import load_index, query


def main():
    parser = argparse.ArgumentParser(description="Query video retrieval index")
    parser.add_argument("query_video", help="Path to query video (or image for single-frame)")
    parser.add_argument("--index-dir", default="index_store", help="Index directory")
    parser.add_argument("--k", type=int, default=5, help="Number of results")
    args = parser.parse_args()

    query_path = Path(args.query_video)
    if not query_path.exists():
        print(f"Error: not found: {query_path}")
        sys.exit(1)

    index_path = Path(args.index_dir)
    if not (index_path / "features.npy").exists():
        print(f"Error: index not found in {index_path}. Run build_index first.")
        sys.exit(1)

    index = load_index(args.index_dir)
    results, elapsed = query(index, query_path, k=args.k)
    print(f"Query: {query_path.name}")
    print(f"Retrieval time: {elapsed:.3f} s")
    print(f"Top-{args.k} similar videos:")
    for i, (vid_id, path, dist) in enumerate(results, 1):
        print(f"  {i}. {vid_id} (distance={dist:.4f})")


if __name__ == "__main__":
    main()
