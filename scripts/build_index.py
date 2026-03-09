"""
Build the video index from a directory of videos. Saves to index_store/ by default.
Usage: python -m scripts.build_index [video_dir] [--index-dir index_store]
"""
from pathlib import Path
import argparse
import sys

# Allow running as script from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.extract import video_to_feature
from src.index import VideoIndex

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def find_videos(directory: Path) -> list[Path]:
    return [p for p in directory.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS]


def main():
    parser = argparse.ArgumentParser(description="Build video retrieval index")
    parser.add_argument(
        "video_dir",
        nargs="?",
        default="data",
        help="Directory containing videos (default: data)",
    )
    parser.add_argument(
        "--index-dir",
        default="index_store",
        help="Where to save the index (default: index_store)",
    )
    parser.add_argument("--fps", type=float, default=1.0, help="Frames per second to sample")
    parser.add_argument("--max-frames", type=int, default=50, help="Max frames per video")
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    if not video_dir.exists():
        print(f"Error: directory not found: {video_dir}")
        print("Create a 'data' folder and add some videos, or pass another path.")
        sys.exit(1)

    videos = find_videos(video_dir)
    if not videos:
        print(f"No videos found in {video_dir} (looking for {VIDEO_EXTENSIONS})")
        sys.exit(1)

    print(f"Found {len(videos)} videos. Building index...")
    index = VideoIndex()
    for i, path in enumerate(videos):
        try:
            feat = video_to_feature(path, fps=args.fps, max_frames=args.max_frames)
            video_id = path.stem
            index.add(video_id, str(path.resolve()), feat)
            print(f"  [{i + 1}/{len(videos)}] {path.name}")
        except Exception as e:
            print(f"  Skip {path.name}: {e}")

    index.save(args.index_dir)
    print(f"Index saved to {args.index_dir} ({len(index.ids)} videos).")


if __name__ == "__main__":
    main()
