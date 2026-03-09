"""
Build the object index: for each video, detect objects and save to index_store/objects.json.
- --use-yolo (recommended): YOLOv8 COCO detector. Reliable, runs on CPU. pip install ultralytics
- Default: MobileNet-SSD (run python -m scripts.download_detector_model first).
- --use-clip: CLIP ViT-B-32 image-text similarity; runs on CPU. pip install -r requirements-clip.txt
Usage: python -m scripts.build_object_index [video_dir] [--use-yolo] [--index-dir index_store]
"""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def find_videos(directory: Path) -> list[Path]:
    return [p for p in directory.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS]


def main():
    parser = argparse.ArgumentParser(description="Build object index for videos")
    parser.add_argument("video_dir", nargs="?", default="data", help="Directory containing videos")
    parser.add_argument("--index-dir", default="index_store", help="Where to save objects.json")
    parser.add_argument("--fps", type=float, default=0.5, help="Frames per second to sample")
    parser.add_argument("--max-frames", type=int, default=30, help="Max frames per video")
    parser.add_argument("--use-yolo", action="store_true", help="Use YOLOv8 (COCO) for reliable object detection; pip install ultralytics")
    parser.add_argument("--use-clip", action="store_true", help="Use CLIP (ViT-B-32) on CPU for semantic matching")
    parser.add_argument("--add-clip-concepts", action="store_true", help="With --use-yolo: also detect globe, fire, surfboard, etc. (needs CLIP)")
    parser.add_argument("--yolo-model", choices=("s", "n"), default="s", help="YOLO size: s=small (better accuracy), n=nano (faster)")
    parser.add_argument("--min-frames", type=int, default=2, help="Min frames a class must appear in to count (default 2)")
    parser.add_argument("--confidence", type=float, default=None, help="SSD/YOLO confidence threshold (default 0.58 for YOLO, 0.6 for SSD)")
    parser.add_argument("--clip-threshold", type=float, default=None, help="CLIP similarity threshold (default 0.24)")
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    if not video_dir.exists():
        print(f"Error: directory not found: {video_dir}")
        sys.exit(1)

    videos = find_videos(video_dir)
    if not videos:
        print(f"No videos found in {video_dir}")
        sys.exit(1)

    def build_one(path, get_classes_and_frames):
        video_id = path.stem
        classes, frames = get_classes_and_frames(path)
        return video_id, {"classes": sorted(classes), "frames": frames}

    object_index = {}
    if args.use_yolo:
        try:
            from src.yolo_detect import video_to_object_timeline as yolo_timeline, DEFAULT_CONFIDENCE
        except Exception as e:
            print("YOLO not available:", e)
            print("Install with: pip install -r requirements-yolo.txt")
            sys.exit(1)
        conf = args.confidence if args.confidence is not None else DEFAULT_CONFIDENCE
        min_f = args.min_frames
        model_sz = args.yolo_model
        msg = f"Using YOLOv8{model_sz} (COCO), confidence={conf}, min_frames={min_f}"
        if args.add_clip_concepts:
            msg += " + CLIP concepts (globe, fire, surfboard, etc.)"
        print(f"{msg}. Found {len(videos)} videos...")
        if not args.add_clip_concepts:
            print("Tip: for globe, fire, surfing, cooking add: --add-clip-concepts (needs pip install -r requirements-clip.txt)")
        for i, path in enumerate(videos):
            try:
                def _yolo(p, c=conf, mf=min_f, ms=model_sz):
                    return yolo_timeline(p, fps=args.fps, max_frames=args.max_frames, confidence=c, min_frames_per_class=mf, model_size=ms)
                vid_id, data = build_one(path, _yolo)
                if args.add_clip_concepts:
                    try:
                        from src.clip_detect import video_to_extra_concepts_clip
                        clip_classes, clip_timeline = video_to_extra_concepts_clip(
                            path, fps=args.fps, max_frames=args.max_frames, min_frames_per_class=min_f
                        )
                        # Merge: same number of frames, same t; merge classes and per-frame objects
                        data["classes"] = sorted(set(data["classes"]) | clip_classes)
                        for j, entry in enumerate(data["frames"]):
                            if j < len(clip_timeline):
                                entry["objects"] = list(dict.fromkeys(entry["objects"] + clip_timeline[j]["objects"]))
                    except Exception as e:
                        print(f"  (CLIP concepts skip: {e})")
                object_index[vid_id] = data
                cls_str = ", ".join(data["classes"]) or "(none)"
                print(f"  [{i + 1}/{len(videos)}] {path.name} -> {cls_str}")
            except Exception as e:
                print(f"  Skip {path.name}: {e}")
    elif args.use_clip:
        try:
            from src.clip_detect import video_to_object_timeline_clip
        except Exception as e:
            print("CLIP not available:", e)
            print("Install with: pip install -r requirements-clip.txt")
            sys.exit(1)
        clip_th = args.clip_threshold if args.clip_threshold is not None else 0.24
        print(f"Using CLIP (ViT-B-32, CPU), threshold={clip_th}. Found {len(videos)} videos...")
        for i, path in enumerate(videos):
            try:
                vid_id, data = build_one(path, lambda p, th=clip_th: video_to_object_timeline_clip(p, fps=args.fps, max_frames=args.max_frames, threshold=th))
                object_index[vid_id] = data
                cls_str = ", ".join(data["classes"]) or "(none)"
                print(f"  [{i + 1}/{len(videos)}] {path.name} -> {cls_str}")
            except Exception as e:
                print(f"  Skip {path.name}: {e}")
    else:
        from src.detect import load_net, video_to_object_timeline, CONFIDENCE_THRESHOLD
        models_dir = Path(__file__).resolve().parent.parent / "models"
        try:
            net = load_net(models_dir)
        except FileNotFoundError as e:
            print(e)
            sys.exit(1)
        conf = args.confidence if args.confidence is not None else CONFIDENCE_THRESHOLD
        print(f"Using MobileNet-SSD, confidence={conf}. Found {len(videos)} videos...")
        for i, path in enumerate(videos):
            try:
                vid_id, data = build_one(path, lambda p, c=conf: video_to_object_timeline(p, net, fps=args.fps, max_frames=args.max_frames, confidence_threshold=c))
                object_index[vid_id] = data
                cls_str = ", ".join(data["classes"]) or "(none)"
                print(f"  [{i + 1}/{len(videos)}] {path.name} -> {cls_str}")
            except Exception as e:
                print(f"  Skip {path.name}: {e}")

    index_dir = Path(args.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    out_path = index_dir / "objects.json"
    with open(out_path, "w") as f:
        json.dump(object_index, f, indent=2)
    print(f"Object index saved to {out_path}")


if __name__ == "__main__":
    main()
