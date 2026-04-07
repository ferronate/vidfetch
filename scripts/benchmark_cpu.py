"""
Simple CPU benchmarking harness for detectors.

Usage examples:
  python scripts/benchmark_cpu.py --video data/sample.mp4 --detector onnx --frames 500
  python scripts/benchmark_cpu.py --video data/sample.mp4 --detector yolo --frames 500

This script measures inference-only time (not file read overhead) and reports
frames processed, total inference time and average ms/frame.
"""
from __future__ import annotations

import argparse
import time
import json
from pathlib import Path
import logging

import cv2
import numpy as np

from src.detectors.manager import DetectorManager
from src.detectors.onnx_detector import ONNXDetector
from src.detectors.yolo import YOLODetector
from src.cpu_profile import get_cpu_profile

logger = logging.getLogger("benchmark")


def build_detector(detector_type: str, model_path: str | None, batch_size: int, input_size: int):
    profile = get_cpu_profile()
    detector_type = (detector_type or "auto").lower()
    if detector_type == "onnx":
        model = model_path if model_path else str(profile.model_path)
        return ONNXDetector(model_path=model, input_size=input_size)
    elif detector_type == "yolo":
        # If a model path is provided, pass it as custom model
        return YOLODetector(model_path=model_path) if model_path else YOLODetector()
    else:
        mgr = DetectorManager(detector_type="auto")
        return mgr.get_detector()


def run_benchmark(detector, video_path: Path, frames: int, batch_size: int, warmup: int = 10, confidence: float = 0.25):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    processed = 0
    total_detections = 0
    classes_seen = set()

    # Warmup: read & drop a few frames to stabilize IO
    for _ in range(warmup):
        ret, _ = cap.read()
        if not ret:
            break

    frames_buffer = []
    infer_time = 0.0

    try:
        while processed < frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR -> RGB for detectors
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Use batch API if available
            if hasattr(detector, "detect_batch") and batch_size > 1:
                frames_buffer.append(frame_rgb)
                if len(frames_buffer) < batch_size:
                    continue

                t0 = time.perf_counter()
                results = detector.detect_batch(frames_buffer, None, confidence)
                t1 = time.perf_counter()
                infer_time += (t1 - t0)

                for res in results:
                    total_detections += len(res)
                    for d in res:
                        classes_seen.add(d.class_name)

                processed += len(frames_buffer)
                frames_buffer = []

            else:
                t0 = time.perf_counter()
                dets = detector.detect(frame_rgb, None, confidence)
                t1 = time.perf_counter()
                infer_time += (t1 - t0)

                total_detections += len(dets)
                for d in dets:
                    classes_seen.add(d.class_name)

                processed += 1

    finally:
        cap.release()

    fps = processed / infer_time if infer_time > 0 else 0.0

    return {
        "detector": detector.name,
        "processed_frames": processed,
        "total_infer_time_s": infer_time,
        "fps": fps,
        "total_detections": total_detections,
        "unique_classes": sorted(list(classes_seen)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to input video")
    ap.add_argument("--detector", choices=["auto", "onnx", "yolo"], default="auto")
    ap.add_argument("--model", help="Optional model path (ONNX or .pt)")
    ap.add_argument("--frames", type=int, default=500, help="Number of frames to process")
    _profile = get_cpu_profile()
    ap.add_argument("--batch-size", type=int, default=_profile.batch_size, help="Batch size for ONNX")
    ap.add_argument("--input-size", type=int, default=_profile.input_size, help="Input size for ONNX detector")
    ap.add_argument("--warmup", type=int, default=10, help="Warmup frames to skip")
    ap.add_argument("--confidence", type=float, default=0.25, help="Confidence threshold")
    ap.add_argument("--out", help="Output JSON file to save results")

    args = ap.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    detector = build_detector(args.detector, args.model, args.batch_size, args.input_size)
    logger.info(f"Using detector: {detector.name}")

    res = run_benchmark(detector, video_path, args.frames, args.batch_size, args.warmup, args.confidence)

    print(json.dumps(res, indent=2))

    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
    else:
        # Save to benchmarks/ with timestamp
        out_dir = Path("benchmarks")
        out_dir.mkdir(exist_ok=True)
        stamp = int(time.time())
        out_file = out_dir / f"benchmark_{detector.name.replace(' ', '_')}_{stamp}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
