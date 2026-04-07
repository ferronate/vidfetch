#!/usr/bin/env python3
"""
Simple object detection for videos.
Usage: python -m scripts.detect_objects [video_path] [options]

Examples:
    python -m scripts.detect_objects data/video.mp4
    python -m scripts.detect_objects data/video.mp4 --detector yolo-world
    python -m scripts.detect_objects data/video.mp4 --prompt "person, car, dog"
    python -m scripts.detect_objects --all
"""
import argparse
import json
import sys
import concurrent.futures
from pathlib import Path
from typing import Any

from src.detector import detect_objects
from src.utils import VIDEO_EXTENSIONS, calculate_iou

FALLBACK_FPS = 0.5
FALLBACK_WORLD_CONFIDENCE = 0.12

DOMAIN_PROMPT_PRESETS: dict[str, str] = {
    "nature": (
        "tree, forest, mountain, hill, river, waterfall, lake, ocean, beach, sky, cloud, "
        "flower, grass, field, rock, snow, bird, fish, deer, bear"
    ),
    "food": (
        "food, meal, fruit, vegetable, plate, bowl, sandwich, pizza, burger, sushi, "
        "salad, bread, drink, cup, bottle"
    ),
    "nature-food": (
        "tree, forest, mountain, hill, river, waterfall, lake, ocean, beach, sky, cloud, "
        "flower, grass, field, rock, snow, bird, fish, deer, bear, "
        "food, meal, fruit, vegetable, plate, bowl, sandwich, pizza, burger, sushi, "
        "salad, bread, drink, cup, bottle"
    ),
}


def find_videos(directory: Path) -> list[Path]:
    """Find all video files in directory."""
    return [p for p in directory.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS]


def _parse_detector_list(detectors_arg: str | None) -> list[str]:
    if not detectors_arg:
        return []
    return [d.strip() for d in detectors_arg.split(",") if d.strip()]


def _domain_prompt(domain: str | None) -> str | None:
    if not domain:
        return None
    if domain == "none":
        return None
    return DOMAIN_PROMPT_PRESETS.get(domain)


def _is_low_yield_result(result: dict[str, Any], max_classes: int, max_timeline_frames: int) -> bool:
    classes_count = len(result.get("classes", []))
    timeline_frames = int(
        result.get("pipeline_stats", {}).get("timeline_frames", result.get("total_detections", 0)) or 0
    )
    return classes_count <= max_classes or timeline_frames <= max_timeline_frames


def _merge_multi_detector_results(results_by_detector: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Merge detector outputs into one accumulated result with source provenance."""
    detector_names = list(results_by_detector.keys())
    first = results_by_detector[detector_names[0]]

    timeline_by_ts: dict[float, list[dict[str, Any]]] = {}
    class_union: set[str] = set()
    pipeline_totals = {
        # Keep frame-domain totals from the video itself, not summed per model.
        "video_total_frames": int(first.get("pipeline_stats", {}).get("video_total_frames", 0) or 0),
        "sampled_frames": 0,
        "raw_detection_frames": 0,
        "raw_detection_objects": 0,
        "tracked_detection_frames": 0,
        "tracked_detection_objects": 0,
        "aggregated_detection_frames": 0,
        "aggregated_detection_objects": 0,
        "timeline_frames": 0,
    }

    for detector_name, result in results_by_detector.items():
        for c in result.get("classes", []):
            class_union.add(c)

        stats = result.get("pipeline_stats", {})
        for k in (
            "sampled_frames",
            "raw_detection_frames",
            "raw_detection_objects",
            "tracked_detection_frames",
            "tracked_detection_objects",
            "aggregated_detection_frames",
            "aggregated_detection_objects",
        ):
            pipeline_totals[k] += int(stats.get(k, 0) or 0)

        for entry in result.get("timeline", []):
            ts = float(entry.get("t", 0.0))
            if ts not in timeline_by_ts:
                timeline_by_ts[ts] = []

            for obj in entry.get("objects", []):
                merged_obj = dict(obj)
                merged_obj["source_detector"] = detector_name
                timeline_by_ts[ts].append(merged_obj)

    def dedup_frame_objects(frame_objects: list[dict[str, Any]], iou_threshold: float = 0.55) -> list[dict[str, Any]]:
        deduped: list[dict[str, Any]] = []
        for obj in sorted(frame_objects, key=lambda o: float(o.get("confidence", 0.0)), reverse=True):
            bbox = obj.get("bbox")
            cls = obj.get("class")
            if not bbox or len(bbox) != 4:
                deduped.append(obj)
                continue

            is_duplicate = False
            for existing in deduped:
                existing_bbox = existing.get("bbox")
                if not existing_bbox or len(existing_bbox) != 4:
                    continue
                if existing.get("class") != cls:
                    continue
                if calculate_iou(bbox, existing_bbox) >= iou_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                deduped.append(obj)
        return deduped

    merged_timeline = [
        {"t": t, "objects": dedup_frame_objects(timeline_by_ts[t])}
        for t in sorted(timeline_by_ts.keys())
    ]
    pipeline_totals["timeline_frames"] = len(merged_timeline)

    merged_detector_info = dict(first.get("detector", {}))
    merged_detector_info["name"] = "Ensemble (accumulated)"
    merged_detector_info["ensemble_detectors"] = detector_names
    merged_detector_info["merge_strategy"] = "union"

    return {
        "video": first.get("video"),
        "detector": merged_detector_info,
        "pipeline_stats": pipeline_totals,
        "classes": sorted(class_union),
        "total_classes": len(class_union),
        "timeline": merged_timeline,
        "total_detections": len(merged_timeline),
        "prompt": first.get("prompt"),
    }


def process_single_video(video_path: str, detector_type: str, confidence: float,
                         sample_fps: float | None, prompt: str,
                         detector_types: list[str] | None = None,
                         domain: str | None = None,
                         fallback_world: bool = False,
                         fallback_max_classes: int = 3,
                         fallback_max_timeline_frames: int = 8) -> tuple:
    """Process a single video and return results. Must be top-level for pickling."""
    domain_prompt = _domain_prompt(domain)
    effective_prompt = prompt or domain_prompt

    if detector_types:
        results_by_detector: dict[str, dict[str, Any]] = {}
        for d in detector_types:
            detector_prompt = effective_prompt if d == "yolo-world" else prompt
            results_by_detector[d] = detect_objects(
                video_path=video_path,
                detector_type=d,
                confidence=confidence,
                sample_fps=sample_fps,
                prompt=detector_prompt,
            )

        if (
            fallback_world
            and "yolo-world" not in detector_types
            and _is_low_yield_result(
                _merge_multi_detector_results(results_by_detector),
                fallback_max_classes,
                fallback_max_timeline_frames,
            )
        ):
            fallback_prompt = effective_prompt
            if fallback_prompt:
                results_by_detector["yolo-world-fallback"] = detect_objects(
                    video_path=video_path,
                    detector_type="yolo-world",
                    confidence=min(confidence, FALLBACK_WORLD_CONFIDENCE),
                    sample_fps=sample_fps,
                    prompt=fallback_prompt,
                )

        results = _merge_multi_detector_results(results_by_detector)
        if "yolo-world-fallback" in results_by_detector:
            results["fallback_world_applied"] = True
            results["fallback_world_reason"] = "low-yield initial result"
            results["fallback_prompt"] = effective_prompt
        return video_path, results

    results = detect_objects(
        video_path=video_path,
        detector_type=detector_type,
        confidence=confidence,
        sample_fps=sample_fps,
        prompt=effective_prompt if detector_type == "yolo-world" else prompt,
    )

    if (
        fallback_world
        and detector_type != "yolo-world"
        and _is_low_yield_result(results, fallback_max_classes, fallback_max_timeline_frames)
        and effective_prompt
    ):
        fallback_world_result = detect_objects(
            video_path=video_path,
            detector_type="yolo-world",
            confidence=min(confidence, FALLBACK_WORLD_CONFIDENCE),
            sample_fps=sample_fps,
            prompt=effective_prompt,
        )
        results = _merge_multi_detector_results(
            {
                detector_type: results,
                "yolo-world-fallback": fallback_world_result,
            }
        )
        results["fallback_world_applied"] = True
        results["fallback_world_reason"] = "low-yield initial result"
        results["fallback_prompt"] = effective_prompt

    return video_path, results


def main():
    parser = argparse.ArgumentParser(
        description="Detect objects in videos using pluggable detector architecture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("video", nargs="?", help="Path to video file")
    parser.add_argument("--all", action="store_true",
                        help="Process all videos in 'data' directory")
    parser.add_argument("--detector", "-d", default="auto",
                        choices=["auto", "onnx", "yolo-world", "yolo", "legacy"],
                        help="Detector type (default: auto)")
    parser.add_argument("--detectors", default=None,
                        help="Comma-separated detector list to run and accumulate (e.g., yolo,yolo-world,onnx)")
    parser.add_argument("--profile", default="balanced",
                        choices=["balanced", "high-recall", "high-precision"],
                        help="Detection profile preset (default: balanced)")
    parser.add_argument("--confidence", "-c", type=float, default=None,
                        help="Detection confidence threshold 0.0-1.0 (default from --profile)")
    parser.add_argument("--fps", type=float, default=None,
                        help="Frames per second to sample (default: auto from CPU profile)")
    parser.add_argument("--prompt", "-p", type=str,
                        help="Text prompt for detection (e.g., 'person, car, dog')")
    parser.add_argument("--domain", default="none",
                        choices=["none", "nature", "food", "nature-food"],
                        help="Domain prompt preset for yolo-world or fallback-world")
    parser.add_argument("--fallback-world", action="store_true",
                        help="Run a yolo-world fallback pass when initial detections are low-yield")
    parser.add_argument("--fallback-max-classes", type=int, default=3,
                        help="Fallback trigger: max unique classes in initial result (default: 3)")
    parser.add_argument("--fallback-max-timeline", type=int, default=8,
                        help="Fallback trigger: max timeline frames in initial result (default: 8)")
    parser.add_argument("--output", "-o", help="Save results to JSON file")

    args = parser.parse_args()

    profile_defaults = {
        "balanced": {"confidence": 0.25, "fps": None},
        "high-recall": {"confidence": 0.15, "fps": 2.0},
        "high-precision": {"confidence": 0.35, "fps": 1.0},
    }

    if args.confidence is None:
        args.confidence = profile_defaults[args.profile]["confidence"]
    if args.fps is None and profile_defaults[args.profile]["fps"] is not None:
        args.fps = profile_defaults[args.profile]["fps"]

    detector_types = _parse_detector_list(args.detectors)
    domain_prompt = _domain_prompt(args.domain)
    selected_detector = args.detector
    if args.domain != "none" and args.detector == "auto" and not detector_types:
        selected_detector = "yolo-world"
        print("Info: --domain with --detector auto defaults to yolo-world for better open-vocabulary coverage.")
    if args.prompt and args.domain != "none":
        print("Info: --prompt is set; it takes precedence over --domain preset.")
    if args.fallback_world and not (args.prompt or domain_prompt):
        print("Warning: --fallback-world enabled without --prompt/--domain; fallback may be skipped.")

    if args.fps is not None and args.fps <= 0:
        print("Error: --fps must be greater than 0")
        sys.exit(1)

    # Handle --all option
    if args.all:
        videos = find_videos(Path("data"))
        if not videos:
            print("No videos found in 'data' directory")
            sys.exit(1)
        print(f"Found {len(videos)} videos to process\n")
    elif args.video:
        video_path = Path(args.video)
        if not video_path.exists():
            print(f"Error: Video not found: {video_path}")
            sys.exit(1)
        videos = [video_path]
    else:
        print("Error: Please provide a video path or use --all")
        sys.exit(1)

    all_results = []
    failed_videos = []
    recovered_count = 0

    try:
        if args.all and len(videos) > 1:
            # Parallel processing for multiple videos
            max_workers = min(len(videos), 4)  # Cap at 4 workers to avoid OOM
            print(f"Processing {len(videos)} videos with {max_workers} parallel workers...\n")
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_video = {
                    executor.submit(
                        process_single_video,
                        str(vp),
                        selected_detector,
                        args.confidence,
                        args.fps,
                        args.prompt,
                        detector_types,
                        args.domain,
                        args.fallback_world,
                        args.fallback_max_classes,
                        args.fallback_max_timeline,
                    ): vp 
                    for vp in videos
                }
                
                for i, future in enumerate(concurrent.futures.as_completed(future_to_video)):
                    video_path = future_to_video[future]
                    try:
                        _, results = future.result()
                        all_results.append(results)
                        print(f"\n[{i+1}/{len(videos)}] ✓ {video_path.name}")
                        print(f"  Detector: {results['detector']['name']}")
                        effective_fps = results['detector'].get('effective_fps', 'unknown')
                        print(f"  Sampling FPS: {effective_fps}")
                        stats = results.get("pipeline_stats", {})
                        sampled = stats.get("sampled_frames", 0)
                        timeline_frames = stats.get("timeline_frames", 0)
                        print(f"  Frames sampled/timeline: {sampled}/{timeline_frames}")
                        if results.get("fallback_world_applied"):
                            print("  Fallback: yolo-world pass applied")
                        print(f"  Objects: {', '.join(results['classes']) if results['classes'] else 'None'}")
                    except Exception as e:
                        print(f"\n[{i+1}/{len(videos)}] ✗ {video_path.name}: {e}")

                        # If FPS was auto-selected, retry once with a safe fallback.
                        if args.fps is None:
                            try:
                                print(f"  Retrying with fallback sampling: {FALLBACK_FPS} FPS")
                                _, retry_results = process_single_video(
                                    str(video_path),
                                    selected_detector,
                                    args.confidence,
                                    FALLBACK_FPS,
                                    args.prompt,
                                    None,
                                    args.domain,
                                    args.fallback_world,
                                    args.fallback_max_classes,
                                    args.fallback_max_timeline,
                                )
                                all_results.append(retry_results)
                                recovered_count += 1
                                print(f"  ✓ Recovered: {video_path.name}")
                                print(f"  Detector: {retry_results['detector']['name']}")
                                effective_fps = retry_results['detector'].get('effective_fps', 'unknown')
                                print(f"  Sampling FPS: {effective_fps}")
                                stats = retry_results.get("pipeline_stats", {})
                                sampled = stats.get("sampled_frames", 0)
                                timeline_frames = stats.get("timeline_frames", 0)
                                print(f"  Frames sampled/timeline: {sampled}/{timeline_frames}")
                                if retry_results.get("fallback_world_applied"):
                                    print("  Fallback: yolo-world pass applied")
                                print(
                                    "  Objects: "
                                    f"{', '.join(retry_results['classes']) if retry_results['classes'] else 'None'}"
                                )
                                continue
                            except Exception as retry_error:
                                print(f"  Retry failed: {retry_error}")

                        failed_videos.append(video_path.name)
        else:
            # Single video - process directly
            video_path = videos[0]
            print(f"Detecting objects in: {video_path.name}")
            if detector_types:
                print(f"Detectors: {', '.join(detector_types)}")
            else:
                print(f"Detector: {selected_detector}")
            print(f"Profile: {args.profile}")
            print(f"Confidence: {args.confidence}")
            if args.fps:
                print(f"Sampling: {args.fps} FPS")
            else:
                print("Sampling: auto (based on CPU profile)")
            if args.prompt:
                print(f"Prompt: {args.prompt}")
            elif args.domain != "none":
                print(f"Domain preset: {args.domain}")
            if args.fallback_world:
                print(
                    "Fallback-world: enabled "
                    f"(classes<={args.fallback_max_classes} or timeline<={args.fallback_max_timeline})"
                )
            print()

            if detector_types:
                _, results = process_single_video(
                    str(video_path),
                    selected_detector,
                    args.confidence,
                    args.fps,
                    args.prompt,
                    detector_types,
                    args.domain,
                    args.fallback_world,
                    args.fallback_max_classes,
                    args.fallback_max_timeline,
                )
            else:
                _, results = process_single_video(
                    str(video_path),
                    selected_detector,
                    args.confidence,
                    args.fps,
                    args.prompt,
                    None,
                    args.domain,
                    args.fallback_world,
                    args.fallback_max_classes,
                    args.fallback_max_timeline,
                )

            all_results.append(results)

            # Print results
            print(f"\n{'='*60}")
            print("RESULTS")
            print(f"{'='*60}")
            print(f"Video: {results['video']}")
            print(f"Detector: {results['detector']['name']}")
            effective_fps = results['detector'].get('effective_fps', 'unknown')
            print(f"Effective Sampling FPS: {effective_fps}")
            print(f"CPU Tier: {results['detector']['cpu_tier']}")
            print(f"CPU: {results['detector'].get('cpu_brand', 'unknown')}")
            stats = results.get("pipeline_stats", {})
            print(f"Video frames total: {stats.get('video_total_frames', 'unknown')}")
            print(f"Frames sampled: {stats.get('sampled_frames', 0)}")
            print(f"Frames with raw detections: {stats.get('raw_detection_frames', 0)}")
            print(f"Frames after tracking: {stats.get('tracked_detection_frames', 0)}")
            print(f"Frames after aggregation: {stats.get('aggregated_detection_frames', 0)}")
            print(f"Timeline frames kept: {stats.get('timeline_frames', results['total_detections'])}")
            if results.get("fallback_world_applied"):
                print(f"Fallback-world: applied ({results.get('fallback_world_reason', 'low-yield initial result')})")
            print(f"Unique objects found: {len(results['classes'])}")
            print(f"\nObjects detected:")
            for obj in results["classes"]:
                print(f"  • {obj}")

            if results["timeline"] and not args.all:
                print(f"\nFirst few detections:")
                for entry in results["timeline"][:5]:
                    objs = [d["class"] for d in entry["objects"]]
                    print(f"  [{entry['t']:.1f}s] {', '.join(objs)}")

            print(f"{'='*60}")

        # Save to file if requested
        if args.output:
            output_path = Path(args.output)
            data = all_results if args.all else all_results[0]
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"\nResults saved to: {output_path}")

        # Print summary for --all
        if args.all and len(all_results) > 1:
            print(f"\n{'='*60}")
            print("SUMMARY")
            print(f"{'='*60}")
            print(f"Total videos requested: {len(videos)}")
            print(f"Total videos processed: {len(all_results)}")
            print(f"Recovered by retry: {recovered_count}")
            print(f"Failed videos: {len(failed_videos)}")
            if failed_videos:
                print(f"Failed list: {', '.join(failed_videos)}")
            all_unique_objects = set()
            for r in all_results:
                all_unique_objects.update(r["classes"])
            print(f"Total unique objects across all videos: {len(all_unique_objects)}")
            print(f"All objects: {', '.join(sorted(all_unique_objects))}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()