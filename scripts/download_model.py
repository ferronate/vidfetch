#!/usr/bin/env python3
"""
Download models for vidfetch object detection.
Usage: python -m scripts.download_model [options]

Examples:
    python -m scripts.download_model                    # Download default model
    python -m scripts.download_model --model yolov8n   # Download specific model
    python -m scripts.download_model --all              # Download all models
    python -m scripts.download_model --list             # List available models
"""
import argparse
import sys
from pathlib import Path
import requests
from tqdm import tqdm

# Model URLs - ONNX models from ultralytics releases
ONNX_MODEL_URLS = {
    "yolov8n.onnx": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.onnx",
    "yolov8s.onnx": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.onnx",
    "yolov8m.onnx": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.onnx",
}

# Model URLs - PyTorch .pt models from ultralytics releases
PT_MODEL_URLS = {
    "yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt",
    "yolov8s.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt",
    "yolov8m.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt",
    "yolov8n-oiv7.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-oiv7.pt",
    "yolov8s-world.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-world.pt",
    "yolov8s-worldv2.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-worldv2.pt",
    "yolo11n.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
}

# Combined model URLs
ALL_MODEL_URLS = {**ONNX_MODEL_URLS, **PT_MODEL_URLS}

# Model descriptions
MODEL_DESCRIPTIONS = {
    "yolov8n.onnx": "YOLOv8-nano ONNX (80 classes, CPU-optimized)",
    "yolov8s.onnx": "YOLOv8-small ONNX (80 classes, CPU-optimized)",
    "yolov8m.onnx": "YOLOv8-medium ONNX (80 classes, CPU-optimized)",
    "yolov8n.pt": "YOLOv8-nano PyTorch (80 classes COCO)",
    "yolov8s.pt": "YOLOv8-small PyTorch (80 classes COCO)",
    "yolov8m.pt": "YOLOv8-medium PyTorch (80 classes COCO)",
    "yolov8n-oiv7.pt": "YOLOv8-nano Open Images V7 (600 classes)",
    "yolov8s-world.pt": "YOLOv8-small World (open-vocabulary, unlimited classes)",
    "yolov8s-worldv2.pt": "YOLOv8-small World v2 (open-vocabulary, unlimited classes)",
    "yolo11n.pt": "YOLO11-nano PyTorch (80 classes COCO)",
}

# Recommended models by CPU tier
TIER_RECOMMENDATIONS = {
    "high": ["yolo11n.pt", "yolov8s-world.pt", "yolov8n-oiv7.pt"],
    "medium": ["yolo11n.pt", "yolov8s-world.pt"],
    "low": ["yolov8n.onnx", "yolov8n.pt"],
}

# Priority order for auto-selection (best to fallback per tier)
MODEL_PRIORITY = {
    "high": [
        "yolo11n.pt",           # Latest YOLO11
        "yolov8s-world.pt",     # Open-vocabulary
        "yolov8n-oiv7.pt",      # 600 classes
        "yolov8s.pt",           # Small COCO
        "yolov8n.pt",           # Nano COCO
    ],
    "medium": [
        "yolo11n.pt",
        "yolov8s-world.pt",
        "yolov8n.pt",
        "yolov8n.onnx",         # ONNX fallback
    ],
    "low": [
        "yolov8n.onnx",         # ONNX is fastest on low-tier
        "yolov8n.pt",           # PyTorch fallback
    ],
}


def get_available_models(models_dir: Path) -> list[str]:
    """Get list of models already downloaded."""
    if not models_dir.exists():
        return []
    return [f.name for f in models_dir.iterdir() if f.suffix in ['.pt', '.onnx']]


def select_best_model(models_dir: Path, cpu_tier: str = "medium") -> str | None:
    """
    Select the best available model based on what's already downloaded.
    Returns the best model name, or None if no models are available.
    """
    available = get_available_models(models_dir)
    if not available:
        return None
    
    # Get priority list for this CPU tier
    priority = MODEL_PRIORITY.get(cpu_tier, MODEL_PRIORITY["medium"])
    
    # Return the highest priority model that exists
    for model_name in priority:
        if model_name in available:
            return model_name
    
    # If no priority models found, return any available model
    return available[0] if available else None


def download_file(url: str, destination: Path, description: str = ""):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as f:
        with tqdm(
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            desc=description
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))


def download_model(model_name: str, models_dir: Path) -> bool:
    """Download a specific model."""
    if model_name not in ALL_MODEL_URLS:
        print(f"Error: Unknown model '{model_name}'")
        print(f"Available models: {', '.join(ALL_MODEL_URLS.keys())}")
        return False
    
    url = ALL_MODEL_URLS[model_name]
    model_path = models_dir / model_name
    
    # Check if already exists
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"✓ Model already exists: {model_path} ({size_mb:.1f} MB)")
        return True
    
    # Create models directory
    models_dir.mkdir(exist_ok=True)
    
    desc = MODEL_DESCRIPTIONS.get(model_name, model_name)
    print(f"\nDownloading {model_name}...")
    print(f"  {desc}")
    print(f"  URL: {url}")
    print(f"  Destination: {model_path}")
    
    try:
        download_file(url, model_path, model_name)
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"✓ Successfully downloaded {model_name} ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"✗ Error downloading {model_name}: {e}")
        if model_path.exists():
            model_path.unlink()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download models for vidfetch object detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m scripts.download_model --list                    # List all models
  python -m scripts.download_model --model yolov8n.pt        # Download specific model
  python -m scripts.download_model --recommended             # Download for your CPU tier
  python -m scripts.download_model --all                     # Download all models
        """
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="Specific model to download (e.g., yolov8n.pt, yolov8s-world.pt)"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Download all available models"
    )
    parser.add_argument(
        "--recommended", "-r",
        action="store_true",
        help="Download models recommended for your CPU tier"
    )
    parser.add_argument(
        "--dir", "-d",
        default="models",
        help="Directory to save models (default: models)"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available models"
    )
    
    args = parser.parse_args()
    models_dir = Path(args.dir)
    
    # List mode
    if args.list:
        print("\nAvailable models for vidfetch:\n")
        print("ONNX Models (CPU-optimized):")
        for name, desc in MODEL_DESCRIPTIONS.items():
            if name.endswith('.onnx'):
                print(f"  {name:25} - {desc}")
        print("\nPyTorch Models (.pt):")
        for name, desc in MODEL_DESCRIPTIONS.items():
            if name.endswith('.pt'):
                print(f"  {name:25} - {desc}")
        print("\nModels in registry (src/cpu_profile.py):")
        print("  object365          - yolo11n_object365.pt (365 classes)")
        print("  coco-nano          - yolov8n.pt (80 classes)")
        print("  coco-small         - yolov8s.pt (80 classes)")
        print("  world-small        - yolov8s-world.pt (open-vocabulary)")
        print("  coco-oiv7-nano     - yolov8n-oiv7.pt (600 classes)")
        print("  onnx-nano          - yolov8n.onnx (80 classes)")
        print("  onnx-nano-quant    - yolov8n.quant.onnx (80 classes, INT8)")
        return
    
    # Download recommended models for CPU tier
    if args.recommended:
        try:
            from src.cpu_profile import get_cpu_profile
            profile = get_cpu_profile()
            tier = profile.tier
            recommended = TIER_RECOMMENDATIONS.get(tier, [])
            print(f"Detected CPU tier: {tier}")
            print(f"Recommended models: {', '.join(recommended)}")
            
            success_count = 0
            for model_name in recommended:
                if download_model(model_name, models_dir):
                    success_count += 1
            print(f"\nDownloaded {success_count}/{len(recommended)} recommended models")
        except Exception as e:
            print(f"Error detecting CPU tier: {e}")
            print("Falling back to downloading all models...")
            args.all = True
    
    # Download all models
    if args.all:
        print(f"Downloading all models to {models_dir}/")
        success_count = 0
        for model_name in ALL_MODEL_URLS:
            if download_model(model_name, models_dir):
                success_count += 1
        print(f"\nDownloaded {success_count}/{len(ALL_MODEL_URLS)} models")
    elif args.model:
        download_model(args.model, models_dir)
    else:
        # Auto-select best model
        try:
            from src.cpu_profile import get_cpu_profile
            profile = get_cpu_profile()
            cpu_tier = profile.tier
        except Exception:
            cpu_tier = "medium"
        
        best_model = select_best_model(models_dir, cpu_tier)
        
        if best_model:
            print(f"✓ Found existing model: {best_model}")
            print(f"  Using this model for detection.")
            return
        else:
            # No models found, download the best one for this tier
            priority = MODEL_PRIORITY.get(cpu_tier, MODEL_PRIORITY["medium"])
            best_to_download = priority[0] if priority else "yolov8n.pt"
            
            print(f"Detected CPU tier: {cpu_tier}")
            print(f"No models found in {models_dir}/")
            print(f"Downloading best model for your CPU: {best_to_download}")
            
            if download_model(best_to_download, models_dir):
                print(f"\n✓ Ready to use! Run detection with:")
                print(f"  python -m scripts.detect_objects --all")


if __name__ == "__main__":
    main()