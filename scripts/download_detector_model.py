"""
Download MobileNet-SSD (Caffe) model files for object detection into models/.
Run once: python -m scripts.download_detector_model
"""
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
PROTOTXT_URL = "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/voc/MobileNetSSD_deploy.prototxt"
# Weights file (same repo; save as MobileNetSSD_deploy.caffemodel for detect.py)
CAFFEMODEL_URL = "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/mobilenet_iter_73000.caffemodel"


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    prototxt_path = MODELS_DIR / "MobileNetSSD_deploy.prototxt"
    caffemodel_path = MODELS_DIR / "MobileNetSSD_deploy.caffemodel"

    if prototxt_path.exists() and caffemodel_path.exists():
        print("Model files already present in models/")
        return

    if not prototxt_path.exists():
        print(f"Downloading prototxt to {prototxt_path} ...")
        try:
            urllib.request.urlretrieve(PROTOTXT_URL, prototxt_path)
            print("  Done.")
        except Exception as e:
            print(f"  Failed: {e}")
            sys.exit(1)

    if not caffemodel_path.exists():
        print(f"Downloading caffemodel to {caffemodel_path} (~23MB) ...")
        try:
            urllib.request.urlretrieve(CAFFEMODEL_URL, caffemodel_path)
            print("  Done.")
        except Exception as e:
            print(f"  Failed: {e}")
            print("  You can download manually from:")
            print("  https://github.com/chuanqi305/MobileNet-SSD")
            print("  Save as models/MobileNetSSD_deploy.caffemodel")
            sys.exit(1)

    print("Object detector model ready.")


if __name__ == "__main__":
    main()
