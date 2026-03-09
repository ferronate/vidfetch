# vidfetch — Lightweight Video Retrieval

Content-based video retrieval by **color** (HSV histograms) and **object**. **Recommended:** YOLOv8 (real COCO detection). Alternatives: MobileNet-SSD or CLIP. CPU-only; no GPU required.

---

## Setup

1. **Clone / enter project**
   ```bash
   cd vidfetch
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Optional — object search** (pick one or both)
   - **YOLO (recommended):** real object detection, COCO 80 classes (person, car, skateboard, etc.). Runs on CPU.
     ```bash
     pip install -r requirements-yolo.txt
     ```
   - **CLIP concepts (recommended with YOLO):** adds globe, fire, rain, lab, forest, snow, incense, papaya, etc. Install if you use `--add-clip-concepts` when building the object index.
     ```bash
     pip install -r requirements-clip.txt
     ```
   - **SSD (lightweight):** download detector once, then build with no extra deps. `python -m scripts.download_detector_model`

4. **Add videos**
   Put videos (e.g. `.mp4`, `.avi`, `.mov`) in `data/` (or another folder you’ll pass to the scripts).

---

## Run (step by step)

1. **Build the color index** (required for search)
   ```bash
   python -m scripts.build_index data
   ```
   Output: `index_store/` (features + meta).

2. **Build the object index** (optional; enables object search and gallery in the UI)
   - **Recommended (YOLO + CLIP concepts):** best coverage — COCO objects plus globe, fire, rain, lab, forest, snow, incense, surfboard, papaya, etc. Requires both `requirements-yolo.txt` and `requirements-clip.txt`.
     ```bash
     python -m scripts.build_object_index data --use-yolo --add-clip-concepts
     ```
   - YOLO only (no extra concepts):
     ```bash
     python -m scripts.build_object_index data --use-yolo
     ```
   - With SSD (after `download_detector_model`): `python -m scripts.build_object_index data`
   - With CLIP only: `python -m scripts.build_object_index data --use-clip`  
   Output: `index_store/objects.json`.

3. **Start the API** (from project root)
   ```bash
   python -m uvicorn api.main:app --reload --port 8000
   ```

4. **Start the web UI** (separate terminal)
   ```bash
   cd web
   npm install
   npm run dev
   ```

5. Open **http://localhost:3000**
   - **Gallery:** indexed videos appear at the top (no titles); browse before searching.
   - **Search:** type object terms (e.g. person, fire, surfboard) and press Enter or click Fetch videos; use the Filter panel for object types and color.
   - **Results:** show video name and friendly match labels (Best match, Very similar, Similar, Related); optional timeline bar when an object is found in the video.

---

## CLI (no UI)

- **Query by video path**
  ```bash
  python -m scripts.query data/clip1.mp4 --k 5
  ```
- **Evaluation** (precision@k, recall@k, retrieval time)
  ```bash
  python -m scripts.run_evaluation --index-dir index_store --k 5
  ```

---

## Script options (reference)

| Script | Main options |
|--------|----------------|
| `build_index` | `video_dir` (default `data`), `--index-dir`, `--fps`, `--max-frames` |
| `build_object_index` | `video_dir`, `--use-yolo`, `--add-clip-concepts`, `--yolo-model` (s\|n), `--min-frames`, `--confidence`, `--index-dir`, `--fps`, `--max-frames` |
| `query` | `query_video`, `--index-dir`, `--k` |
| `run_evaluation` | `--index-dir`, `--k` |

---

## Reducing false positives & filling COCO gaps

1. **YOLO (defaults):** confidence `0.5`, **min 2 frames** per class, and a **blacklist** (toilet, sports ball, cake, sandwich, etc.) so common false positives are never tagged. Model is YOLOv8 **small** (`--yolo-model s`); use `--yolo-model n` for faster but less accurate. Use `--min-frames 3` for stricter.
2. **Globe, fire, rain, lab, forest, snow, incense, papaya, etc.:** COCO doesn’t include these. Use **`--add-clip-concepts`** with `--use-yolo` (and `pip install -r requirements-clip.txt`). Concepts and synonyms are in `src/clip_detect.EXTRA_CONCEPTS` and `CONCEPT_TO_CANONICAL`.
3. **Stricter thresholds**  
   - **SSD:** default confidence is `0.6`. Rebuild with a higher value to be stricter:
     ```bash
     python -m scripts.build_object_index data --confidence 0.7
     ```
   - **CLIP:** default similarity threshold is `0.30`. Rebuild with a higher value:
     ```bash
     python -m scripts.build_object_index data --use-clip --clip-threshold 0.35
     ```
   Then rebuild the object index and restart the API so the UI uses the new index.

4. **Reference: COCO and pretrained models**  
   For stronger detection without training, use a model trained on **COCO** (Common Objects in Context), the standard benchmark for object detection:
   - **Dataset:** [COCO](https://cocodataset.org/) — 80 classes including person, dog, car; models trained on COCO are typically more reliable than older Pascal VOC models.
   - **Drop-in options (CPU, no training):** Pretrained COCO models in **ONNX** format run with ONNX Runtime on CPU. Examples: **YOLOX-Tiny** ([e.g. on Hugging Face](https://huggingface.co/cj-mills/yolox-coco-baseline-onnx)), **YOLOv3-Tiny**, or small **YOLOv8** exported to ONNX. You’d swap the detector in `src/detect.py` (or add a COCO/ONNX path) and map COCO class IDs to labels — no training, just a different pretrained model.

---

## Project layout

| Path | Role |
|------|------|
| `api/main.py` | FastAPI: list videos, run query, serve files |
| `web/` | Next.js + shadcn UI |
| `src/extract.py` | Frame sampling, HSV histograms, color presets |
| `src/index.py` | Color index, L2 search |
| `src/retrieval.py` | Load index, query helper |
| `src/yolo_detect.py` | **YOLOv8 (COCO)** — recommended object detection |
| `src/detect.py` | Object detection (OpenCV + MobileNet-SSD) |
| `src/clip_detect.py` | Optional CLIP detection (CPU) |
| `src/object_index.py` | Load/search object index, segments |
| `scripts/build_index.py` | Build color index |
| `scripts/build_object_index.py` | Build object index (`--use-yolo`, `--add-clip-concepts`, or SSD) |
| `scripts/download_detector_model.py` | Download SSD model to `models/` |

---

## Tips

- Use **5–10 short clips** in `data/` for fast indexing and queries.
- Run **build_index** (and optionally **build_object_index**) once, then start the API and web UI to demo in the browser.
