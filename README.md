# vidfetch

Lightweight video object detection with comprehensive video-specific features and an intuitive web UI for search, detection, and correction workflows.

## Features

### Core Detection & Processing
- ✅ **Automatic CPU Profiling** - Dynamically benchmarks your hardware and auto-selects the optimal thread counts, batch sizes, and model tier.
- ✅ **Pluggable Detectors** - Supports wide-vocabulary YOLO (Object365/COCO) and ONNX models out of the box.
- ✅ **CPU-Optimized** - Designed exclusively to run fast on CPUs without needing a dedicated GPU.
- ✅ **Object Tracking** - Tracks objects across frames reliably using the SORT algorithm.
- ✅ **Temporal Aggregation** - Votes across adjacent frames to drastically reduce detection flickering.
- ✅ **Adaptive Sampling** - Automatically samples more frames during high-action motion and backs off during static scenes.
- ✅ **Scene Change Detection** - Naturally handles video cuts and transitions without breaking tracking context.

### Search & Query
- ✅ **Semantic Object Search** - Find objects by type with case-insensitive matching across all videos.
- ✅ **Color-Based Filtering** - Filter videos by warm/cool/dominant color characteristics.
- ✅ **Fast Index-Backed Queries** - Returns results in milliseconds using a persistent detection index.
- ✅ **Detection Reuse** - Automatically reuses existing detections instead of re-running jobs on already-indexed videos.

### Web UI & Workflows
- ✅ **Modern UI** - Built with Next.js 16, Tailwind CSS v4, and shadcn/ui components.
- ✅ **Video Gallery** - Browse, select, and manage videos with live detection status.
- ✅ **Detection Control** - Run detection on a single video or batch-process all videos with live progress tracking.
- ✅ **Review & Correction Tab** - Inspect detections frame-by-frame with per-class time-range summaries.
- ✅ **Relabel / Delete** - Correct misclassified detections and delete false positives.
- ✅ **Auto-Generated Rules** - Create reusable relabeling rules from corrections.

---

## Quick Start

### 1. Install dependencies
```bash
# Backend dependencies (we highly recommend using a virtual environment)
python -m venv .venv
.\.venv\Scripts\activate  # On Windows, or: source .venv/bin/activate (macOS/Linux)

pip install -r requirements.txt
pip install python-multipart  # Required for FastAPI form endpoints
```

### 2. Add videos to your library
Place any `.mp4`, `.avi`, or `.mov` files into the `data/` directory.
```bash
cp /path/to/videos/*.mp4 data/
```

### 3. Start the Backend API
The backend automatically schedules background indexing for any new videos found in `data/` upon boot.
```bash
python -m uvicorn api.main:app --reload --port 8000
```

### 4. Start the Frontend UI
The modern Web UI runs on port 3000 and connects to the backend at `http://localhost:8000`.
```bash
cd web
pnpm install  # or: npm install
pnpm dev      # or: npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## Using the Web UI

### Search for Objects
1. Click the **Filter** button to open the filter panel.
2. Select object types (e.g., "Person", "Car", "Building") and/or choose a color characteristic.
3. Click **Apply Filters** to refine the search.
4. Enter an object name in the search field (e.g., "person", "dog") and click **Fetch** to query.
5. Results show matching video segments sorted by relevance.

### Detect Objects in Videos
1. Browse the **Gallery** section and check the video(s) you want to analyze.
2. Click **Detect selected** to run detection on one video, or **Detect all** for batch processing.
3. Monitor the progress bar as the backend indexes detections.
4. Existing detections are automatically reused to avoid redundant re-processing.

### Review & Correct Detections
1. In the **Gallery**, click the **Review** button on any video card.
2. View **Object time ranges** at the top showing per-class temporal windows (e.g., "person: 0.4s-6.2s, 10.1s-12.5s").
3. Scroll through the timeline to inspect each detection.
4. Use **Relabel** to correct a mislabeled object, or **Delete** to remove false positives.
5. Switch to the **Corrections** tab to view all changes made to this video.
6. Use **Rules** tab to create reusable patterns (e.g., auto-relabel "dog" → "pet").

---

## Hardware-Aware Detection (Auto-Profiling)

Vidfetch is built to run anywhere without configuration. When you start the system, `src/cpu_profile.py` probes your machine's capabilities (Core Count, AVX2, AVX-512 extensions) and automatically assigns you to a performance tier:

- **High-Tier CPU** (8+ cores, AVX2/AVX-512): Sets batch size to 4, 640px input resolution, and uses the **Object365 model** (detects 365 unique object classes).
- **Medium-Tier CPU** (4+ cores, SSE4): Sets batch size to 2, 480px input resolution, and uses the Object365 model.
- **Low-Tier CPU** (< 4 cores): Sets batch size to 1, 320px input resolution, and falls back to a highly-quantized ONNX model or minimalist YOLOv8n.

---

## CLI Usage

If you want to use vidfetch headlessly without the web UI, use the CLI in the `scripts/` directory.

### Detect objects in a specific video
```bash
python -m scripts.detect_objects data/video.mp4
```

### Batch process all videos in data/
```bash
python -m scripts.detect_objects --all
```

### Override the auto-detector
```bash
python -m scripts.detect_objects data/video.mp4 --detector onnx
```

### Adjust confidence threshold and save results
```bash
python -m scripts.detect_objects data/video.mp4 --confidence 0.15 --output results.json
```

---

## API Endpoints

The API is served at `http://localhost:8000` by default.

### **Videos**

#### `GET /api/videos`
Lists all recognized video files in the `data/` directory.

**Response:**
```json
[
  {"id": "nature_walk.mp4", "name": "nature_walk.mp4"},
  {"id": "city_street.mp4", "name": "city_street.mp4"}
]
```

#### `GET /api/video/{video_id}`
Streams a specific video file directly to the client.

#### `GET /api/video/{video_id}/detections`
Retrieves cached detection results for a video.

**Response:**
```json
{
  "video": "nature_walk.mp4",
  "classes": ["person", "dog", "tree"],
  "timeline": [
    {
      "t": 0.0,
      "objects": [
        {"class": "person", "confidence": 0.92, "bbox": [10, 20, 100, 150]},
        {"class": "dog", "confidence": 0.87, "bbox": [120, 60, 200, 170]}
      ]
    }
  ]
}
```

### **Detection**

#### `GET /api/objects`
Returns a unified list of every unique object type detected across all indexed videos.

**Response:**
```json
["person", "car", "dog", "building", "bicycle", "tree"]
```

#### `GET /api/detector-info`
Returns machine hardware profile and active detector configuration.

**Response:**
```json
{
  "device": "CPU",
  "cpu_model": "Intel Core i7-9700K",
  "cores": 8,
  "has_avx2": true,
  "has_avx512": false,
  "active_detector": "yolo_object365",
  "batch_size": 4,
  "input_resolution": 640
}
```

#### `POST /api/detect`
Trigger detection on a video. Returns immediately with a job reference or reuse status.

**Payload:**
```json
{
  "video_id": "nature_walk.mp4",
  "detector_type": "auto"
}
```

**Response (new job queued):**
```json
{
  "success": true,
  "reused": false,
  "job_id": "abc123",
  "status_url": "/api/detect/jobs/abc123"
}
```

**Response (detection reused):**
```json
{
  "success": true,
  "reused": true,
  "video_id": "nature_walk.mp4",
  "message": "Using existing detection from index"
}
```

#### `GET /api/detect/jobs`
Lists all detection jobs (submitted, queued, running, completed, failed).

**Response:**
```json
[
  {"id": "abc123", "video_id": "nature_walk.mp4", "status": "running"},
  {"id": "def456", "video_id": "city_street.mp4", "status": "done"}
]
```

### **Search & Query**

#### `POST /api/query`
Search your video library by object types and color characteristics.

**Payload:**
```json
{
  "search_input": "person",
  "object_types": ["person", "dog"],
  "color_filter": "any",
  "color_ref_video_id": "",
  "k": 5
}
```

**Response:**
```json
{
  "query_object": "person",
  "results": [
    {
      "id": "nature_walk.mp4",
      "name": "nature_walk.mp4",
      "distance": 0.12,
      "object_segments": [
        {"start": 0.0, "end": 6.5},
        {"start": 10.1, "end": 15.3}
      ]
    }
  ],
  "time_ms": 48
}
```

### **Corrections & Rules**

#### `GET /api/corrections/{video_id}`
Get all corrections made to detections in a specific video.

#### `POST /api/corrections/{video_id}`
Add a correction (relabel or delete) for a detection.

**Payload:**
```json
{
  "video_id": "nature_walk.mp4",
  "frame_number": 5,
  "original_class": "dog",
  "corrected_class": "wolf",
  "action": "relabel"
}
```

#### `GET /api/rules`
List all correction rules.

#### `POST /api/rules`
Create a new correction rule.

**Payload:**
```json
{
  "pattern_class": "dog",
  "target_class": "pet",
  "confidence_threshold": 0.75
}
```

---

## System Architecture

```text
┌─────────────────────────────────────────────┐
│                 Frontend UI                  │
│  (Next.js 16, Tailwind v4, shadcn/ui)       │
│  • Search & Filter                          │
│  • Video Gallery & Detection                │
│  • Review & Correction Workflow             │
└─────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────┐
│                   API Layer                  │
│          (FastAPI - Port 8000)              │
│  • /api/query (search)                      │
│  • /api/detect (job submission)             │
│  • /api/objects (catalog)                   │
│  • /api/corrections (review workflow)       │
└─────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────┐
│           Detection Index Cache             │
│    (detection_results.json, thread-safe)    │
│  • Per-video detection results              │
│  • Timestamp & class metadata               │
│  • Enables reuse & fast queries             │
└─────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────┐
│              Hardware Profiler              │
│  (Detects CPU tier & selects model/batch)   │
└─────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────┐
│            Pluggable Detectors              │
│  ┌────────────┐ ┌────────────┐ ┌─────────┐ │
│  │ YOLO Ob365 │ │ YOLO COCO  │ │  ONNX   │ │
│  │ (primary)  │ │ (fallback) │ │ (fast)  │ │
│  └────────────┘ └────────────┘ └─────────┘ │
└─────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────┐
│          Video Processing Pipeline          │
│  ┌─────────┐ ┌────────────┐ ┌──────────┐   │
│  │  Scene  │ │ Adaptive   │ │ Temporal │   │
│  │ Change  │ │ Sampling   │ │Aggreg.  │   │
│  └─────────┘ └────────────┘ └──────────┘   │
└─────────────────────────────────────────────┘
```

---

## Project Directory Structure

```text
vidfetch/
├── api/
│   └── main.py                  # FastAPI backend & routes
├── data/                        # Your video files (mp4, avi, mov)
├── index_store/
│   └── detection_results.json   # Persistent detection index cache
├── models/                      # Pre-downloaded neural network weights
├── scripts/
│   ├── benchmark_cpu.py         # Test CPU inference speeds
│   ├── detect_objects.py        # Headless CLI detection tool
│   └── download_model.py        # Model fetcher utility
├── src/
│   ├── config.py                # Environment & runtime settings
│   ├── cpu_profile.py           # Hardware detection & tiering logic
│   ├── utils.py                 # Geometry (IoU), NMS, path utilities
│   ├── detector.py              # Core VideoDetector orchestrator
│   ├── video_catalog.py         # Video library management
│   ├── inference_worker.py      # Background job processing
│   ├── inference/
│   │   └── service.py           # Inference service & job queue
│   └── detectors/
│       ├── base.py              # Abstract detector interface
│       ├── manager.py           # Detector selection logic
│       ├── onnx_detector.py     # CPU-optimized ONNX runtime
│       ├── yolo.py              # Ultralytics YOLOv8 integration
│       ├── tracking.py          # SORT object tracking algorithm
│       └── motion.py            # Scene change & motion detection
└── web/                         # Next.js web application
    ├── app/                     # App routes & layout
    ├── components/
    │   ├── search/              # Search & filter components
    │   ├── ui/                  # shadcn/ui components
    │   ├── video-gallery.tsx    # Video gallery & review modal
    │   ├── correction-review.tsx # Review & correction tab
    │   └── result-card.tsx      # Search result cards
    ├── hooks/                   # React hooks (useVideoSearch, useCorrectionData)
    ├── lib/                     # API client & type definitions
    └── public/                  # Static assets
```

---

## Known Limitations & Future Improvements

- **Multi-GPU support**: Currently CPU-only; GPU support can be added via `torch.device` routing.
- **Streaming detection**: Large videos are processed end-to-end; windowed/streaming processing could reduce memory footprint.
- **Custom models**: Fine-tuning workflows are not yet built in; users can swap detector implementations.
- **Timeline scrolling**: Scrollable detection timeline added; infinite scroll in compact mode can be tuned.
- **Popover transparency**: Popover styling can be customized further (planned for later).

---

## Development Notes

### Running Frontend Build
```bash
cd web
pnpm build      # Production build
pnpm run dev    # Hot-reload dev server
pnpm run lint   # ESLint check
```

### Running Backend Tests (if available)
```bash
pytest tests/
```

### Environment Variables
Create a `.env` file in the project root if needed:
```env
API_BASE=http://localhost:8000
NEXT_PUBLIC_API_BASE=http://localhost:8000
```

---

## License

Vidfetch is provided as-is for research, evaluation, and personal use.