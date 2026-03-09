"""
CLIP-based "what's in the frame" detection. Uses a small CLIP model (ViT-B-32) on CPU
to score frame similarity to text labels—better semantic understanding than a fixed detector.
Install: pip install -r requirements-clip.txt
Runs on CPU only (no GPU).
"""
from pathlib import Path
import os
import numpy as np

# Force CPU-only (must be set before torch is imported)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Same labels as Pascal VOC for compatibility with existing index/API
CLIP_LABELS = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "dining table", "dog", "horse", "motorbike", "person",
    "potted plant", "sheep", "sofa", "train", "tv monitor",
]

# Balance recall vs false positives. Raise with --clip-threshold if you get bad hits.
DEFAULT_THRESHOLD = 0.24


def _load_clip_cpu():
    """Load CLIP model and preprocess, on CPU."""
    import torch
    import open_clip
    from PIL import Image

    device = "cpu"
    # OpenAI weights expect QuickGELU; without this, embeddings are wrong and detection fails.
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai", force_quick_gelu=True
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model = model.to(device).eval()
    return model, preprocess, tokenizer, device


def _encode_labels(model, tokenizer, labels: list[str], device: str):
    """Precompute normalized text embeddings for all labels."""
    import torch

    with torch.no_grad():
        text = tokenizer(labels)
        text_t = text.to(device)
        text_features = model.encode_text(text_t)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features


def _frame_to_labels(
    frame_bgr: np.ndarray,
    model,
    preprocess,
    text_features,
    labels: list[str],
    device: str,
    threshold: float = DEFAULT_THRESHOLD,
) -> list[str]:
    """Run CLIP on one BGR frame; return list of labels whose similarity >= threshold."""
    import torch
    import cv2
    from PIL import Image

    # OpenCV BGR -> PIL RGB
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    img_t = preprocess(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(img_t)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        sim = (image_features @ text_features.T).cpu().numpy().ravel()
    return [labels[i] for i in range(len(labels)) if sim[i] >= threshold]


def video_to_object_timeline_clip(
    video_path: str | Path,
    labels: list[str] | None = None,
    fps: float = 0.5,
    max_frames: int = 30,
    threshold: float = DEFAULT_THRESHOLD,
):
    """
    Use CLIP to get per-frame "what's in the frame" for the given labels.
    Returns (set of all detected classes, list of {t, objects}) — same format as detect.video_to_object_timeline.
    Runs on CPU only.
    """
    from .extract import sample_frames_with_time

    try:
        model, preprocess, tokenizer, device = _load_clip_cpu()
    except Exception as e:
        raise RuntimeError(
            "CLIP not available. Install with: pip install open-clip-torch torch pillow"
        ) from e

    labels = labels or CLIP_LABELS
    text_features = _encode_labels(model, tokenizer, labels, device)

    frames_with_time = sample_frames_with_time(video_path, fps=fps, max_frames=max_frames)
    classes = set()
    timeline = []
    for frame, t_sec in frames_with_time:
        names = _frame_to_labels(
            frame, model, preprocess, text_features, labels, device, threshold
        )
        for name in names:
            classes.add(name)
        timeline.append({"t": round(t_sec, 2), "objects": names})
    return classes, timeline


# Extra concepts not in COCO: use with --add-clip-concepts when building with YOLO
EXTRA_CONCEPTS = [
    "globe", "fire", "flames", "fireplace", "burning", "campfire", "incense",
    "surfboard", "skateboard", "cooking", "ocean", "beach",
    "bread", "flatbread", "sand", "waves", "surfing",
    "rain", "rainy", "leaves", "droplets",
    "lab", "computer", "research",
    "forest", "snow", "winter", "trees",
    "papaya",
]

# Map synonyms to one canonical label for the index (so "fire" search finds woodfire, etc.)
CONCEPT_TO_CANONICAL = {
    "flames": "fire", "fireplace": "fire", "burning": "fire", "campfire": "fire",
    "rainy": "rain", "droplets": "rain",
    "computer": "lab", "research": "lab",
    "winter": "snow", "trees": "forest",
}


def video_to_extra_concepts_clip(
    video_path: str | Path,
    prompts: list[str] | None = None,
    fps: float = 0.5,
    max_frames: int = 30,
    threshold: float = 0.26,
    min_frames_per_class: int = 2,
) -> tuple[set[str], list[dict]]:
    """
    Run CLIP with custom prompts (e.g. globe, fire, surfboard). Returns (set of matching concepts, timeline).
    Only concepts that appear in >= min_frames_per_class frames are included.
    Use with YOLO to fill COCO gaps. Requires CLIP installed.
    """
    from .extract import sample_frames_with_time

    prompts = prompts or EXTRA_CONCEPTS
    try:
        model, preprocess, tokenizer, device = _load_clip_cpu()
    except Exception as e:
        raise RuntimeError("CLIP not available. pip install -r requirements-clip.txt") from e

    text_features = _encode_labels(model, tokenizer, prompts, device)
    frames_with_time = sample_frames_with_time(video_path, fps=fps, max_frames=max_frames)
    concept_frame_count = {}
    timeline = []
    for frame, t_sec in frames_with_time:
        names = _frame_to_labels(frame, model, preprocess, text_features, prompts, device, threshold)
        for name in names:
            concept_frame_count[name] = concept_frame_count.get(name, 0) + 1
        timeline.append({"t": round(t_sec, 2), "objects": names})
    # Merge synonym counts into canonical (e.g. flames+fireplace+burning -> fire)
    canonical_count = {}
    for c, n in concept_frame_count.items():
        canonical = CONCEPT_TO_CANONICAL.get(c, c)
        canonical_count[canonical] = canonical_count.get(canonical, 0) + n
    classes = {c for c, n in canonical_count.items() if n >= min_frames_per_class}
    for entry in timeline:
        # Per-frame: show canonical labels that made it into classes
        entry["objects"] = list(dict.fromkeys(
            CONCEPT_TO_CANONICAL.get(o, o) for o in entry["objects"]
            if CONCEPT_TO_CANONICAL.get(o, o) in classes
        ))
    return classes, timeline
