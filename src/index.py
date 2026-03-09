"""
In-memory feature index with save/load. Brute-force L2 search (simple and fine for demo).
"""
from pathlib import Path
import json
import numpy as np


class VideoIndex:
    """Stores video IDs, paths, and feature vectors; supports top-k search by L2 distance."""

    def __init__(self):
        self.ids: list[str] = []
        self.paths: list[str] = []
        self.features: np.ndarray | None = None

    def add(self, video_id: str, path: str, feature: np.ndarray) -> None:
        if self.features is None:
            self.features = feature.reshape(1, -1)
        else:
            self.features = np.vstack([self.features, feature.reshape(1, -1)])
        self.ids.append(video_id)
        self.paths.append(path)

    def search(self, query_feature: np.ndarray, k: int = 5) -> list[tuple[str, str, float]]:
        """
        Return top-k (video_id, path, distance) sorted by L2 distance (ascending).
        """
        if self.features is None or len(self.ids) == 0:
            return []
        q = query_feature.reshape(1, -1)
        dists = np.linalg.norm(self.features - q, axis=1)
        order = np.argsort(dists)
        return [
            (self.ids[i], self.paths[i], float(dists[i]))
            for i in order[: min(k, len(self.ids))]
        ]

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "features.npy", self.features)
        meta = {"ids": self.ids, "paths": self.paths}
        with open(directory / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, directory: str | Path) -> "VideoIndex":
        directory = Path(directory)
        idx = cls()
        idx.features = np.load(directory / "features.npy")
        with open(directory / "meta.json") as f:
            meta = json.load(f)
        idx.ids = meta["ids"]
        idx.paths = meta["paths"]
        return idx
