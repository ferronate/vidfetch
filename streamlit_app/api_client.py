from __future__ import annotations

import os
from typing import Any

import requests


DEFAULT_API_BASE = os.environ.get("VIDFETCH_API_URL", "http://localhost:8000").rstrip("/")


class APIError(RuntimeError):
    pass


class VidfetchClient:
    def __init__(self, base_url: str | None = None):
        self.base_url = (base_url or DEFAULT_API_BASE).rstrip("/")

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _request(self, method: str, path: str, **kwargs) -> Any:
        try:
            response = requests.request(method, self._url(path), timeout=60, **kwargs)
        except requests.RequestException as exc:
            raise APIError(str(exc)) from exc

        if not response.ok:
            detail = None
            try:
                detail = response.json().get("detail")
            except Exception:
                detail = response.text.strip() or None
            raise APIError(detail or f"{method} {path} failed with {response.status_code}")

        if not response.content:
            return None

        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            return response.json()
        return response.text

    def get_videos(self) -> list[dict[str, Any]]:
        data = self._request("GET", "/api/videos") or {}
        return list(data.get("videos", []))

    def get_objects(self) -> list[str]:
        data = self._request("GET", "/api/objects") or {}
        return list(data.get("objects", []))

    def query_videos(
        self,
        *,
        color_filter: str,
        color_ref_video_id: str,
        search_input: str,
        filter_types: list[str],
        k: int = 8,
    ) -> dict[str, Any]:
        form_items: list[tuple[str, str]] = []
        form_items.append(
            (
                "color_filter",
                color_ref_video_id if color_filter == "same" and color_ref_video_id else color_filter,
            )
        )
        form_items.append(("k", str(k)))

        query_types = [
            item.strip()
            for item in search_input.split(",")
            if item.strip()
        ]
        if not query_types:
            query_types = filter_types

        for value in query_types:
            form_items.append(("object_types", value))

        return self._request("POST", "/api/query", data=form_items) or {}

    def get_video_detections(self, video_id: str) -> dict[str, Any] | None:
        try:
            return self._request("GET", f"/api/video/{video_id}/detections")
        except APIError:
            return None

    def run_detection(self, video_id: str, detector_type: str = "auto") -> dict[str, Any]:
        return self._request(
            "POST",
            "/api/detect",
            params={"video_id": video_id},
            data={"detector_type": detector_type},
        ) or {}

    def get_jobs(self) -> list[dict[str, Any]]:
        data = self._request("GET", "/api/detect/jobs") or {}
        return list(data.get("jobs", []))

    def get_corrections(self, video_id: str, limit: int = 100) -> list[dict[str, Any]]:
        data = self._request("GET", "/api/corrections", params={"video_id": video_id, "limit": limit}) or {}
        return list(data.get("corrections", []))

    def add_correction(self, payload: dict[str, Any]) -> dict[str, Any]:
        form_data: dict[str, str] = {
            "video_id": payload["video_id"],
            "frame_number": str(payload["frame_number"]),
            "timestamp": str(payload["timestamp"]),
            "original_class": payload["original_class"],
            "corrected_class": payload["corrected_class"],
            "original_confidence": str(payload.get("original_confidence", 0.0)),
            "action": payload.get("action", "relabel"),
            "notes": payload.get("notes", ""),
        }
        bbox = payload.get("bbox")
        if bbox is not None:
            import json

            form_data["bbox"] = json.dumps(bbox)
        return self._request("POST", "/api/corrections", data=form_data) or {}

    def delete_correction(self, correction_id: int) -> dict[str, Any]:
        return self._request("DELETE", f"/api/corrections/{correction_id}") or {}

    def get_rules(self, enabled_only: bool = False) -> list[dict[str, Any]]:
        data = self._request("GET", "/api/corrections/rules", params={"enabled_only": str(enabled_only).lower()}) or {}
        return list(data.get("rules", []))

    def add_rule(self, pattern: str, target: str, confidence_threshold: str = "", video_pattern: str = "") -> dict[str, Any]:
        form_data: dict[str, str] = {
            "pattern_class": pattern,
            "target_class": target,
        }
        if confidence_threshold.strip():
            form_data["confidence_threshold"] = confidence_threshold.strip()
        if video_pattern.strip():
            form_data["video_pattern"] = video_pattern.strip()
        return self._request("POST", "/api/corrections/rules", data=form_data) or {}

    def toggle_rule(self, rule_id: int, enabled: bool) -> dict[str, Any]:
        return self._request("POST", f"/api/corrections/rules/{rule_id}/toggle", data={"disable": str(not enabled).lower()}) or {}

    def delete_rule(self, rule_id: int) -> dict[str, Any]:
        return self._request("DELETE", f"/api/corrections/rules/{rule_id}") or {}

    def generate_rules(self, min_occurrences: int = 3) -> dict[str, Any]:
        return self._request("POST", "/api/corrections/generate-rules", data={"min_occurrences": str(min_occurrences)}) or {}
