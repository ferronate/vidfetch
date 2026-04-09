from __future__ import annotations

import os
from typing import Any

import streamlit as st

try:
    from streamlit_app.api_client import APIError, DEFAULT_API_BASE, VidfetchClient
except ModuleNotFoundError:
    # Streamlit Cloud may execute this file as a script where the package root
    # is not importable as "streamlit_app".
    from api_client import APIError, DEFAULT_API_BASE, VidfetchClient


st.set_page_config(page_title="Vidfetch", page_icon="🎬", layout="wide")


def _pretty_job_status(status: str) -> str:
    return {
        "submitted": "Submitted",
        "queued": "Queued",
        "running": "Running",
        "done": "Complete",
        "failed": "Failed",
    }.get(status, status.title())


def _flatten_detections(detections: dict[str, Any] | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not detections:
        return rows

    for frame_index, entry in enumerate(detections.get("timeline", [])):
        for detection_index, detection in enumerate(entry.get("objects", [])):
            rows.append(
                {
                    "frame_index": frame_index,
                    "detection_index": detection_index,
                    "timestamp": float(entry.get("t", 0.0)),
                    "class": detection.get("class", ""),
                    "confidence": float(detection.get("confidence", 0.0)),
                    "corrected": bool(detection.get("corrected", False)),
                    "bbox": detection.get("bbox"),
                    "track_id": detection.get("track_id"),
                }
            )
    return rows


def _selected_video(videos: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not videos:
        return None

    default_video = st.session_state.get("selected_video_id")
    options = [video["id"] for video in videos]
    if default_video not in options:
        default_video = options[0]

    selected_video_id = st.sidebar.selectbox(
        "Selected video",
        options=options,
        index=options.index(default_video),
        format_func=lambda video_id: next((video["name"] for video in videos if video["id"] == video_id), video_id),
        key="selected_video_id",
    )
    return next((video for video in videos if video["id"] == selected_video_id), None)


def _video_url(client: VidfetchClient, video_id: str) -> str:
    return f"{client.base_url}/api/video/{video_id}"


def render_dashboard(client: VidfetchClient, videos: list[dict[str, Any]], objects: list[str], jobs: list[dict[str, Any]]) -> None:
    st.subheader("Dashboard")

    columns = st.columns(4)
    columns[0].metric("Videos", len(videos))
    columns[1].metric("Detected objects", len(objects))
    columns[2].metric("Jobs", len(jobs))
    active_jobs = sum(1 for job in jobs if job.get("status") in {"submitted", "queued", "running"})
    columns[3].metric("Active jobs", active_jobs)

    left, right = st.columns(2)
    with left:
        st.markdown("### Library")
        st.dataframe(videos, use_container_width=True, hide_index=True)
    with right:
        st.markdown("### Jobs")
        st.dataframe(jobs[:10], use_container_width=True, hide_index=True)


def render_gallery(client: VidfetchClient, videos: list[dict[str, Any]], jobs_by_video: dict[str, dict[str, Any]], selected_video: dict[str, Any] | None) -> None:
    st.subheader("Gallery & Detect")

    if not videos:
        st.info("No videos found in the data directory.")
        return

    left, right = st.columns([0.42, 0.58])
    with left:
        st.markdown("### Videos")
        st.dataframe(videos, use_container_width=True, hide_index=True)

    with right:
        if not selected_video:
            st.info("Pick a video from the sidebar.")
            return

        video_id = selected_video["id"]
        job = jobs_by_video.get(video_id)

        st.markdown(f"### {selected_video['name']}")
        st.video(_video_url(client, video_id))

        status_col, action_col = st.columns([0.62, 0.38])
        with status_col:
            if job:
                st.write(f"**Job status:** {_pretty_job_status(str(job.get('status', 'submitted')))}")
                if job.get("error"):
                    st.error(job["error"])
            else:
                st.write("**Job status:** Not queued")

        with action_col:
            detector_type = st.selectbox("Detector", ["auto", "yolo-world", "onnx", "yolo_object365"], key="gallery_detector")
            if st.button("Run detection", use_container_width=True):
                try:
                    response = client.run_detection(video_id, detector_type=detector_type)
                    if response.get("reused"):
                        st.success(response.get("message", "Reused existing detections."))
                    else:
                        st.success(f"Detection queued: {response.get('job_id', 'unknown')}")
                except APIError as exc:
                    st.error(str(exc))

        detections = client.get_video_detections(video_id)
        if detections:
            st.markdown("### Detection summary")
            summary_cols = st.columns(3)
            summary_cols[0].metric("Classes", len(detections.get("classes", [])))
            summary_cols[1].metric("Timeline frames", len(detections.get("timeline", [])))
            summary_cols[2].metric("Detections", detections.get("total_detections", len(detections.get("timeline", []))))

            with st.expander("Timeline preview", expanded=False):
                st.json({
                    "video": detections.get("video", video_id),
                    "classes": detections.get("classes", []),
                    "timeline": detections.get("timeline", [])[:5],
                })


def render_search(client: VidfetchClient, videos: list[dict[str, Any]], objects: list[str]) -> None:
    st.subheader("Search")

    with st.form("search_form"):
        search_input = st.text_input("Object names", placeholder="person, car, dog")
        filter_types = st.multiselect("Quick type filters", options=objects)
        color_filter = st.selectbox("Color filter", ["any", "warm", "cool", "bright", "dark", "same"])
        color_ref_video_id = ""
        if color_filter == "same":
            color_ref_video_id = st.selectbox(
                "Reference video",
                options=[video["id"] for video in videos],
                format_func=lambda video_id: next((video["name"] for video in videos if video["id"] == video_id), video_id),
            )
        k = st.slider("Max results", 1, 20, 8)
        submit = st.form_submit_button("Search")

    if submit:
        try:
            result = client.query_videos(
                color_filter=color_filter,
                color_ref_video_id=color_ref_video_id,
                search_input=search_input,
                filter_types=filter_types,
                k=k,
            )
            st.session_state["last_search_result"] = result
        except APIError as exc:
            st.error(str(exc))

    result = st.session_state.get("last_search_result")
    if not result:
        st.info("Run a search to see ranked video matches.")
        return

    st.markdown(f"**Query object:** {result.get('query_object') or 'none'}")
    st.markdown(f"**Time:** {result.get('time_ms', 0):.2f} ms")

    for match in result.get("results", []):
        with st.expander(f"{match['name']}  ·  distance {match['distance']:.4f}", expanded=False):
            if match.get("object_segments"):
                st.write("Matches:")
                st.json(match["object_segments"])
            st.video(_video_url(client, match["id"]))


def render_review(client: VidfetchClient, selected_video: dict[str, Any] | None, videos: list[dict[str, Any]]) -> None:
    st.subheader("Review & Corrections")

    if not selected_video:
        st.info("Pick a video from the sidebar.")
        return

    video_id = selected_video["id"]
    detections = client.get_video_detections(video_id)
    if not detections or not detections.get("timeline"):
        st.info("Run detection first to enable review and corrections.")
        return

    flattened = _flatten_detections(detections)
    detected_classes = sorted({str(item.get("class", "")).strip() for item in flattened if str(item.get("class", "")).strip()})
    class_options = detected_classes + ["Custom..."] if detected_classes else ["Custom..."]

    review_tab, corrections_tab, rules_tab = st.tabs(["Review", "Corrections", "Rules"])

    with review_tab:
        st.markdown(f"### {selected_video['name']}")
        st.video(_video_url(client, video_id))

        st.markdown("#### Detections")
        selection_labels = [
            f"Frame {row['frame_index']} @ {row['timestamp']:.1f}s - {row['class']} ({row['confidence'] * 100:.1f}%)"
            for row in flattened
        ]
        selected_label = st.selectbox("Pick a detection", options=selection_labels, key="review_detection_select")
        selected_index = selection_labels.index(selected_label)
        selected_detection = flattened[selected_index]

        st.write(
            {
                "frame_number": selected_detection["frame_index"],
                "timestamp": selected_detection["timestamp"],
                "class": selected_detection["class"],
                "confidence": selected_detection["confidence"],
                "bbox": selected_detection["bbox"],
            }
        )

        with st.form("relabel_form"):
            relabel_choice = st.selectbox("Relabel as", options=class_options, key="relabel_choice")
            custom_value = ""
            if relabel_choice == "Custom...":
                custom_value = st.text_input("Custom class", placeholder="Enter a new class")
            notes = st.text_input("Notes", placeholder="Optional note")
            save = st.form_submit_button("Save relabel")

        if save:
            target_class = custom_value.strip() if relabel_choice == "Custom..." else relabel_choice.strip()
            if not target_class:
                st.error("Please enter a target class.")
            elif target_class == selected_detection["class"]:
                st.warning("The target class is the same as the current class.")
            else:
                try:
                    client.add_correction(
                        {
                            "video_id": video_id,
                            "frame_number": selected_detection["frame_index"],
                            "timestamp": selected_detection["timestamp"],
                            "original_class": selected_detection["class"],
                            "corrected_class": target_class,
                            "original_confidence": selected_detection["confidence"],
                            "action": "relabel",
                            "bbox": selected_detection["bbox"],
                            "notes": notes,
                        }
                    )
                    st.success("Correction saved.")
                    st.rerun()
                except APIError as exc:
                    st.error(str(exc))

        if st.button("Delete selected detection", type="secondary"):
            try:
                client.add_correction(
                    {
                        "video_id": video_id,
                        "frame_number": selected_detection["frame_index"],
                        "timestamp": selected_detection["timestamp"],
                        "original_class": selected_detection["class"],
                        "corrected_class": "__DELETE__",
                        "original_confidence": selected_detection["confidence"],
                        "action": "delete",
                        "bbox": selected_detection["bbox"],
                        "notes": "Deleted from Streamlit review",
                    }
                )
                st.success("Detection marked for deletion.")
                st.rerun()
            except APIError as exc:
                st.error(str(exc))

    with corrections_tab:
        corrections = client.get_corrections(video_id)
        st.markdown(f"**{len(corrections)} correction(s)**")
        if not corrections:
            st.info("No corrections recorded yet.")
        else:
            st.dataframe(corrections, use_container_width=True, hide_index=True)
            correction_labels = [
                f"#{item['id']} {item['original_class']} → {item['corrected_class']} @ {item['timestamp']:.1f}s"
                for item in corrections
            ]
            selected_correction = st.selectbox("Select correction to delete", correction_labels)
            selected_correction_id = corrections[correction_labels.index(selected_correction)]["id"]
            if st.button("Delete correction"):
                try:
                    client.delete_correction(int(selected_correction_id))
                    st.success("Correction deleted.")
                    st.rerun()
                except APIError as exc:
                    st.error(str(exc))

    with rules_tab:
        rules = client.get_rules(enabled_only=False)
        st.markdown(f"**{len(rules)} rule(s)**")

        if rules:
            st.dataframe(rules, use_container_width=True, hide_index=True)
            rule_labels = [
                f"#{item['id']} {item['pattern_class']} → {item['target_class']} ({'on' if item.get('enabled', True) else 'off'})"
                for item in rules
            ]
            selected_rule = st.selectbox("Select rule", rule_labels)
            selected_rule_obj = rules[rule_labels.index(selected_rule)]
            action_col_1, action_col_2 = st.columns(2)
            if action_col_1.button("Toggle rule"):
                try:
                    client.toggle_rule(int(selected_rule_obj["id"]), bool(selected_rule_obj.get("enabled", True)))
                    st.rerun()
                except APIError as exc:
                    st.error(str(exc))
            if action_col_2.button("Delete rule"):
                try:
                    client.delete_rule(int(selected_rule_obj["id"]))
                    st.rerun()
                except APIError as exc:
                    st.error(str(exc))

        with st.form("add_rule_form"):
            st.markdown("#### Add rule")
            pattern_class = st.text_input("Pattern class", placeholder="e.g. dog")
            target_class = st.text_input("Target class", placeholder="e.g. pet")
            confidence_threshold = st.text_input("Confidence threshold", placeholder="Optional, e.g. 0.5")
            video_pattern = st.text_input("Video pattern", placeholder="Optional substring match")
            add_rule = st.form_submit_button("Create rule")

        if add_rule:
            if not pattern_class.strip() or not target_class.strip():
                st.error("Pattern and target are required.")
            else:
                try:
                    client.add_rule(pattern_class, target_class, confidence_threshold, video_pattern)
                    st.success("Rule added.")
                    st.rerun()
                except APIError as exc:
                    st.error(str(exc))

        if st.button("Generate rules from corrections"):
            try:
                client.generate_rules()
                st.success("Rule generation complete.")
                st.rerun()
            except APIError as exc:
                st.error(str(exc))


def render_settings(client: VidfetchClient) -> None:
    st.subheader("Settings")
    st.write("Backend API base:", client.base_url)
    st.write("Default API base from environment:", DEFAULT_API_BASE)
    st.caption("Set VIDFETCH_API_URL if the backend is not running on localhost:8000.")


def main() -> None:
    st.title("Vidfetch")
    st.caption("Streamlit rebuild for local video search, detection, and corrections.")

    api_base = st.sidebar.text_input("Backend API base", value=os.environ.get("VIDFETCH_API_URL", DEFAULT_API_BASE))
    client = VidfetchClient(api_base)

    try:
        videos = client.get_videos()
        objects = client.get_objects()
        jobs = client.get_jobs()
    except APIError as exc:
        st.error(f"Backend unavailable: {exc}")
        st.stop()

    jobs_by_video = {job.get("video_id", ""): job for job in jobs}
    selected_video = _selected_video(videos)

    page = st.sidebar.radio(
        "Navigate",
        ["Dashboard", "Gallery & Detect", "Search", "Review & Corrections", "Settings"],
    )

    if page == "Dashboard":
        render_dashboard(client, videos, objects, jobs)
    elif page == "Gallery & Detect":
        render_gallery(client, videos, jobs_by_video, selected_video)
    elif page == "Search":
        render_search(client, videos, objects)
    elif page == "Review & Corrections":
        render_review(client, selected_video, videos)
    else:
        render_settings(client)


if __name__ == "__main__":
    main()
