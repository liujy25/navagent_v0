from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from navclaw.agent.visual_action_context import ordered_panorama_angles_left_to_right
from navclaw.agent.visual_action_context import ordered_visual_views_left_to_right
from navclaw.agent.visual_action_context import VisualActionContext, VisualViewContext
from navclaw.agent.visual_policy_prompt_images import _compose_labeled_image_strip
from navclaw.agent.visual_policy_prompt_images import _image_array_for_view
from navclaw.llm.image_preprocessing import LLM_CAMERA_IMAGE_MAX_SIZE
from navclaw.llm.image_preprocessing import resize_rgb_to_fit
from navclaw.llm.image_preprocessing import scale_bbox
from navclaw.visualization.action_mode_overlays import _node_id_sort_key

if TYPE_CHECKING:
    from navclaw.graph.graph import Graph
    from navclaw.perception.landmark_detection import LandmarkDetectionController
    from navclaw.runtime.cache import RuntimeCache


@dataclass(frozen=True)
class VlnLandmarkEvidence:
    landmark_id: str
    class_name: str
    obs_id: str
    angle_deg: int
    bbox: list[float]
    score: float
    world_position: dict[str, float] = field(default_factory=dict)
    display_label: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "landmark_id": str(self.landmark_id),
            "class_name": str(self.class_name),
            "obs_id": str(self.obs_id),
            "angle_deg": int(self.angle_deg),
            "bbox": [float(value) for value in self.bbox],
            "score": float(self.score),
            "world_position": dict(self.world_position),
            "display_label": str(self.display_label),
        }


@dataclass(frozen=True)
class VlnLandmarkBevMarker:
    label: int
    class_name: str
    xy: tuple[float, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "label": int(self.label),
            "class_name": str(self.class_name),
            "xy": [float(self.xy[0]), float(self.xy[1])],
        }


@dataclass(frozen=True)
class VlnLandmarkContext:
    evidences: list[VlnLandmarkEvidence] = field(default_factory=list)
    bev_markers: list[VlnLandmarkBevMarker] = field(default_factory=list)
    text: str = ""


def build_vln_landmark_context(
    *,
    landmark_controller: "LandmarkDetectionController",
    graph: "Graph",
    visual_context: VisualActionContext,
    floor_id: str,
) -> VlnLandmarkContext:
    if not landmark_controller.is_active():
        return VlnLandmarkContext()
    labels_by_landmark_id = _landmark_labels_by_id(
        landmark_controller=landmark_controller,
        graph=graph,
        floor_id=floor_id,
    )
    evidences = _current_view_evidences(
        landmark_controller=landmark_controller,
        visual_context=visual_context,
        labels_by_landmark_id=labels_by_landmark_id,
    )
    bev_markers = _bev_markers(
        landmark_controller=landmark_controller,
        labels_by_landmark_id=labels_by_landmark_id,
    )
    return VlnLandmarkContext(
        evidences=evidences,
        bev_markers=bev_markers,
        text=_landmark_context_text(
            evidences=evidences,
            visual_context=visual_context,
            landmark_controller=landmark_controller,
        ),
    )


def _landmark_context_text(
    *,
    evidences: list[VlnLandmarkEvidence],
    visual_context: VisualActionContext,
    landmark_controller: "LandmarkDetectionController",
) -> str:
    sections = [_detected_landmarks_by_view_text(evidences, visual_context)]
    semantic_lines: list[str] = []
    seen: set[str] = set()
    for evidence in evidences:
        candidate = landmark_controller.buffer.find_candidate(str(evidence.landmark_id))
        if candidate is None:
            continue
        semantic_text = str(candidate.attributes.get("semantic_text", "")).strip()
        if semantic_text == "" or semantic_text.casefold() in seen:
            continue
        seen.add(semantic_text.casefold())
        semantic_lines.append(f"- {evidence.landmark_id}: {semantic_text}")
    if semantic_lines != []:
        sections.append("Completed sign/room-number readings:\n" + "\n".join(semantic_lines))
    return "\n\n".join(section for section in sections if section.strip() != "")


def draw_vln_landmark_panorama_strip(
    *,
    cache: "RuntimeCache",
    visual_context: VisualActionContext,
    evidences: list[VlnLandmarkEvidence],
) -> np.ndarray:
    images: list[np.ndarray] = []
    labels: list[str] = []
    for view in ordered_visual_views_left_to_right(visual_context.views):
        raw_rgb = np.asarray(_image_array_for_view(cache=cache, view=view), dtype=np.uint8)
        resized = resize_rgb_to_fit(raw_rgb, max_size=LLM_CAMERA_IMAGE_MAX_SIZE)
        image = np.asarray(resized.image, dtype=np.uint8)
        image = draw_vln_landmarks_on_view_image(
            image=image,
            cache=cache,
            view=view,
            evidences=evidences,
        )
        images.append(image)
        labels.append(f"angle_{int(view.angle_deg)}")
    return _compose_labeled_image_strip(images=images, labels=labels)


def draw_vln_landmarks_on_view_image(
    *,
    image: np.ndarray,
    cache: "RuntimeCache",
    view: VisualViewContext,
    evidences: list[VlnLandmarkEvidence],
) -> np.ndarray:
    canvas = Image.fromarray(np.asarray(image, dtype=np.uint8).copy()).convert("RGB")
    observation = cache.get_observation(str(view.obs_id)).observation
    original_rgb = np.asarray(observation.rgb, dtype=np.uint8)
    original_height, original_width = original_rgb.shape[:2]
    render_height, render_width = np.asarray(image).shape[:2]
    scale_x = float(render_width) / float(original_width)
    scale_y = float(render_height) / float(original_height)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for evidence in evidences:
        if str(evidence.obs_id) != str(view.obs_id):
            continue
        _draw_landmark_bbox(draw=draw, evidence=evidence, scale_x=scale_x, scale_y=scale_y, font=font)
    return np.asarray(canvas, dtype=np.uint8)


def _landmark_labels_by_id(
    *,
    landmark_controller: "LandmarkDetectionController",
    graph: "Graph",
    floor_id: str,
) -> dict[str, int]:
    del graph, floor_id
    candidates = _all_landmark_candidates(landmark_controller)
    labels: dict[str, int] = {}
    for candidate in candidates:
        landmark_id = str(candidate.get("landmark_id", ""))
        labels[landmark_id] = _global_landmark_index(landmark_id)
    return labels


def _global_landmark_index(landmark_id: str) -> int:
    prefix, separator, index_text = str(landmark_id).rpartition("_")
    if prefix != "landmark" or separator == "" or not index_text.isdigit():
        raise ValueError(f"invalid global landmark id: {landmark_id!r}")
    return int(index_text)


def _current_view_evidences(
    *,
    landmark_controller: "LandmarkDetectionController",
    visual_context: VisualActionContext,
    labels_by_landmark_id: dict[str, int],
) -> list[VlnLandmarkEvidence]:
    angle_by_obs_id = {str(view.obs_id): int(view.angle_deg) for view in visual_context.views}
    obs_ids = list(angle_by_obs_id.keys())
    raw_by_obs_id = landmark_controller.detections_for_obs_ids(obs_ids)
    fallback_by_obs_id = _buffer_detections_by_obs_id(
        landmark_controller=landmark_controller,
        obs_ids=set(obs_ids),
    )
    best_by_key: dict[tuple[str, str], dict[str, object]] = {}
    for obs_id in obs_ids:
        detections = list(raw_by_obs_id.get(str(obs_id), []))
        if detections == []:
            detections = list(fallback_by_obs_id.get(str(obs_id), []))
        for detection in detections:
            landmark_id = str(detection.get("landmark_id", "") or "")
            if landmark_id == "" or landmark_id not in labels_by_landmark_id:
                continue
            key = (str(obs_id), landmark_id)
            existing = best_by_key.get(key)
            if existing is None or _detection_rank(detection) > _detection_rank(existing):
                best_by_key[key] = dict(detection)
    evidences: list[VlnLandmarkEvidence] = []
    for (obs_id, landmark_id), detection in sorted(
        best_by_key.items(),
        key=lambda item: (angle_by_obs_id.get(item[0][0], 10**9), labels_by_landmark_id.get(item[0][1], 10**9)),
    ):
        position = dict(detection.get("position", {}))
        evidences.append(
            VlnLandmarkEvidence(
                landmark_id=landmark_id,
                class_name=str(detection.get("class_name", "")),
                obs_id=str(obs_id),
                angle_deg=int(angle_by_obs_id[str(obs_id)]),
                bbox=[float(value) for value in list(detection.get("bbox", []))],
                score=float(detection.get("score", 0.0)),
                world_position={str(key): float(value) for key, value in position.items()},
                display_label=str(labels_by_landmark_id[landmark_id]),
            )
        )
    return evidences


def _bev_markers(
    *,
    landmark_controller: "LandmarkDetectionController",
    labels_by_landmark_id: dict[str, int],
) -> list[VlnLandmarkBevMarker]:
    markers: list[VlnLandmarkBevMarker] = []
    for candidate in _all_landmark_candidates(landmark_controller):
        landmark_id = str(candidate.get("landmark_id", "") or "")
        if landmark_id not in labels_by_landmark_id:
            continue
        position = dict(candidate.get("position", {}))
        if position.get("x") is None or position.get("y") is None:
            continue
        markers.append(
            VlnLandmarkBevMarker(
                label=int(labels_by_landmark_id[landmark_id]),
                class_name=str(candidate.get("class_name", "")),
                xy=(float(position["x"]), float(position["y"])),
            )
        )
    markers.sort(key=lambda item: int(item.label))
    return markers


def _detected_landmarks_by_view_text(
    evidences: list[VlnLandmarkEvidence],
    visual_context: VisualActionContext,
) -> str:
    by_angle: dict[int, list[VlnLandmarkEvidence]] = {}
    for evidence in evidences:
        by_angle.setdefault(int(evidence.angle_deg), []).append(evidence)
    lines = [
        "Detected landmarks by view (canonical refs; boxes show their numeric suffixes):"
    ]
    for angle_value in ordered_panorama_angles_left_to_right(visual_context.available_angles):
        items = sorted(by_angle.get(int(angle_value), []), key=lambda item: int(item.display_label or 10**9))
        item_text = ", ".join(
            f"{item.landmark_id} {item.class_name} confidence={float(item.score):.2f}"
            for item in items
            if str(item.landmark_id).strip() != ""
        )
        if item_text != "":
            lines.append(f"- angle_{int(angle_value)}: {item_text}")
    return "\n".join(lines)


def _all_landmark_candidates(landmark_controller: "LandmarkDetectionController") -> list[dict[str, object]]:
    context = landmark_controller.buffer.to_context()
    candidates: list[dict[str, object]] = []
    for key in ("confirmed_landmarks", "needs_verification_candidates", "pending_candidates"):
        candidates.extend(
            dict(item)
            for item in list(context.get(key, []))
            if isinstance(item, dict)
        )
    candidates.sort(
        key=lambda item: (
            int(item.get("first_step_index", 0)),
            _node_id_sort_key(str(item.get("landmark_id", ""))),
        )
    )
    return candidates


def _buffer_detections_by_obs_id(
    *,
    landmark_controller: "LandmarkDetectionController",
    obs_ids: set[str],
) -> dict[str, list[dict[str, object]]]:
    result: dict[str, list[dict[str, object]]] = {}
    for candidate in _all_landmark_candidates(landmark_controller):
        landmark_id = str(candidate.get("landmark_id", "") or "")
        class_name = str(candidate.get("class_name", "") or "")
        for detection in list(candidate.get("detections", [])):
            if not isinstance(detection, dict):
                continue
            obs_id = str(detection.get("obs_id", "") or "")
            if obs_id not in obs_ids:
                continue
            payload = dict(detection)
            payload["landmark_id"] = landmark_id
            payload["class_name"] = class_name
            result.setdefault(obs_id, []).append(payload)
    return result


def _detection_rank(detection: dict[str, object]) -> tuple[float, int, str]:
    return (
        float(detection.get("score", 0.0)),
        int(detection.get("step_index", 0) or 0),
        str(detection.get("det_id", "")),
    )


def _draw_landmark_bbox(
    *,
    draw: ImageDraw.ImageDraw,
    evidence: VlnLandmarkEvidence,
    scale_x: float,
    scale_y: float,
    font: ImageFont.ImageFont,
) -> None:
    if len(evidence.bbox) != 4:
        return
    x0, y0, x1, y1 = scale_bbox(evidence.bbox, scale_x=scale_x, scale_y=scale_y)
    color = _landmark_color(str(evidence.class_name))
    draw.rectangle((x0, y0, x1, y1), outline=color, width=4)
    label = str(evidence.display_label).strip()
    if label == "":
        return
    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_w = int(text_bbox[2] - text_bbox[0])
    text_h = int(text_bbox[3] - text_bbox[1])
    pad = 3
    box = (x0, max(0, y0 - text_h - 2 * pad), x0 + text_w + 2 * pad, max(text_h + 2 * pad, y0))
    draw.rectangle(box, fill=color)
    draw.text((box[0] + pad, box[1] + pad), label, fill=(255, 255, 255), font=font)


def _landmark_color(class_name: str) -> tuple[int, int, int]:
    palette = [
        (235, 67, 53),
        (52, 168, 83),
        (66, 133, 244),
        (251, 188, 5),
        (171, 71, 188),
        (0, 172, 193),
    ]
    index = sum(ord(ch) for ch in str(class_name)) % len(palette)
    return palette[index]
