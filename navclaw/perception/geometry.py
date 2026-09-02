from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image
from PIL import ImageDraw

from navclaw.llm.image_preprocessing import LLM_CAMERA_IMAGE_MAX_SIZE
from navclaw.llm.image_preprocessing import resize_rgb_to_fit
from navclaw.llm.image_preprocessing import scale_bbox
from navclaw.schemas import ActionResult
from navclaw.env.interface import Detection3D
from navclaw.env.interface import RawDetection
from navclaw.runtime.cache import RuntimeCache

if TYPE_CHECKING:
    from navclaw.perception.detectors.interface import DetectorInterface
    from navclaw.env.interface import EnvInterface
    from navclaw.mapping.exploration.manager import ExplorationManager

DETECT_MAX_RESULTS_PER_CLASS = 2
DETECTION_OVERLAY_BBOX_SCALE = 1.3
DETECTION_OVERLAY_BBOX_LINE_WIDTH = 3
DETECTION_DEPTH_MIN_M = 0.1
DETECTION_DEPTH_MAX_M = 9.5
DETECTION_DEPTH_SATURATION_EPS_M = 0.05
DETECTION_DEPTH_PROXY_SATURATION_RATIO = 0.8
DETECTION_PROXY_MOVE_STEP_M = 9.5


def _normalize_detection_class_name(class_name: object) -> str:
    return str(class_name).strip().replace("_", " ")


def _detection_query_classes(class_name: str, confusable_class_names: list[str] | None) -> list[str]:
    query_classes: list[str] = []
    seen: set[str] = set()
    for item in [class_name, *(confusable_class_names or [])]:
        normalized = _normalize_detection_class_name(item)
        if normalized == "" or normalized in seen:
            continue
        query_classes.append(normalized)
        seen.add(normalized)
    return query_classes


def _detection_synonym_classes(
    class_name: str,
    synonym_class_names: list[str] | None,
) -> list[str]:
    query_classes: list[str] = []
    seen: set[str] = set()
    for item in [class_name, *(synonym_class_names or [])]:
        normalized = _normalize_detection_class_name(item)
        if normalized == "" or normalized in seen:
            continue
        query_classes.append(normalized)
        seen.add(normalized)
    return query_classes


def get_obs(env: EnvInterface, cache: RuntimeCache, exploration: ExplorationManager) -> ActionResult:
    record = cache.store_observation(env.get_obs())
    frontier_update = exploration.observe_observation(cache=cache, obs_id=record.id)
    if hasattr(env, "emit_visualization_observation"):
        env.emit_visualization_observation(record.observation)
    pose = record.observation.pose
    return ActionResult(
        ok=True,
        data={
            "obs_id": record.id,
            "pose": pose.to_dict(),
            "rgb_id": f"{record.id}:rgb",
            "depth_id": f"{record.id}:depth",
            "text_hint": record.observation.text_hint,
            "frontier_count": frontier_update["frontier_count"],
            "frontiers": frontier_update["frontiers"],
        },
        message=f"Captured observation {record.id}.",
    )


def detect(
    detector: DetectorInterface,
    cache: RuntimeCache,
    class_name: str,
    obs_id: str | None = None,
    store_overlay: bool = True,
    min_score: float | None = None,
    confusable_class_names: list[str] | None = None,
    synonym_class_names: list[str] | None = None,
    canonical_class_name: str | None = None,
    canonical_class_map: dict[str, str] | None = None,
    agnostic_nms: bool | None = None,
    max_results: int | None = None,
    overlay_bbox_scale: float | None = None,
) -> ActionResult:
    if obs_id is None:
        raise ValueError("detect requires obs_id")
    observation = cache.get_observation(obs_id).observation
    if observation.rgb is None:
        raise ValueError("detect requires observation.rgb")

    image_rgb = np.asarray(observation.rgb)
    if synonym_class_names is None:
        query_classes = _detection_query_classes(
            class_name=str(class_name),
            confusable_class_names=confusable_class_names,
        )
        accepted_classes = {_normalize_detection_class_name(query_classes[0])} if query_classes else set()
    else:
        query_classes = _detection_synonym_classes(
            class_name=str(class_name),
            synonym_class_names=synonym_class_names,
        )
        accepted_classes = {
            _normalize_detection_class_name(item)
            for item in query_classes
        }
    if query_classes == []:
        raise ValueError("detect requires at least one non-empty class_name")
    target_class_name = str(query_classes[0])
    canonical_class_name_text = (
        target_class_name
        if canonical_class_name is None
        else _normalize_detection_class_name(canonical_class_name)
    )
    normalized_canonical_class_map = {
        _normalize_detection_class_name(key): _normalize_detection_class_name(value)
        for key, value in dict(canonical_class_map or {}).items()
    }

    def output_class_name(detector_class_name: object) -> str:
        if canonical_class_name is not None:
            return canonical_class_name_text
        normalized = _normalize_detection_class_name(detector_class_name)
        return normalized_canonical_class_map.get(normalized, canonical_class_name_text)

    detector_outputs = detector.detect_classes(
        image_rgb=image_rgb,
        class_names=query_classes,
        agnostic_nms=(len(query_classes) > 1) if agnostic_nms is None else bool(agnostic_nms),
    )
    detector_outputs = [
        item
        for item in detector_outputs
        if _normalize_detection_class_name(item.class_name) in accepted_classes
    ]
    if min_score is not None:
        threshold = float(min_score)
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError("detect min_score must be in [0, 1]")
        detector_outputs = [
            item
            for item in detector_outputs
            if float(item.score) >= threshold
        ]
    result_limit = int(DETECT_MAX_RESULTS_PER_CLASS) if max_results is None else int(max_results)
    detector_outputs = detector_outputs[:result_limit]
    raw_detections = [
        RawDetection(
            class_name=output_class_name(item.class_name),
            bbox=list(item.bbox),
            score=item.score,
            mask=item.mask,
            metadata={"detector_class_name": str(item.class_name)},
        )
        for item in detector_outputs
    ]
    records = cache.store_detections(obs_id=obs_id, detections=raw_detections)
    overlay_ids: list[str] = []
    detection_payloads = []
    for record in records:
        payload = {
            "det_id": record.id,
            "class_name": record.detection.class_name,
            "detector_class_name": str(
                record.detection.metadata.get("detector_class_name", record.detection.class_name)
            ),
            "bbox": record.detection.bbox,
            "score": record.detection.score,
            "mask_id": f"{record.id}:mask",
        }
        if bool(store_overlay):
            overlay = _render_detection_overlay(
                image_rgb=image_rgb,
                records=[record],
                bbox_scale=overlay_bbox_scale,
                image_max_size=LLM_CAMERA_IMAGE_MAX_SIZE,
            )
            overlay_id = cache.store_image(
                overlay,
                kind="detection_overlay",
                metadata={
                    "det_id": record.id,
                    "class_name": record.detection.class_name,
                },
            ).id
            payload["overlay_id"] = overlay_id
            overlay_ids.append(overlay_id)
        detection_payloads.append(payload)
    data = {
        "obs_id": obs_id,
        "detections": detection_payloads,
    }
    if overlay_ids != []:
        data["overlay_ids"] = overlay_ids
        data["overlay_id"] = overlay_ids[0]
    return ActionResult(
        ok=True,
        data=data,
        message=f"Detected {len(records)} objects for class {class_name}.",
    )


def _render_detection_overlay(
    image_rgb: np.ndarray,
    records,
    bbox_scale: float | None = None,
    image_max_size: tuple[int, int] | None = None,
) -> np.ndarray:
    overlay_items = [
        {
            "det_id": str(record.id),
            "bbox": [float(value) for value in record.detection.bbox],
        }
        for record in records
    ]
    return render_detection_overlay_from_payloads(
        image_rgb=np.asarray(image_rgb, dtype=np.uint8),
        detections=overlay_items,
        bbox_scale=bbox_scale,
        image_max_size=image_max_size,
    )


def render_detection_overlay_from_payloads(
    image_rgb: np.ndarray,
    detections: list[dict[str, object]],
    bbox_scale: float | None = None,
    image_max_size: tuple[int, int] | None = None,
) -> np.ndarray:
    resized = resize_rgb_to_fit(image_rgb, max_size=image_max_size)
    scale_x = float(resized.metadata["scale_x"])
    scale_y = float(resized.metadata["scale_y"])
    image = Image.fromarray(np.asarray(resized.image, dtype=np.uint8).copy())
    draw = ImageDraw.Draw(image)
    for detection in detections:
        det_id = str(detection.get("det_id", "")).strip()
        bbox = [float(value) for value in list(detection.get("bbox", []))]
        if det_id == "" or len(bbox) != 4:
            continue
        scaled_bbox = scale_bbox(bbox, scale_x=scale_x, scale_y=scale_y)
        x0, y0, x1, y1 = _expand_bbox_for_overlay(
            bbox=list(scaled_bbox),
            image_width=image.width,
            image_height=image.height,
            bbox_scale=bbox_scale,
        )
        draw.rectangle(
            (x0, y0, x1, y1),
            outline=(255, 0, 0),
            width=DETECTION_OVERLAY_BBOX_LINE_WIDTH,
        )
        label = det_id
        text_bbox = draw.textbbox((0, 0), label)
        text_w = int(text_bbox[2] - text_bbox[0])
        text_h = int(text_bbox[3] - text_bbox[1])
        pad_x = 10
        pad_y = 6
        box_w = text_w + 2 * pad_x
        box_h = text_h + 2 * pad_y
        label_x = int(max(0, min(image.width - box_w, x0)))
        label_y = int(max(0, y0 - box_h - 4))
        if label_y == 0:
            label_y = int(min(image.height - box_h, y0 + 4))
        draw.rectangle(
            (label_x, label_y, label_x + box_w, label_y + box_h),
            fill=(255, 255, 255),
            outline=(0, 0, 0),
            width=3,
        )
        text_x = label_x + (box_w - text_w) / 2.0
        text_y = label_y + (box_h - text_h) / 2.0 - text_bbox[1]
        draw.text((text_x, text_y), label, fill=(0, 0, 0))
    return np.asarray(image, dtype=np.uint8)


def _expand_bbox_for_overlay(
    bbox: list[float],
    image_width: int,
    image_height: int,
    bbox_scale: float | None = None,
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = [float(value) for value in bbox]
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"invalid detection bbox for overlay: {bbox}")
    scale = DETECTION_OVERLAY_BBOX_SCALE if bbox_scale is None else float(bbox_scale)
    if scale <= 0.0:
        raise ValueError("detection overlay bbox_scale must be positive")

    center_x = (x0 + x1) / 2.0
    center_y = (y0 + y1) / 2.0
    half_width = (x1 - x0) * scale / 2.0
    half_height = (y1 - y0) * scale / 2.0

    expanded_x0 = max(0.0, center_x - half_width)
    expanded_y0 = max(0.0, center_y - half_height)
    expanded_x1 = min(float(image_width - 1), center_x + half_width)
    expanded_y1 = min(float(image_height - 1), center_y + half_height)
    return expanded_x0, expanded_y0, expanded_x1, expanded_y1


def _build_mask_from_detection(depth_shape: tuple[int, int], detection: RawDetection) -> np.ndarray:
    if detection.mask is not None:
        mask = np.asarray(detection.mask).astype(bool)
        if mask.shape != depth_shape:
            raise ValueError("detection mask shape does not match depth shape")
        return mask

    height, width = depth_shape
    x0 = max(0, min(width, int(round(detection.bbox[0]))))
    y0 = max(0, min(height, int(round(detection.bbox[1]))))
    x1 = max(0, min(width, int(round(detection.bbox[2]))))
    y1 = max(0, min(height, int(round(detection.bbox[3]))))
    mask = np.zeros((height, width), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def _compute_detection_depth_statistics(observation, detection: RawDetection) -> dict[str, float]:
    if observation.depth is None:
        raise ValueError("depth statistics require observation.depth")
    depth = np.asarray(observation.depth, dtype=np.float32)
    mask = _build_mask_from_detection(depth.shape, detection)
    valid = mask & np.isfinite(depth) & (depth >= float(DETECTION_DEPTH_MIN_M))
    if not np.any(valid):
        raise ValueError("depth statistics could not find valid depth inside detection mask")
    depths = depth[valid]
    saturation_threshold = float(DETECTION_DEPTH_MAX_M - DETECTION_DEPTH_SATURATION_EPS_M)
    saturated_ratio = float(np.mean(depths >= saturation_threshold))
    return {
        "valid_count": float(depths.size),
        "median_depth": float(np.median(depths)),
        "min_depth": float(np.min(depths)),
        "max_depth": float(np.max(depths)),
        "saturated_ratio": saturated_ratio,
        "min_depth_m": float(DETECTION_DEPTH_MIN_M),
        "max_depth_m": float(DETECTION_DEPTH_MAX_M),
        "saturation_threshold_m": saturation_threshold,
    }


def _locate_from_rgbd_and_pose(observation, detection: RawDetection) -> Detection3D:
    if observation.depth is None:
        raise ValueError("locate requires observation.depth")
    if observation.intrinsics is None:
        raise ValueError("locate requires observation.intrinsics")
    if observation.T_cam_odom is None:
        raise ValueError("locate requires observation.T_cam_odom")

    depth = np.asarray(observation.depth, dtype=np.float32)
    mask = _build_mask_from_detection(depth.shape, detection)
    valid = mask & np.isfinite(depth) & (depth >= float(DETECTION_DEPTH_MIN_M))
    if not np.any(valid):
        raise ValueError("locate could not find valid depth inside detection mask")

    ys, xs = np.nonzero(valid)
    depths = depth[valid]
    median_depth = float(np.median(depths))
    anchor_index = int(np.argmin(np.abs(depths - median_depth)))
    u = float(xs[anchor_index])
    v = float(ys[anchor_index])

    intrinsics = np.asarray(observation.intrinsics, dtype=np.float32)
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])

    x_cam = (u - cx) * median_depth / fx
    y_cam = (v - cy) * median_depth / fy
    z_cam = median_depth

    T_cam_odom = np.asarray(observation.T_cam_odom, dtype=np.float32)
    T_odom_cam = np.linalg.inv(T_cam_odom)
    point_cam = np.asarray([x_cam, y_cam, z_cam, 1.0], dtype=np.float32)
    point_odom = T_odom_cam @ point_cam
    return Detection3D(
        x=float(point_odom[0]),
        y=float(point_odom[1]),
        z=float(point_odom[2]),
    )


def compute_object_approach(cache: RuntimeCache, det_id: str, stop_distance: float) -> dict[str, object]:
    if stop_distance < 0.0:
        raise ValueError("stop_distance must be non-negative")

    detection_record = cache.get_detection(det_id)
    obs_id = detection_record.obs_id
    observation = cache.get_observation(obs_id).observation
    detection = detection_record.detection
    depth_stats = _compute_detection_depth_statistics(observation=observation, detection=detection)
    location = _locate_from_rgbd_and_pose(observation=observation, detection=detection)

    current_x = float(observation.pose.x)
    current_y = float(observation.pose.y)
    target_x = float(location.x)
    target_y = float(location.y)
    delta_x = target_x - current_x
    delta_y = target_y - current_y
    distance_xy = math.hypot(delta_x, delta_y)
    yaw = float(math.degrees(math.atan2(delta_y, delta_x)))

    proxy_move_due_to_max_depth = (
        float(depth_stats["median_depth"]) >= float(depth_stats["saturation_threshold_m"])
        and float(depth_stats["saturated_ratio"]) >= float(DETECTION_DEPTH_PROXY_SATURATION_RATIO)
    )
    proxy_move_direction_available = distance_xy > 1e-6
    if proxy_move_due_to_max_depth:
        move_distance = float(DETECTION_PROXY_MOVE_STEP_M) if proxy_move_direction_available else 0.0
    elif distance_xy <= stop_distance:
        move_distance = 0.0
    else:
        move_distance = distance_xy - stop_distance

    if not proxy_move_direction_available:
        goal_x = current_x
        goal_y = current_y
    else:
        scale = move_distance / distance_xy
        goal_x = current_x + delta_x * scale
        goal_y = current_y + delta_y * scale

    return {
        "obs_id": obs_id,
        "det_id": det_id,
        "class_name": detection.class_name,
        "current_pose": observation.pose.to_dict(),
        "target_position": location.to_dict(),
        "proxy_move_due_to_max_depth": bool(proxy_move_due_to_max_depth),
        "proxy_move_step_m": (
            float(DETECTION_PROXY_MOVE_STEP_M)
            if proxy_move_due_to_max_depth
            else None
        ),
        "proxy_move_direction_available": bool(proxy_move_direction_available),
        "planned_move_distance_m": float(move_distance),
        "estimated_distance_xy_m": float(distance_xy),
        "depth_stats": {
            "valid_count": int(depth_stats["valid_count"]),
            "median_depth": float(depth_stats["median_depth"]),
            "min_depth": float(depth_stats["min_depth"]),
            "max_depth": float(depth_stats["max_depth"]),
            "saturated_ratio": float(depth_stats["saturated_ratio"]),
            "min_depth_m": float(depth_stats["min_depth_m"]),
            "max_depth_m": float(depth_stats["max_depth_m"]),
            "saturation_threshold_m": float(depth_stats["saturation_threshold_m"]),
        },
        "goal_pose": {
            "x": float(goal_x),
            "y": float(goal_y),
            "yaw": yaw,
        },
    }
