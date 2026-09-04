from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable

import numpy as np

from navclaw.env.interface import RawDetection, RawObservation
from navclaw.perception import geometry as perception
from navclaw.perception.landmark_buffer import (
    LANDMARK_BUFFER_MERGE_DISTANCE_M,
    LandmarkBuffer,
    LandmarkBufferCandidate,
)
from navclaw.perception.landmark_verification import (
    verification_attributes,
    verify_landmark_candidate,
)
from navclaw.schemas import ActionCall, ActionResult
from navclaw.runtime.cache import RuntimeCache

if TYPE_CHECKING:
    from navclaw.llm.client import LLMClient
    from navclaw.perception.detectors.interface import DetectorInterface


LANDMARK_DETECTOR_BOX_THRESHOLD = 0.5
LANDMARK_CLASS_MAP: dict[str, dict[str, object]] = {
    "door": {
        "class_name": "door",
        "threshold": LANDMARK_DETECTOR_BOX_THRESHOLD,
        "merge_distance_m": 0.5,
        "geometry": "bbox_corners",
        "verification_question": (
            "Is the target inside the red bounding box likely to be a door-related landmark, "
            "such as a door, doorway, door frame, or partially visible door, even if it is "
            "small, occluded, or hard to read?"
        ),
    },
    "doorway": {
        "class_name": "door",
        "threshold": LANDMARK_DETECTOR_BOX_THRESHOLD,
        "merge_distance_m": 0.5,
        "geometry": "bbox_corners",
        "verification_question": (
            "Is the target inside the red bounding box likely to be a door-related landmark, "
            "such as a door, doorway, door frame, or partially visible door, even if it is "
            "small, occluded, or hard to read?"
        ),
    },
    "directional sign": {
        "class_name": "directional sign",
        "threshold": LANDMARK_DETECTOR_BOX_THRESHOLD,
        "merge_distance_m": 0.3,
        "geometry": "bbox_corners",
        "verification_question": (
            "Is the target inside the red bounding box likely to be a navigation-relevant sign, "
            "guide, map, or room/area information board, even if the text is too small or unreadable? "
            "Return False if the sign is just a room number sign."
        ),
        "reject_near_class_name": "door",
        "reject_near_distance_m": 1.4,
    },
}
LANDMARK_DETECTION_MAX_RESULTS = 8
LANDMARK_EDGE_MARGIN_PX = 4
LANDMARK_SAME_FRAME_IOU_THRESHOLD = 0.5
LANDMARK_PLANE_MAX_RESIDUAL_M = 0.3
LANDMARK_DEPTH_MIN_M = 0.1
LANDMARK_DEPTH_MAX_M = 4.5
LANDMARK_GEOMETRY_BBOX_CORNERS = "bbox_corners"
LANDMARK_GEOMETRY_BBOX_CENTER = "bbox_center"
LANDMARK_GEOMETRY_PROVIDED_POSITION = "provided_position"
SYSTEM_DOOR_CLASS_NAME = "door"
ROOM_LABEL_CLASS_NAME = "room label"
LANDMARK_QUERY_CLASS_NAMES = list(LANDMARK_CLASS_MAP.keys())


class LandmarkDetectionController:
    def __init__(
        self,
        *,
        detector: DetectorInterface,
        cache_provider: Callable[[], RuntimeCache | None],
        history_len_provider: Callable[[], int],
        landmark_class_map: dict[str, object] | None = None,
        llm_client: "LLMClient | None" = None,
        auto_confirm_without_llm: bool = False,
    ) -> None:
        self.detector = detector
        self._cache_provider = cache_provider
        self._history_len_provider = history_len_provider
        self.landmark_class_map = _normalize_landmark_class_map(landmark_class_map)
        self.llm_client = llm_client
        self.auto_confirm_without_llm = bool(auto_confirm_without_llm)
        self.buffer = LandmarkBuffer(
            merge_distance_by_class=_merge_distance_by_system_class(self.landmark_class_map)
        )
        self._detections_by_obs_id: dict[str, list[dict[str, object]]] = {}
        self.semantic_annotations_only = False
        self.enabled = True

    def is_active(self) -> bool:
        return bool(self.enabled) and self.landmark_class_map != {}

    def configure_landmark_class_map(
        self,
        landmark_class_map: dict[str, object],
        *,
        reset_buffer: bool = True,
        confirmation_min_observations: int | None = None,
        auto_confirm_without_llm: bool | None = None,
        semantic_annotations_only: bool | None = None,
    ) -> None:
        self.landmark_class_map = _normalize_landmark_class_map(landmark_class_map)
        if auto_confirm_without_llm is not None:
            self.auto_confirm_without_llm = bool(auto_confirm_without_llm)
        if semantic_annotations_only is not None:
            self.semantic_annotations_only = bool(semantic_annotations_only)
        if bool(reset_buffer):
            self.buffer = LandmarkBuffer(
                merge_distance_by_class=_merge_distance_by_system_class(self.landmark_class_map),
                confirmation_min_observations=(
                    self.buffer.confirmation_min_observations
                    if confirmation_min_observations is None
                    else int(confirmation_min_observations)
                ),
            )
            self._detections_by_obs_id = {}

    def detections_for_obs_ids(self, obs_ids: list[str]) -> dict[str, list[dict[str, object]]]:
        return {
            str(obs_id): [dict(item) for item in self._detections_by_obs_id.get(str(obs_id), [])]
            for obs_id in obs_ids
        }

    def _require_cache(self) -> RuntimeCache:
        cache = self._cache_provider()
        if cache is None:
            raise ValueError("landmark detection requires runtime cache")
        return cache

    def update_for_obs_id(self, obs_id: str, source: str) -> dict[str, object] | None:
        if not self.is_active():
            return None
        cache = self._require_cache()
        if bool(self.semantic_annotations_only):
            return self.update_from_detect_result(
                detect_result=_semantic_annotation_detect_result(cache=cache, obs_id=str(obs_id)),
                source=source,
            )
        query_class_names = list(self.landmark_class_map.keys())
        detect_result = perception.detect(
            self.detector,
            cache,
            class_name=query_class_names[0],
            synonym_class_names=query_class_names[1:],
            canonical_class_name=None,
            canonical_class_map={
                query_class_name: str(config["class_name"])
                for query_class_name, config in self.landmark_class_map.items()
            },
            obs_id=str(obs_id),
            min_score=_min_landmark_threshold(self.landmark_class_map),
            agnostic_nms=False,
            max_results=LANDMARK_DETECTION_MAX_RESULTS,
            overlay_bbox_scale=1.0,
        )
        return self.update_from_detect_result(detect_result=detect_result, source=source)

    def update_from_detect_result(
        self,
        *,
        detect_result: ActionResult,
        source: str,
    ) -> dict[str, object]:
        cache = self._require_cache()
        obs_id = detect_result.data.get("obs_id")
        if obs_id is None:
            raise ValueError("landmark detection update requires detect_result.data.obs_id")
        raw_detections = detect_result.data.get("detections", [])
        if not isinstance(raw_detections, list):
            raise ValueError("landmark detection update requires list data.detections")

        step_index = self._history_len_provider()
        skipped_detections: list[dict[str, object]] = []
        mapped_detections: list[tuple[dict[str, object], dict[str, object]]] = []
        for raw_detection in raw_detections:
            if not isinstance(raw_detection, dict):
                continue
            detection = dict(raw_detection)
            detection["obs_id"] = str(obs_id)
            detector_class_name = _normalize_landmark_class_name(
                detection.get("detector_class_name", detection.get("class_name", ""))
            )
            class_config = self.landmark_class_map.get(detector_class_name)
            if class_config is None:
                continue
            if float(detection.get("score", 0.0)) < float(class_config["threshold"]):
                continue
            canonical_class_name = str(class_config["class_name"])
            detection["class_name"] = canonical_class_name
            detection["geometry"] = str(class_config["geometry"])
            if "near_class_name" in class_config:
                detection["near_class_name"] = str(class_config["near_class_name"])
            if "near_distance_m" in class_config:
                detection["near_distance_m"] = float(class_config["near_distance_m"])
            if "reject_near_class_name" in class_config:
                detection["reject_near_class_name"] = str(class_config["reject_near_class_name"])
            if "reject_near_distance_m" in class_config:
                detection["reject_near_distance_m"] = float(class_config["reject_near_distance_m"])
            mapped_detections.append((detection, class_config))

        door_detections = [
            item
            for item in mapped_detections
            if str(item[0].get("class_name", "")) == SYSTEM_DOOR_CLASS_NAME
        ]
        non_door_detections = [
            item
            for item in mapped_detections
            if str(item[0].get("class_name", ""))
            not in (
                {SYSTEM_DOOR_CLASS_NAME}
                if bool(self.semantic_annotations_only)
                else {SYSTEM_DOOR_CLASS_NAME, ROOM_LABEL_CLASS_NAME}
            )
        ]
        label_detections = [
            item
            for item in mapped_detections
            if (
                not bool(self.semantic_annotations_only)
                and str(item[0].get("class_name", "")) == ROOM_LABEL_CLASS_NAME
            )
        ]

        valid_detections: list[dict[str, object]] = []
        duplicate_detections: list[dict[str, object]] = []
        buffered_candidates: dict[str, LandmarkBufferCandidate] = {}
        current_obs_detections: list[dict[str, object]] = []

        door_valid = self._detections_with_geometry(
            detections=door_detections,
            cache=cache,
            obs_id=str(obs_id),
            skipped_detections=skipped_detections,
        )
        door_valid = _filter_blacklisted_landmark_detections(
            detections=door_valid,
            buffer=self.buffer,
            skipped_detections=skipped_detections,
        )
        valid_detections.extend(door_valid)
        kept_door_detections, door_duplicates = _dedupe_same_frame_landmarks(
            door_valid,
            buffer=self.buffer,
        )
        duplicate_detections.extend(door_duplicates)
        for detection in kept_door_detections:
            candidate = self.buffer.add_detection(
                detection=detection,
                source=source,
                step_index=step_index,
            )
            _apply_semantic_detection_attributes(candidate, detection)
            buffered_candidates[str(candidate.landmark_id)] = candidate
            current_obs_detections.append(_landmark_detection_context_payload(detection, candidate))

        non_door_valid = self._detections_with_geometry(
            detections=non_door_detections,
            cache=cache,
            obs_id=str(obs_id),
            skipped_detections=skipped_detections,
        )
        non_door_valid = _filter_blacklisted_landmark_detections(
            detections=non_door_valid,
            buffer=self.buffer,
            skipped_detections=skipped_detections,
        )
        filtered_non_door_valid: list[dict[str, object]] = []
        for detection in non_door_valid:
            reject_reason = _near_class_rejection_reason(
                detection=detection,
                buffer=self.buffer,
            )
            if reject_reason is not None:
                skipped_detections.append(
                    {
                        "det_id": str(detection.get("det_id", "") or ""),
                        "reason": str(reject_reason),
                    }
                )
                continue
            filtered_non_door_valid.append(detection)
        valid_detections.extend(filtered_non_door_valid)

        kept_non_door_detections, non_door_duplicates = _dedupe_same_frame_landmarks(
            filtered_non_door_valid,
            buffer=self.buffer,
        )
        duplicate_detections.extend(non_door_duplicates)
        for detection in kept_non_door_detections:
            candidate = self.buffer.add_detection(
                detection=detection,
                source=source,
                step_index=step_index,
            )
            _apply_semantic_detection_attributes(candidate, detection)
            buffered_candidates[str(candidate.landmark_id)] = candidate
            current_obs_detections.append(_landmark_detection_context_payload(detection, candidate))

        label_valid = self._detections_with_geometry(
            detections=label_detections,
            cache=cache,
            obs_id=str(obs_id),
            skipped_detections=skipped_detections,
        )
        for detection in label_valid:
            skipped_detections.append(
                {
                    "det_id": str(detection.get("det_id", "") or ""),
                    "reason": "room_label_ignored",
                }
            )

        verification_results = self._verify_candidates_ready_for_confirmation(cache=cache)
        self._detections_by_obs_id[str(obs_id)] = current_obs_detections
        return {
            "source": str(source),
            "obs_id": str(obs_id),
            "landmark_class_map": _landmark_class_map_to_dict(self.landmark_class_map),
            "query_classes": list(self.landmark_class_map.keys()),
            "canonical_class_names": sorted(
                {str(config["class_name"]) for config in self.landmark_class_map.values()}
            ),
            "min_score": float(_min_landmark_threshold(self.landmark_class_map)),
            "raw_detection_count": int(len(raw_detections)),
            "valid_detection_count": int(len(valid_detections)),
            "buffered_detection_count": int(
                len(kept_door_detections) + len(kept_non_door_detections)
            ),
            "skipped_detections": skipped_detections,
            "same_frame_duplicate_detections": duplicate_detections,
            "landmark_verification_results": verification_results,
            "buffered_candidates": [
                candidate.to_dict()
                for candidate in buffered_candidates.values()
            ],
            "buffer": self.buffer.to_context(step_index=step_index),
        }

    def _verify_candidates_ready_for_confirmation(
        self,
        *,
        cache: RuntimeCache,
    ) -> list[dict[str, object]]:
        if self.llm_client is None:
            if bool(self.auto_confirm_without_llm):
                results: list[dict[str, object]] = []
                for candidate in self.buffer.candidates_needing_verification():
                    updated = self.buffer.mark_verified(str(candidate.landmark_id), attributes={})
                    results.append(
                        {
                            "landmark_id": str(updated.landmark_id),
                            "class_name": str(updated.class_name),
                            "accepted": True,
                            "status": str(updated.status),
                            "verification": {"source": "auto_confirm_without_llm"},
                        }
                    )
                return results
            return []
        results: list[dict[str, object]] = []
        for candidate in self.buffer.candidates_needing_verification():
            result = verify_landmark_candidate(
                candidate=candidate,
                cache=cache,
                llm_client=self.llm_client,
                landmark_class_map=self.landmark_class_map,
            )
            attributes = verification_attributes(result)
            if result.accepted:
                updated = self.buffer.mark_verified(
                    str(candidate.landmark_id),
                    attributes=attributes,
                )
                results.append(
                    {
                        "landmark_id": str(updated.landmark_id),
                        "class_name": str(updated.class_name),
                        "accepted": True,
                        "status": str(updated.status),
                        "verification": result.to_dict(),
                    }
                )
                continue
            blacklisted = self.buffer.reject_candidate(
                str(candidate.landmark_id),
                attributes=attributes,
                rejected_mode="semantic_verification",
                blacklist_reason="landmark_semantic_verification_false_positive",
            )
            results.append(
                {
                    "landmark_id": str(candidate.landmark_id),
                    "class_name": str(candidate.class_name),
                    "accepted": False,
                    "status": "blacklisted",
                    "verification": result.to_dict(),
                    "blacklist_entry": dict(blacklisted),
                }
            )
        return results

    def _detections_with_geometry(
        self,
        *,
        detections: list[tuple[dict[str, object], dict[str, object]]],
        cache: RuntimeCache,
        obs_id: str,
        skipped_detections: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        valid_detections: list[dict[str, object]] = []
        for detection, class_config in detections:
            geometry, skip_reason = _landmark_geometry_from_detection(
                cache=cache,
                obs_id=obs_id,
                det_id=str(detection.get("det_id", "") or ""),
                geometry=str(class_config["geometry"]),
            )
            if geometry is None:
                skipped_detections.append(
                    {
                        "det_id": str(detection.get("det_id", "") or ""),
                        "reason": str(skip_reason),
                    }
                )
                continue
            detection.update(geometry)
            valid_detections.append(detection)
        return valid_detections

    def handle_post_action_execute(self, call: ActionCall, result: ActionResult) -> None:
        if call.action != "get_obs":
            return
        obs_id = result.data.get("obs_id")
        if obs_id is None:
            return
        update = self.update_for_obs_id(obs_id=str(obs_id), source="get_obs")
        if update is not None:
            result.data["landmark_detection_buffer_update"] = update

    def handle_step_observation(self, observation: RawObservation, source: str) -> dict[str, object] | None:
        if not self.is_active():
            return None
        cache = self._require_cache()
        record = cache.store_observation(observation)
        update = self.update_for_obs_id(obs_id=record.id, source=source)
        payload: dict[str, object] = {
            "obs_id": record.id,
            "rgb_id": f"{record.id}:rgb",
            "depth_id": f"{record.id}:depth",
            "pose": observation.pose.to_dict(),
            "source": str(source),
        }
        if update is not None:
            payload["landmark_detection_buffer_update"] = update
        return payload


def _normalize_landmark_class_name(value: object) -> str:
    return str(value).strip().replace("_", " ")


def _landmark_detection_context_payload(
    detection: dict[str, object],
    candidate: LandmarkBufferCandidate,
) -> dict[str, object]:
    payload = {
        "landmark_id": str(candidate.landmark_id),
        "class_name": str(candidate.class_name),
        "status": str(candidate.status),
        "obs_id": str(detection.get("obs_id", "") or ""),
        "det_id": str(detection.get("det_id", "") or ""),
        "bbox": [float(value) for value in list(detection.get("bbox", []))],
        "score": float(detection.get("score", 0.0)),
        "position": dict(detection.get("position", candidate.position)),
    }
    if isinstance(detection.get("semantic_attributes"), dict):
        payload["semantic_attributes"] = dict(detection["semantic_attributes"])
    return payload


def _normalize_landmark_class_map(
    landmark_class_map: dict[str, object] | None,
) -> dict[str, dict[str, object]]:
    raw_map = LANDMARK_CLASS_MAP if landmark_class_map is None else landmark_class_map
    normalized: dict[str, dict[str, object]] = {}
    for query_class_name, raw_config in raw_map.items():
        query = _normalize_landmark_class_name(query_class_name)
        if isinstance(raw_config, str):
            class_name = _normalize_landmark_class_name(raw_config)
            threshold = LANDMARK_DETECTOR_BOX_THRESHOLD
            merge_distance_m = LANDMARK_BUFFER_MERGE_DISTANCE_M
            geometry = LANDMARK_GEOMETRY_BBOX_CENTER
            near_class_name = None
            near_distance_m = None
            reject_near_class_name = None
            reject_near_distance_m = None
            verification_question = None
        else:
            class_name = _normalize_landmark_class_name(raw_config.get("class_name", ""))
            threshold = float(raw_config.get("threshold", LANDMARK_DETECTOR_BOX_THRESHOLD))
            merge_distance_m = float(
                raw_config.get("merge_distance_m", LANDMARK_BUFFER_MERGE_DISTANCE_M)
            )
            geometry = str(raw_config.get("geometry", LANDMARK_GEOMETRY_BBOX_CENTER)).strip()
            near_class_name = raw_config.get("near_class_name")
            near_distance_m = raw_config.get("near_distance_m")
            reject_near_class_name = raw_config.get("reject_near_class_name")
            reject_near_distance_m = raw_config.get("reject_near_distance_m")
            verification_question = raw_config.get("verification_question")
        normalized[query] = {
            "class_name": class_name,
            "threshold": float(threshold),
            "merge_distance_m": float(merge_distance_m),
            "geometry": geometry,
        }
        if verification_question is not None and str(verification_question).strip() != "":
            normalized[query]["verification_question"] = str(verification_question).strip()
        if near_class_name is not None:
            normalized_near_class_name = _normalize_landmark_class_name(near_class_name)
            normalized[query]["near_class_name"] = normalized_near_class_name
            normalized[query]["near_distance_m"] = float(near_distance_m)
        if reject_near_class_name is not None:
            normalized_reject_class_name = _normalize_landmark_class_name(reject_near_class_name)
            normalized[query]["reject_near_class_name"] = normalized_reject_class_name
            normalized[query]["reject_near_distance_m"] = float(reject_near_distance_m)
    return normalized


def _min_landmark_threshold(landmark_class_map: dict[str, dict[str, object]]) -> float:
    return min(float(config["threshold"]) for config in landmark_class_map.values())


def _merge_distance_by_system_class(
    landmark_class_map: dict[str, dict[str, object]],
) -> dict[str, float]:
    return {
        str(config["class_name"]): float(config["merge_distance_m"])
        for config in landmark_class_map.values()
    }


def _landmark_class_map_to_dict(
    landmark_class_map: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    return {
        str(query_class): _landmark_class_config_to_dict(config)
        for query_class, config in landmark_class_map.items()
    }


def _landmark_class_config_to_dict(config: dict[str, object]) -> dict[str, object]:
    payload: dict[str, object] = {
        "class_name": str(config["class_name"]),
        "threshold": float(config["threshold"]),
        "merge_distance_m": float(config["merge_distance_m"]),
        "geometry": str(config["geometry"]),
    }
    if "near_class_name" in config:
        payload["near_class_name"] = str(config["near_class_name"])
    if "near_distance_m" in config:
        payload["near_distance_m"] = float(config["near_distance_m"])
    if "reject_near_class_name" in config:
        payload["reject_near_class_name"] = str(config["reject_near_class_name"])
    if "reject_near_distance_m" in config:
        payload["reject_near_distance_m"] = float(config["reject_near_distance_m"])
    if "verification_question" in config:
        payload["verification_question"] = str(config["verification_question"])
    return payload


def _landmark_geometry_from_detection(
    *,
    cache: RuntimeCache,
    obs_id: str,
    det_id: str,
    geometry: str,
) -> tuple[dict[str, object] | None, str | None]:
    if geometry == LANDMARK_GEOMETRY_PROVIDED_POSITION:
        detection_record = cache.get_detection(str(det_id))
        position = detection_record.detection.metadata.get("position")
        if not isinstance(position, dict):
            return None, "missing_provided_position"
        return {
            "position": {
                "x": float(position["x"]),
                "y": float(position["y"]),
                "z": float(position.get("z", 0.0)),
            },
            "geometry": LANDMARK_GEOMETRY_PROVIDED_POSITION,
        }, None
    if geometry == LANDMARK_GEOMETRY_BBOX_CORNERS:
        return _landmark_corner_geometry_from_detection(
            cache=cache,
            obs_id=obs_id,
            det_id=det_id,
        )
    if geometry == LANDMARK_GEOMETRY_BBOX_CENTER:
        return _landmark_center_geometry_from_detection(
            cache=cache,
            obs_id=obs_id,
            det_id=det_id,
        )
    return _landmark_center_geometry_from_detection(
        cache=cache,
        obs_id=obs_id,
        det_id=det_id,
    )


def _semantic_annotation_detect_result(*, cache: RuntimeCache, obs_id: str) -> ActionResult:
    observation = cache.get_observation(str(obs_id)).observation
    annotations = [
        dict(item)
        for item in observation.semantic_annotations
        if isinstance(item, dict)
    ]
    raw_detections = [
        RawDetection(
            class_name=str(item.get("class_name", "")),
            bbox=[float(value) for value in list(item.get("bbox", []))],
            score=float(item.get("score", 1.0)),
            metadata={
                "detector_class_name": str(item.get("class_name", "")),
                "semantic_id": str(item.get("semantic_id", "")),
                "position": dict(item.get("position", {})),
                "semantic_attributes": dict(item.get("semantic_attributes", {})),
            },
        )
        for item in annotations
    ]
    records = cache.store_detections(obs_id=str(obs_id), detections=raw_detections)
    payloads: list[dict[str, object]] = []
    for record in records:
        metadata = dict(record.detection.metadata)
        payloads.append(
            {
                "det_id": str(record.id),
                "class_name": str(record.detection.class_name),
                "detector_class_name": str(metadata["detector_class_name"]),
                "bbox": [float(value) for value in record.detection.bbox],
                "score": float(record.detection.score),
                "semantic_id": str(metadata["semantic_id"]),
                "semantic_attributes": dict(metadata["semantic_attributes"]),
            }
        )
    return ActionResult(
        ok=True,
        data={"obs_id": str(obs_id), "detections": payloads},
        message=f"Loaded {len(payloads)} ReasonNav semantic annotations.",
    )


def _apply_semantic_detection_attributes(
    candidate: LandmarkBufferCandidate,
    detection: dict[str, object],
) -> None:
    attributes = detection.get("semantic_attributes")
    if not isinstance(attributes, dict):
        return
    candidate.attributes.update(dict(attributes))
    semantic_id = str(detection.get("semantic_id", "") or "")
    if semantic_id != "":
        candidate.attributes["semantic_id"] = semantic_id


def _landmark_corner_geometry_from_detection(
    *,
    cache: RuntimeCache,
    obs_id: str,
    det_id: str,
) -> tuple[dict[str, object] | None, str | None]:
    observation = cache.get_observation(str(obs_id)).observation
    detection_record = cache.get_detection(str(det_id))
    bbox = [float(value) for value in detection_record.detection.bbox]
    image_shape = _observation_image_shape(observation)
    if _bbox_touches_image_edge(
        bbox=bbox,
        image_width=image_shape[1],
        image_height=image_shape[0],
        margin_px=LANDMARK_EDGE_MARGIN_PX,
    ):
        return None, "bbox_touches_image_edge"
    corner_points, corner_skip_reason = _project_bbox_corners(observation=observation, bbox=bbox)
    if corner_points is None:
        return None, str(corner_skip_reason or "invalid_corner_depth")
    plane_residual = _plane_max_residual_m(corner_points)
    if plane_residual > LANDMARK_PLANE_MAX_RESIDUAL_M:
        return None, "nonplanar_corners"
    position = {
        "x": float(np.mean([point["x"] for point in corner_points])),
        "y": float(np.mean([point["y"] for point in corner_points])),
        "z": float(np.mean([point["z"] for point in corner_points])),
    }
    return {
        "position": position,
        "corner_points": corner_points,
        "plane_max_residual_m": float(plane_residual),
        "geometry": LANDMARK_GEOMETRY_BBOX_CORNERS,
    }, None


def _landmark_center_geometry_from_detection(
    *,
    cache: RuntimeCache,
    obs_id: str,
    det_id: str,
) -> tuple[dict[str, object] | None, str | None]:
    observation = cache.get_observation(str(obs_id)).observation
    detection_record = cache.get_detection(str(det_id))
    bbox = [float(value) for value in detection_record.detection.bbox]
    center_point = _project_bbox_center(observation=observation, bbox=bbox)
    if center_point is None:
        return None, "invalid_center_depth"
    return {
        "position": dict(center_point),
        "center_point": dict(center_point),
        "geometry": LANDMARK_GEOMETRY_BBOX_CENTER,
    }, None


def _observation_image_shape(observation: RawObservation) -> tuple[int, int]:
    if observation.rgb is not None:
        rgb = np.asarray(observation.rgb)
        return int(rgb.shape[0]), int(rgb.shape[1])
    depth = np.asarray(observation.depth)
    return int(depth.shape[0]), int(depth.shape[1])


def _bbox_touches_image_edge(
    *,
    bbox: list[float],
    image_width: int,
    image_height: int,
    margin_px: int,
) -> bool:
    x0, y0, x1, y1 = [float(value) for value in bbox]
    margin = float(margin_px)
    return bool(
        x0 <= margin
        or y0 <= margin
        or x1 >= float(image_width - 1) - margin
        or y1 >= float(image_height - 1) - margin
    )


def _project_bbox_corners(
    *,
    observation: RawObservation,
    bbox: list[float],
) -> tuple[list[dict[str, float]] | None, str | None]:
    depth, intrinsics, T_odom_cam = _projection_inputs(observation)
    x0, y0, x1, y1 = [float(value) for value in bbox]
    corners = [
        (x0, y0),
        (x1, y0),
        (x1, y1),
        (x0, y1),
    ]
    points: list[dict[str, float]] = []
    for u, v in corners:
        depth_m = _pixel_depth_m(depth=depth, u=float(u), v=float(v))
        if depth_m is None:
            return None, "invalid_corner_depth"
        if depth_m > LANDMARK_DEPTH_MAX_M:
            return None, "corner_depth_exceeds_max"
        point = _project_pixel_to_odom(
            depth=depth,
            intrinsics=intrinsics,
            T_odom_cam=T_odom_cam,
            u=float(u),
            v=float(v),
        )
        if point is None:
            return None, "invalid_corner_depth"
        points.append(point)
    return points, None


def _project_bbox_center(
    *,
    observation: RawObservation,
    bbox: list[float],
) -> dict[str, float] | None:
    depth, intrinsics, T_odom_cam = _projection_inputs(observation)
    x0, y0, x1, y1 = [float(value) for value in bbox]
    return _project_pixel_to_odom(
        depth=depth,
        intrinsics=intrinsics,
        T_odom_cam=T_odom_cam,
        u=(x0 + x1) / 2.0,
        v=(y0 + y1) / 2.0,
    )


def _projection_inputs(observation: RawObservation) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    depth = np.asarray(observation.depth, dtype=np.float32)
    intrinsics = np.asarray(observation.intrinsics, dtype=np.float32)
    T_cam_odom = np.asarray(observation.T_cam_odom, dtype=np.float32)
    T_odom_cam = np.linalg.inv(T_cam_odom)
    return depth, intrinsics, T_odom_cam


def _project_pixel_to_odom(
    *,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    T_odom_cam: np.ndarray,
    u: float,
    v: float,
) -> dict[str, float] | None:
    depth_m = _pixel_depth_m(depth=depth, u=float(u), v=float(v))
    if depth_m is None:
        return None
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    x_cam = (float(u) - cx) * depth_m / fx
    y_cam = (float(v) - cy) * depth_m / fy
    point_cam = np.asarray([x_cam, y_cam, depth_m, 1.0], dtype=np.float32)
    point_odom = T_odom_cam @ point_cam
    return {
        "x": float(point_odom[0]),
        "y": float(point_odom[1]),
        "z": float(point_odom[2]),
    }


def _pixel_depth_m(*, depth: np.ndarray, u: float, v: float) -> float | None:
    height, width = depth.shape
    px = int(round(float(u)))
    py = int(round(float(v)))
    if px < 0 or py < 0 or px >= width or py >= height:
        return None
    depth_m = float(depth[py, px])
    if not math.isfinite(depth_m) or depth_m < float(LANDMARK_DEPTH_MIN_M):
        return None
    return depth_m


def _plane_max_residual_m(points: list[dict[str, float]]) -> float:
    point_array = np.asarray(
        [
            [float(point["x"]), float(point["y"]), float(point["z"])]
            for point in points
        ],
        dtype=np.float64,
    )
    center = np.mean(point_array, axis=0)
    centered = point_array - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    distances = np.abs(centered @ normal)
    return float(np.max(distances))


def _dedupe_same_frame_landmarks(
    detections: list[dict[str, object]],
    *,
    buffer: LandmarkBuffer,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    sorted_detections = sorted(
        detections,
        key=lambda item: (-float(item.get("score", 0.0)), str(item.get("det_id", ""))),
    )
    kept: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    for detection in sorted_detections:
        duplicate_of = _same_frame_duplicate_of(
            detection=detection,
            kept=kept,
            buffer=buffer,
        )
        if duplicate_of is not None:
            skipped.append(
                {
                    "det_id": str(detection.get("det_id", "") or ""),
                    "duplicate_of_det_id": str(duplicate_of.get("det_id", "") or ""),
                    "reason": "same_frame_duplicate",
                }
            )
            continue
        kept.append(detection)
    return kept, skipped


def _filter_blacklisted_landmark_detections(
    *,
    detections: list[dict[str, object]],
    buffer: LandmarkBuffer,
    skipped_detections: list[dict[str, object]],
) -> list[dict[str, object]]:
    kept: list[dict[str, object]] = []
    for detection in detections:
        position = dict(detection.get("position", {}))
        match = buffer.blacklist_match(
            class_name=str(detection.get("class_name", "") or ""),
            position={
                "x": float(position["x"]),
                "y": float(position["y"]),
                "z": float(position.get("z", 0.0)),
            },
        )
        if match is not None:
            skipped_detections.append(
                {
                    "det_id": str(detection.get("det_id", "") or ""),
                    "reason": "landmark_blacklisted",
                    "blacklist_match": dict(match),
                }
            )
            continue
        kept.append(detection)
    return kept


def _same_frame_duplicate_of(
    *,
    detection: dict[str, object],
    kept: list[dict[str, object]],
    buffer: LandmarkBuffer,
) -> dict[str, object] | None:
    bbox = [float(value) for value in list(detection.get("bbox", []))]
    position = dict(detection.get("position", {}))
    class_name = str(detection.get("class_name", "") or "")
    merge_distance_m = buffer.merge_distance_for_class(class_name)
    for existing in kept:
        if str(existing.get("class_name", "") or "") != class_name:
            continue
        existing_bbox = [float(value) for value in list(existing.get("bbox", []))]
        if _bbox_iou(bbox, existing_bbox) >= LANDMARK_SAME_FRAME_IOU_THRESHOLD:
            return existing
        existing_position = dict(existing.get("position", {}))
        if _position_distance_xy(position, existing_position) <= merge_distance_m:
            return existing
    return None


def _near_class_rejection_reason(
    *,
    detection: dict[str, object],
    buffer: LandmarkBuffer,
) -> str | None:
    reject_class_name = detection.get("reject_near_class_name")
    if reject_class_name is None:
        return None
    position = dict(detection.get("position", {}))
    max_distance_m = float(detection.get("reject_near_distance_m", 0.0))
    distance = buffer.min_distance_to_class(
        class_name=_normalize_landmark_class_name(reject_class_name),
        position={
            "x": float(position["x"]),
            "y": float(position["y"]),
            "z": float(position.get("z", 0.0)),
        },
    )
    if distance is None or distance > max_distance_m:
        return None
    class_name = _normalize_landmark_class_name(detection.get("class_name", "landmark"))
    return (
        f"{class_name.replace(' ', '_')}_near_"
        f"{_normalize_landmark_class_name(reject_class_name).replace(' ', '_')}"
    )


def _position_distance_xy(a: dict[str, object], b: dict[str, object]) -> float:
    return float(math.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"])))


def _bbox_iou(a: list[float], b: list[float]) -> float:
    ax0, ay0, ax1, ay1 = [float(value) for value in a]
    bx0, by0, bx1, by1 = [float(value) for value in b]
    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)
    inter_w = max(0.0, inter_x1 - inter_x0)
    inter_h = max(0.0, inter_y1 - inter_y0)
    intersection = inter_w * inter_h
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - intersection
    if union <= 0.0:
        return 0.0
    return float(intersection / union)
