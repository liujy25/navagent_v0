from __future__ import annotations

from dataclasses import dataclass, field
import math

from navclaw.memory.entity_knowledge import EntityKnowledge
from navclaw.memory.entity_knowledge import merge_entity_knowledge


LANDMARK_BUFFER_MERGE_DISTANCE_M = 1.0
LANDMARK_BUFFER_CONFIRMATION_MIN_OBSERVATIONS = 4
LANDMARK_BUFFER_MAX_DETECTIONS_PER_ENTITY = 4
LANDMARK_STATUS_PENDING = "pending"
LANDMARK_STATUS_NEEDS_VERIFICATION = "needs_verification"
LANDMARK_STATUS_CONFIRMED = "confirmed"


@dataclass
class LandmarkBufferCandidate:
    landmark_id: str
    class_name: str
    status: str
    visited: bool
    first_step_index: int
    last_step_index: int
    position: dict[str, float]
    detections: list[dict[str, object]] = field(default_factory=list)
    attributes: dict[str, object] = field(default_factory=dict)
    knowledge: list[EntityKnowledge] = field(default_factory=list)

    def observation_count(self) -> int:
        return len(
            {
                str(item.get("obs_id", "") or "")
                for item in self.detections
                if str(item.get("obs_id", "") or "").strip() != ""
            }
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "landmark_id": str(self.landmark_id),
            "class_name": str(self.class_name),
            "status": str(self.status),
            "visited": bool(self.visited),
            "first_step_index": int(self.first_step_index),
            "last_step_index": int(self.last_step_index),
            "observation_count": int(self.observation_count()),
            "position": dict(self.position),
            "detections": [dict(item) for item in self.detections],
            "attributes": dict(self.attributes),
            "knowledge": [item.to_dict() for item in self.knowledge],
        }


class LandmarkBuffer:
    def __init__(
        self,
        *,
        merge_distance_m: float = LANDMARK_BUFFER_MERGE_DISTANCE_M,
        merge_distance_by_class: dict[str, float] | None = None,
        confirmation_min_observations: int = LANDMARK_BUFFER_CONFIRMATION_MIN_OBSERVATIONS,
        max_detections_per_entity: int = LANDMARK_BUFFER_MAX_DETECTIONS_PER_ENTITY,
    ) -> None:
        self.merge_distance_m = float(merge_distance_m)
        self.merge_distance_by_class = self._normalize_merge_distance_by_class(
            merge_distance_by_class
        )
        self.confirmation_min_observations = int(confirmation_min_observations)
        self.max_detections_per_entity = int(max_detections_per_entity)
        self._next_landmark_index = 0
        self._candidates: dict[str, LandmarkBufferCandidate] = {}
        self._blacklist: list[dict[str, object]] = []

    def reset(self) -> None:
        self._next_landmark_index = 0
        self._candidates = {}
        self._blacklist = []

    def _next_landmark_id(self) -> str:
        landmark_id = f"landmark_{self._next_landmark_index}"
        self._next_landmark_index += 1
        return landmark_id

    @staticmethod
    def _normalize_class_name(value: object) -> str:
        return str(value).strip().replace("_", " ")

    @classmethod
    def _normalize_merge_distance_by_class(
        cls,
        merge_distance_by_class: dict[str, float] | None,
    ) -> dict[str, float]:
        if merge_distance_by_class is None:
            return {}
        normalized: dict[str, float] = {}
        for class_name, distance in merge_distance_by_class.items():
            normalized_class_name = cls._normalize_class_name(class_name)
            normalized[normalized_class_name] = float(distance)
        return normalized

    @staticmethod
    def _normalize_position(value: object) -> dict[str, float]:
        return {
            "x": float(value["x"]),
            "y": float(value["y"]),
            "z": float(value.get("z", 0.0)),
        }

    @staticmethod
    def _distance_xy(a: dict[str, float], b: dict[str, float]) -> float:
        return float(math.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"])))

    def merge_distance_for_class(self, class_name: str) -> float:
        normalized_class_name = self._normalize_class_name(class_name)
        return float(self.merge_distance_by_class.get(normalized_class_name, self.merge_distance_m))

    def candidates_by_class(self, class_name: str) -> list[LandmarkBufferCandidate]:
        normalized_class_name = self._normalize_class_name(class_name)
        return [
            candidate
            for candidate in self._candidates.values()
            if self._normalize_class_name(candidate.class_name) == normalized_class_name
        ]

    def min_distance_to_class(
        self,
        *,
        class_name: str,
        position: dict[str, float],
    ) -> float | None:
        distances = [
            self._candidate_min_position_distance_xy(
                position=position,
                candidate=candidate,
            )
            for candidate in self.candidates_by_class(class_name)
        ]
        if distances == []:
            return None
        return float(min(distances))

    def _find_match(
        self,
        *,
        class_name: str,
        position: dict[str, float],
    ) -> tuple[LandmarkBufferCandidate, float] | None:
        best: LandmarkBufferCandidate | None = None
        best_distance = float("inf")
        class_text = self._normalize_class_name(class_name)
        merge_distance_m = self.merge_distance_for_class(class_text)
        for candidate in self._candidates.values():
            if self._normalize_class_name(candidate.class_name) != class_text:
                continue
            distance = self._candidate_min_position_distance_xy(
                position=position,
                candidate=candidate,
            )
            if distance > merge_distance_m:
                continue
            if distance >= best_distance:
                continue
            best = candidate
            best_distance = distance
        if best is None:
            return None
        return best, float(best_distance)

    def _candidate_min_position_distance_xy(
        self,
        *,
        position: dict[str, float],
        candidate: LandmarkBufferCandidate,
    ) -> float:
        positions = self._candidate_evidence_positions(candidate)
        return min(self._distance_xy(position, existing) for existing in positions)

    def _candidate_pair_min_position_distance_xy(
        self,
        a: LandmarkBufferCandidate,
        b: LandmarkBufferCandidate,
    ) -> float:
        a_positions = self._candidate_evidence_positions(a)
        b_positions = self._candidate_evidence_positions(b)
        return min(
            self._distance_xy(a_position, b_position)
            for a_position in a_positions
            for b_position in b_positions
        )

    def _candidate_evidence_positions(
        self,
        candidate: LandmarkBufferCandidate,
    ) -> list[dict[str, float]]:
        positions = [
            self._normalize_position(item["position"])
            for item in candidate.detections
            if isinstance(item.get("position"), dict)
        ]
        if positions == []:
            positions.append(self._normalize_position(candidate.position))
        return positions

    @staticmethod
    def _evidence_sort_key(evidence: dict[str, object]) -> tuple[float, int, str]:
        return (
            -float(evidence.get("score", 0.0)),
            -int(evidence.get("step_index", 0)),
            str(evidence.get("det_id", "")),
        )

    def _refresh_candidate(self, candidate: LandmarkBufferCandidate) -> None:
        candidate.detections.sort(key=self._evidence_sort_key)
        candidate.detections = candidate.detections[: self.max_detections_per_entity]
        positions = self._candidate_evidence_positions(candidate)
        candidate.position = {
            "x": sum(float(item["x"]) for item in positions) / len(positions),
            "y": sum(float(item["y"]) for item in positions) / len(positions),
            "z": sum(float(item["z"]) for item in positions) / len(positions),
        }
        candidate.first_step_index = min(int(item["step_index"]) for item in candidate.detections)
        candidate.last_step_index = max(int(item["step_index"]) for item in candidate.detections)
        if (
            candidate.status != LANDMARK_STATUS_CONFIRMED
            and candidate.observation_count() >= self.confirmation_min_observations
        ):
            candidate.status = LANDMARK_STATUS_NEEDS_VERIFICATION

    def _add_or_replace_evidence(
        self,
        *,
        candidate: LandmarkBufferCandidate,
        evidence: dict[str, object],
    ) -> None:
        obs_id = str(evidence.get("obs_id", "") or "")
        existing_index = None
        for index, item in enumerate(candidate.detections):
            if str(item.get("obs_id", "") or "") == obs_id:
                existing_index = index
                break
        if existing_index is None:
            candidate.detections.append(evidence)
        elif float(evidence["score"]) > float(candidate.detections[existing_index].get("score", 0.0)):
            candidate.detections[existing_index] = evidence

    def _consolidate_candidate(
        self,
        candidate: LandmarkBufferCandidate,
    ) -> LandmarkBufferCandidate:
        while True:
            merge_target = self._find_candidate_consolidation_target(candidate)
            if merge_target is None:
                return candidate
            survivor, absorbed = self._ordered_candidate_pair(candidate, merge_target)
            survivor.visited = bool(survivor.visited or absorbed.visited)
            survivor.attributes = {**dict(absorbed.attributes), **dict(survivor.attributes)}
            merge_entity_knowledge(survivor.knowledge, absorbed.knowledge)
            for evidence in absorbed.detections:
                self._add_or_replace_evidence(candidate=survivor, evidence=dict(evidence))
            self._refresh_candidate(survivor)
            self._candidates.pop(str(absorbed.landmark_id), None)
            candidate = survivor

    def _find_candidate_consolidation_target(
        self,
        candidate: LandmarkBufferCandidate,
    ) -> LandmarkBufferCandidate | None:
        best: LandmarkBufferCandidate | None = None
        best_distance = float("inf")
        class_name = self._normalize_class_name(candidate.class_name)
        merge_distance_m = self.merge_distance_for_class(class_name)
        for other in self._candidates.values():
            if other is candidate:
                continue
            if self._normalize_class_name(other.class_name) != class_name:
                continue
            distance = self._candidate_pair_min_position_distance_xy(candidate, other)
            if distance > merge_distance_m:
                continue
            if distance >= best_distance:
                continue
            best = other
            best_distance = distance
        return best

    @staticmethod
    def _landmark_id_index(landmark_id: str) -> int:
        try:
            return int(str(landmark_id).rsplit("_", 1)[-1])
        except ValueError:
            return 10**9

    def _ordered_candidate_pair(
        self,
        a: LandmarkBufferCandidate,
        b: LandmarkBufferCandidate,
    ) -> tuple[LandmarkBufferCandidate, LandmarkBufferCandidate]:
        ordered = sorted(
            [a, b],
            key=lambda item: (
                int(item.first_step_index),
                self._landmark_id_index(str(item.landmark_id)),
                str(item.landmark_id),
            ),
        )
        return ordered[0], ordered[1]

    def add_detection(
        self,
        *,
        detection: dict[str, object],
        source: str,
        step_index: int,
    ) -> LandmarkBufferCandidate:
        class_name = self._normalize_class_name(detection.get("class_name", "") or "")
        det_id = str(detection.get("det_id", "") or "")
        obs_id = str(detection.get("obs_id", "") or "")
        position = self._normalize_position(detection.get("position"))
        evidence = {
            "det_id": det_id,
            "obs_id": obs_id,
            "class_name": class_name,
            "detector_class_name": str(detection.get("detector_class_name", class_name)),
            "score": float(detection.get("score", 0.0)),
            "bbox": [float(value) for value in list(detection.get("bbox", []))],
            "source": str(source),
            "step_index": int(step_index),
            "position": dict(position),
            "corner_points": [dict(item) for item in list(detection.get("corner_points", []))],
            "center_point": (
                dict(detection["center_point"])
                if isinstance(detection.get("center_point"), dict)
                else None
            ),
            "plane_max_residual_m": float(detection.get("plane_max_residual_m", 0.0)),
            "geometry": str(detection.get("geometry", "")),
            "overlay_id": None if detection.get("overlay_id") is None else str(detection.get("overlay_id")),
        }
        match = self._find_match(class_name=class_name, position=position)
        if match is None:
            candidate = LandmarkBufferCandidate(
                landmark_id=self._next_landmark_id(),
                class_name=class_name,
                status=LANDMARK_STATUS_PENDING,
                visited=False,
                first_step_index=int(step_index),
                last_step_index=int(step_index),
                position=dict(position),
                detections=[evidence],
            )
            self._candidates[str(candidate.landmark_id)] = candidate
            self._refresh_candidate(candidate)
            return self._consolidate_candidate(candidate)

        candidate, _distance = match
        self._add_or_replace_evidence(candidate=candidate, evidence=evidence)
        self._refresh_candidate(candidate)
        return self._consolidate_candidate(candidate)

    def mark_visited(self, landmark_id: str) -> LandmarkBufferCandidate:
        candidate = self._candidates.get(str(landmark_id))
        if candidate is None:
            raise ValueError(f"unknown landmark_id: {landmark_id!r}")
        candidate.visited = True
        return candidate

    def get_candidate(self, landmark_id: str) -> LandmarkBufferCandidate:
        candidate = self._candidates.get(str(landmark_id))
        if candidate is None:
            raise ValueError(f"unknown landmark_id: {landmark_id!r}")
        return candidate

    def find_candidate(self, landmark_id: str) -> LandmarkBufferCandidate | None:
        return self._candidates.get(str(landmark_id))

    def candidates_needing_verification(self) -> list[LandmarkBufferCandidate]:
        candidates = [
            candidate
            for candidate in self._candidates.values()
            if str(candidate.status) == LANDMARK_STATUS_NEEDS_VERIFICATION
        ]
        candidates.sort(
            key=lambda item: (
                int(item.first_step_index),
                self._landmark_id_index(str(item.landmark_id)),
                str(item.landmark_id),
            )
        )
        return candidates

    def mark_verified(
        self,
        landmark_id: str,
        *,
        attributes: dict[str, object],
    ) -> LandmarkBufferCandidate:
        candidate = self.get_candidate(str(landmark_id))
        candidate.attributes.update(dict(attributes))
        candidate.status = LANDMARK_STATUS_CONFIRMED
        return candidate

    def update_attributes(
        self,
        landmark_id: str,
        attributes: dict[str, object],
    ) -> LandmarkBufferCandidate:
        candidate = self.get_candidate(str(landmark_id))
        candidate.attributes.update(dict(attributes))
        return candidate

    def reject_candidate(
        self,
        landmark_id: str,
        *,
        attributes: dict[str, object],
        rejected_mode: str,
        blacklist_reason: str = "focused_landmark_false_positive",
    ) -> dict[str, object]:
        candidate = self.get_candidate(str(landmark_id))
        candidate.attributes.update(dict(attributes))
        candidate.attributes["is_false_positive"] = True
        candidate.attributes["rejected_mode"] = str(rejected_mode)
        payload = candidate.to_dict()
        payload["blacklist_reason"] = str(blacklist_reason)
        self._blacklist.append(payload)
        self._candidates.pop(str(candidate.landmark_id), None)
        return payload

    def blacklist_match(
        self,
        *,
        class_name: str,
        position: dict[str, float],
    ) -> dict[str, object] | None:
        normalized_class_name = self._normalize_class_name(class_name)
        merge_distance_m = self.merge_distance_for_class(normalized_class_name)
        normalized_position = self._normalize_position(position)
        best: dict[str, object] | None = None
        best_distance = float("inf")
        for item in self._blacklist:
            if self._normalize_class_name(item.get("class_name", "")) != normalized_class_name:
                continue
            positions = self._blacklist_entry_positions(item)
            if positions == []:
                continue
            distance = min(self._distance_xy(normalized_position, existing) for existing in positions)
            if distance > merge_distance_m or distance >= best_distance:
                continue
            best = item
            best_distance = float(distance)
        if best is None:
            return None
        return {
            "landmark_id": str(best.get("landmark_id", "")),
            "class_name": str(best.get("class_name", "")),
            "distance_m": float(best_distance),
            "merge_distance_m": float(merge_distance_m),
            "attributes": dict(best.get("attributes", {})),
        }

    def _blacklist_entry_positions(self, entry: dict[str, object]) -> list[dict[str, float]]:
        detections = entry.get("detections", [])
        positions = [
            self._normalize_position(item["position"])
            for item in list(detections)
            if isinstance(item, dict) and isinstance(item.get("position"), dict)
        ]
        if positions == [] and isinstance(entry.get("position"), dict):
            positions.append(self._normalize_position(entry["position"]))
        return positions

    def to_context(self, step_index: int | None = None) -> dict[str, object]:
        del step_index
        candidates = [candidate.to_dict() for candidate in self._candidates.values()]
        candidates.sort(
            key=lambda item: (
                str(item["status"]) != "confirmed",
                -int(item["observation_count"]),
                str(item["landmark_id"]),
            )
        )
        return {
            "merge_distance_m": float(self.merge_distance_m),
            "merge_distance_by_class": dict(self.merge_distance_by_class),
            "confirmation_min_observations": int(self.confirmation_min_observations),
            "max_detections_per_entity": int(self.max_detections_per_entity),
            "pending_candidates": [
                dict(item)
                for item in candidates
                if str(item.get("status", "")) == LANDMARK_STATUS_PENDING
            ],
            "needs_verification_candidates": [
                dict(item)
                for item in candidates
                if str(item.get("status", "")) == LANDMARK_STATUS_NEEDS_VERIFICATION
            ],
            "confirmed_landmarks": [
                dict(item)
                for item in candidates
                if str(item.get("status", "")) == LANDMARK_STATUS_CONFIRMED
            ],
            "blacklisted_landmarks": [dict(item) for item in self._blacklist],
        }
