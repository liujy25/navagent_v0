from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from navclaw.llm.image_preprocessing import encode_rgb_to_jpeg_data_url

if TYPE_CHECKING:
    from navclaw.perception.landmark_buffer import LandmarkBufferCandidate
    from navclaw.runtime.cache import RuntimeCache


@dataclass(frozen=True)
class LandmarkVerificationResult:
    accepted: bool
    confidence: float
    reason: str
    raw_text: str
    overlay_id: str
    question: str

    def to_dict(self) -> dict[str, object]:
        return {
            "accepted": bool(self.accepted),
            "confidence": float(self.confidence),
            "reason": str(self.reason),
            "raw_text": str(self.raw_text),
            "overlay_id": str(self.overlay_id),
            "question": str(self.question),
        }


def verify_landmark_candidate(
    *,
    candidate: "LandmarkBufferCandidate",
    cache: "RuntimeCache",
    llm_client,
    landmark_class_map: dict[str, dict[str, object]],
) -> LandmarkVerificationResult:
    evidence = _highest_score_evidence(candidate)
    overlay_id = str(evidence.get("overlay_id", "") or "").strip()
    if overlay_id == "":
        raise ValueError(f"landmark candidate {candidate.landmark_id!r} has no overlay_id")
    if overlay_id not in cache.images:
        raise ValueError(f"landmark verification overlay_id not found in cache: {overlay_id!r}")

    question = _verification_question(
        class_name=str(candidate.class_name),
        landmark_class_map=landmark_class_map,
    )
    image = np.asarray(cache.get_image(overlay_id).image, dtype=np.uint8)
    user_prompt = [
        {
            "type": "text",
            "text": (
                f"{question}\n"
                "Return exactly one line: true;<confidence_score>;<brief_reason> "
                "or false;<confidence_score>;<brief_reason>.\n"
                "<confidence_score> must be a number from 0.0 to 1.0."
            ),
        },
        {
            "type": "image_url",
            "image_url": {"url": encode_rgb_to_jpeg_data_url(image)},
        },
    ]
    raw_text = llm_client.verify_landmark_candidate(
        system_prompt="Answer exactly in the requested format.",
        user_prompt=user_prompt,
    )
    accepted, confidence, reason = parse_landmark_verification_output(raw_text)
    return LandmarkVerificationResult(
        accepted=accepted,
        confidence=confidence,
        reason=reason,
        raw_text=str(raw_text),
        overlay_id=overlay_id,
        question=question,
    )


def parse_landmark_verification_output(raw_text: str) -> tuple[bool, float, str]:
    parts = [part.strip() for part in str(raw_text).strip().split(";", 2)]
    if len(parts) != 3:
        raise ValueError(
            "landmark verification output must be "
            f"'true;<confidence>;<reason>' or 'false;<confidence>;<reason>', got {raw_text!r}"
        )
    decision = parts[0].lower()
    if decision not in {"true", "false"}:
        raise ValueError(f"landmark verification decision must be true or false, got {raw_text!r}")
    reason = parts[2].strip()
    if reason == "":
        raise ValueError(f"landmark verification reason must be non-empty, got {raw_text!r}")
    return decision == "true", float(parts[1]), reason


def verification_attributes(result: LandmarkVerificationResult) -> dict[str, object]:
    return {
        "semantic_verification": {
            "accepted": bool(result.accepted),
            "confidence": float(result.confidence),
            "reason": str(result.reason),
            "overlay_id": str(result.overlay_id),
            "question": str(result.question),
            "raw_text": str(result.raw_text),
        }
    }


def verification_failure_attributes(
    *,
    message: str,
    overlay_id: str | None = None,
) -> dict[str, object]:
    return {
        "semantic_verification": {
            "accepted": None,
            "confidence": None,
            "reason": str(message),
            "overlay_id": None if overlay_id is None else str(overlay_id),
            "raw_text": "",
        }
    }


def _highest_score_evidence(candidate: "LandmarkBufferCandidate") -> dict[str, object]:
    if candidate.detections == []:
        raise ValueError(f"landmark candidate {candidate.landmark_id!r} has no detections")
    return max(
        (dict(item) for item in candidate.detections),
        key=lambda item: (
            float(item.get("score", 0.0)),
            int(item.get("step_index", 0)),
            str(item.get("det_id", "")),
        ),
    )


def _verification_question(
    *,
    class_name: str,
    landmark_class_map: dict[str, dict[str, object]],
) -> str:
    configured_question = _configured_verification_question(
        class_name=class_name,
        landmark_class_map=landmark_class_map,
    )
    if configured_question != "":
        return configured_question
    aliases = _aliases_for_class(
        class_name=class_name,
        landmark_class_map=landmark_class_map,
    )
    return f"Is the target in the red bounding box a {_join_aliases(aliases)}?"


def _configured_verification_question(
    *,
    class_name: str,
    landmark_class_map: dict[str, dict[str, object]],
) -> str:
    normalized_class = _normalize_class_name(class_name)
    for config in landmark_class_map.values():
        if _normalize_class_name(config.get("class_name", "")) != normalized_class:
            continue
        question = str(config.get("verification_question", "")).strip()
        if question != "":
            return question
    return ""


def _aliases_for_class(
    *,
    class_name: str,
    landmark_class_map: dict[str, dict[str, object]],
) -> list[str]:
    normalized_class = _normalize_class_name(class_name)
    aliases: list[str] = []
    for query_class, config in landmark_class_map.items():
        if _normalize_class_name(config.get("class_name", "")) != normalized_class:
            continue
        alias = _normalize_class_name(query_class)
        if alias != "" and alias not in aliases:
            aliases.append(alias)
    if aliases == []:
        aliases.append(normalized_class)
    return aliases


def _join_aliases(aliases: list[str]) -> str:
    if len(aliases) == 1:
        return str(aliases[0])
    if len(aliases) == 2:
        return f"{aliases[0]} or {aliases[1]}"
    return f"{', '.join(aliases[:-1])}, or {aliases[-1]}"


def _normalize_class_name(value: object) -> str:
    return str(value).strip().replace("_", " ")
