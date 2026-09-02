from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from navclaw.agent.visual_action_context import angle_convention_text
from navclaw.agent.visual_action_context import ordered_visual_views_left_to_right
from navclaw.agent.visual_action_context import panorama_angle_order_text
from navclaw.agent.visual_policy_prompt_images import _compose_labeled_image_strip
from navclaw.agent.visual_policy_prompt_images import image_content_for_array

if TYPE_CHECKING:
    from navclaw.agent.visual_action_context import VisualActionContext
    from navclaw.llm.client import LLMClient
    from navclaw.runtime.cache import RuntimeCache


@dataclass(frozen=True)
class NodeSummaryDecision:
    node_summary: str
    visible_areas: str
    direction_summaries: dict[int, str]

    def to_dict(self) -> dict[str, object]:
        return {
            "node_summary": str(self.node_summary),
            "visible_areas": str(self.visible_areas),
            "direction_summaries": {
                str(angle): str(self.direction_summaries.get(angle, ""))
                for angle in sorted(int(item) for item in self.direction_summaries)
            },
        }


NODE_SUMMARY_ANGLES = [0, 60, 120, 180, 240, 300]


def summarize_current_node(
    *,
    client: "LLMClient",
    cache: "RuntimeCache",
    visual_context: "VisualActionContext",
) -> NodeSummaryDecision:
    allowed_angles = [int(view.angle_deg) for view in visual_context.views]
    system_prompt = """
You are a node summary worker for an embodied navigation agent.
Your job is to summarize the current place node from visual observations.
""".strip()
    text = f"""
Write a compact semantic summary of the current node itself.

Rules:
- Describe the visually supported place type, such as a room, corridor, doorway, junction, entrance, or unclear indoor place.
- Use a more specific room identity, such as bedroom or kitchen, only when diagnostic visual evidence is clear.
- In direction_summaries, describe visible semantic areas, rooms, objects, or landmarks for each panorama angle.
- Mention an object only when its visual appearance is clear; use a generic structural description when the content is ambiguous.
- Base every scene and object claim only on the attached panorama.
- Claim a direction is navigable only when it is visually obvious; otherwise describe visibility rather than policy.
- Exclude raw detection ids, observation ids, bounding boxes, coordinates, and system internals.
- Keep both fields short and concrete.
- direction_summaries must contain exactly these keys: {_direction_summary_keys_text(allowed_angles)}.
- Use an empty string for an angle when there is no distinctive navigation-relevant cue or it merely repeats the overall node summary.

Return JSON only:
{{
  "node_summary": "<one concise sentence describing this node/place itself>",
  "direction_summaries": {_direction_summary_schema_text(allowed_angles)}
}}
""".strip()
    content = [
        {"type": "text", "text": text},
        {"type": "text", "text": _node_summary_panorama_strip_prompt_text(allowed_angles)},
        _image_content_for_panorama_strip(cache=cache, visual_context=visual_context),
    ]
    parsed = client.summarize_node(system_prompt, content)
    return normalize_node_summary(parsed, allowed_angles=allowed_angles)


def normalize_node_summary(
    payload: dict[str, object],
    *,
    allowed_angles: list[int] | None = None,
) -> NodeSummaryDecision:
    valid_angles = NODE_SUMMARY_ANGLES if allowed_angles is None else [int(angle) for angle in allowed_angles]
    node_summary = str(payload.get("node_summary", payload.get("summary", ""))).strip()
    direction_summaries = _normalize_direction_summaries(
        payload.get("direction_summaries", {}),
        allowed_angles=valid_angles,
    )
    if not any(direction_summaries.values()):
        direction_summaries = _normalize_direction_summaries(
            payload.get("visible_areas", ""),
            allowed_angles=valid_angles,
        )
    visible_areas = _visible_areas_from_direction_summaries(
        direction_summaries,
        allowed_angles=valid_angles,
    )
    if visible_areas == "":
        visible_areas = _normalize_visible_areas(payload.get("visible_areas", ""))
    if node_summary == "":
        raise ValueError(f"node summary worker returned empty node_summary: {payload!r}")
    return NodeSummaryDecision(
        node_summary=node_summary,
        visible_areas=visible_areas,
        direction_summaries=direction_summaries,
    )


def _normalize_direction_summaries(value: object, *, allowed_angles: list[int]) -> dict[int, str]:
    summaries = {int(angle): "" for angle in allowed_angles}
    if isinstance(value, dict):
        for angle, description in value.items():
            angle_value = _normalize_angle_key(angle, allowed_angles=allowed_angles)
            if angle_value is None:
                continue
            summaries[angle_value] = str(description).strip()
        return summaries
    if isinstance(value, list):
        for item in value:
            if not isinstance(item, dict):
                continue
            angle_value = _normalize_angle_key(
                item.get("angle_deg", item.get("angle", "")),
                allowed_angles=allowed_angles,
            )
            if angle_value is None:
                continue
            summaries[angle_value] = str(item.get("description", "")).strip()
        return summaries
    if isinstance(value, str):
        for item in value.split(";"):
            label, sep, description = item.partition(":")
            if sep == "":
                continue
            angle_value = _normalize_angle_key(label, allowed_angles=allowed_angles)
            if angle_value is None:
                continue
            summaries[angle_value] = description.strip()
    return summaries


def _normalize_angle_key(value: object, *, allowed_angles: list[int]) -> int | None:
    text = str(value).strip().lower()
    if text.startswith("angle_"):
        text = text[len("angle_"):]
    try:
        angle = int(text)
    except ValueError:
        return None
    if angle not in allowed_angles:
        return None
    return int(angle)


def _visible_areas_from_direction_summaries(
    direction_summaries: dict[int, str],
    *,
    allowed_angles: list[int],
) -> str:
    items = []
    for angle in allowed_angles:
        description = str(direction_summaries.get(int(angle), "")).strip()
        if description != "":
            items.append(f"angle_{int(angle)}: {description}")
    return "; ".join(items)


def _normalize_visible_areas(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        items: list[str] = []
        for item in value:
            if isinstance(item, dict):
                angle = str(item.get("angle_deg", item.get("angle", ""))).strip()
                description = str(item.get("description", "")).strip()
                if description == "":
                    continue
                if angle != "":
                    if not angle.startswith("angle_"):
                        angle = f"angle_{angle}"
                    items.append(f"{angle}: {description}")
                else:
                    items.append(description)
            else:
                text = str(item).strip()
                if text != "":
                    items.append(text)
        return "; ".join(items)
    if isinstance(value, dict):
        items = []
        for angle, description in value.items():
            text = str(description).strip()
            if text == "":
                continue
            angle_text = str(angle).strip()
            if angle_text != "" and not angle_text.startswith("angle_"):
                angle_text = f"angle_{angle_text}"
            items.append(f"{angle_text}: {text}" if angle_text != "" else text)
        return "; ".join(items)
    return str(value).strip()


def _direction_summary_keys_text(angles: list[int]) -> str:
    return ", ".join(f'"{int(angle)}"' for angle in angles)


def _direction_summary_schema_text(angles: list[int]) -> str:
    entries = [
        f'"{int(angle)}": "<visible content at angle_{int(angle)} or empty string>"'
        for angle in angles
    ]
    return "{ " + ", ".join(entries) + " }"


def _node_summary_panorama_strip_prompt_text(angles: list[int]) -> str:
    return f"""
Current panorama strip:
- Views are ordered left-to-right as {panorama_angle_order_text(angles)}.
- {angle_convention_text(angles)}
- Angle labels are view labels, not scene objects.
""".strip()


def _image_content_for_panorama_strip(
    *,
    cache: "RuntimeCache",
    visual_context: "VisualActionContext",
) -> dict[str, object]:
    ordered_views = ordered_visual_views_left_to_right(visual_context.views)
    images = [
        np.asarray(cache.get_observation(str(view.obs_id)).observation.rgb)
        for view in ordered_views
    ]
    labels = [f"angle_{int(view.angle_deg)}" for view in ordered_views]
    strip = _compose_labeled_image_strip(images=images, labels=labels)
    return image_content_for_array(strip)
