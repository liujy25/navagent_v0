from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from navclaw.agent.visual_action_context import angle_for_direction
from navclaw.agent.visual_action_context import direction_for_angle
from navclaw.agent.visual_action_context import ordered_panorama_angles
from navclaw.agent.visual_action_context import VISUAL_ACTION_ANGLES
from navclaw.agent.visual_policy_prompt_images import (
    image_content_for_current_panorama_views,
)

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
                direction_for_angle(angle): str(self.direction_summaries.get(angle, ""))
                for angle in VISUAL_ACTION_ANGLES
                if angle in self.direction_summaries
            },
        }


NODE_SUMMARY_ANGLES = list(VISUAL_ACTION_ANGLES)


def summarize_current_node(
    *,
    client: "LLMClient",
    cache: "RuntimeCache",
    visual_context: "VisualActionContext",
) -> NodeSummaryDecision:
    allowed_angles = ordered_panorama_angles(
        [int(view.angle_deg) for view in visual_context.views]
    )
    system_prompt = """
You are the Current Node Summary module in NavClaw.
Summarize the place represented by the supplied panorama and the distinctive content visible in each local view.
Do not infer task progress, historical events, or a navigation action.
Return only valid JSON matching the provided output contract.
""".strip()
    text = f"""
Decision objective:
Create compact textual entity knowledge for the current place node from the supplied panorama.

Field semantics:
- `node_summary` describes the place as a whole across all supplied views.
- `direction_summaries` describes distinctive visible content in each local panorama direction; these descriptions are observations, not route recommendations.

Evidence rules:
- Base every claim only on the supplied panorama.
- Distinguish visibility from room membership. An object visible through a doorway or opening may belong to an adjacent room; determine the current room from spatial boundaries, not from object visibility alone.
- Use a specific place type such as bedroom or kitchen only when diagnostic visual evidence and spatial boundaries establish it as the current place; otherwise use a generic description such as room, corridor, doorway area, junction, entrance, or unclear indoor place.
- Do not infer the identity or contents of an unseen area behind a doorway or around a corner.
- Mention an object or landmark only when its appearance is visually clear.
- Describe a clearly open passage or visible walking surface without claiming guaranteed reachability from appearance alone.
- Exclude detection ids, observation ids, bounding boxes, coordinates, candidate labels, overlay labels, and other system internals.
- Keep `node_summary` to one concise sentence.
- `direction_summaries` must contain exactly these keys: {_direction_summary_keys_text(allowed_angles)}.
- Use an empty string when a direction has no distinctive navigation-relevant cue or only repeats the place-level summary.

Output contract:
{{
  "node_summary": "<one concise place-level sentence>",
  "direction_summaries": {_direction_summary_schema_text(allowed_angles)}
}}
""".strip()
    content = [{"type": "text", "text": text}]
    content.extend(
        image_content_for_current_panorama_views(
            cache=cache,
            views=visual_context.views,
            include_visited_nodes=False,
            label_prefix="Current node panorama view",
        )
    )
    parsed = client.summarize_node(system_prompt, content)
    return normalize_node_summary(parsed, allowed_angles=allowed_angles)


def normalize_node_summary(
    payload: dict[str, object],
    *,
    allowed_angles: list[int] | None = None,
) -> NodeSummaryDecision:
    valid_angles = (
        NODE_SUMMARY_ANGLES
        if allowed_angles is None
        else ordered_panorama_angles([int(angle) for angle in allowed_angles])
    )
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
                item.get(
                    "direction",
                    item.get("angle_deg", item.get("angle", "")),
                ),
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
    try:
        direction_angle = angle_for_direction(text)
    except ValueError:
        direction_angle = None
    if direction_angle is not None:
        return direction_angle if direction_angle in allowed_angles else None
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
    for angle in ordered_panorama_angles(allowed_angles):
        description = str(direction_summaries.get(int(angle), "")).strip()
        if description != "":
            items.append(f"{direction_for_angle(angle)}: {description}")
    return "; ".join(items)


def _normalize_visible_areas(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        items: list[str] = []
        for item in value:
            if isinstance(item, dict):
                angle = str(
                    item.get("direction", item.get("angle_deg", item.get("angle", "")))
                ).strip()
                description = str(item.get("description", "")).strip()
                if description == "":
                    continue
                if angle != "":
                    normalized_angle = _normalize_angle_key(
                        angle,
                        allowed_angles=NODE_SUMMARY_ANGLES,
                    )
                    direction = (
                        direction_for_angle(normalized_angle)
                        if normalized_angle is not None
                        else angle
                    )
                    items.append(f"{direction}: {description}")
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
            normalized_angle = _normalize_angle_key(
                angle_text,
                allowed_angles=NODE_SUMMARY_ANGLES,
            )
            direction = (
                direction_for_angle(normalized_angle)
                if normalized_angle is not None
                else angle_text
            )
            items.append(f"{direction}: {text}" if direction != "" else text)
        return "; ".join(items)
    return str(value).strip()


def _direction_summary_keys_text(angles: list[int]) -> str:
    return ", ".join(
        f'"{direction_for_angle(angle)}"'
        for angle in ordered_panorama_angles(angles)
    )


def _direction_summary_schema_text(angles: list[int]) -> str:
    entries = [
        f'"{direction_for_angle(angle)}": "<visible content toward {direction_for_angle(angle)} or empty string>"'
        for angle in ordered_panorama_angles(angles)
    ]
    return "{ " + ", ".join(entries) + " }"
