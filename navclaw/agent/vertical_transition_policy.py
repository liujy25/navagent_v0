from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from navclaw.agent.visual_action_context import allowed_directions_text
from navclaw.agent.visual_action_context import angle_for_direction
from navclaw.agent.visual_action_context import direction_for_angle
from navclaw.agent.visual_action_context import VISUAL_ACTION_ANGLES
from navclaw.agent.visual_action_context import VisualActionContext
from navclaw.agent.visual_policy_prompt_images import (
    image_content_for_movement_history_sheet,
    image_content_for_vertical_transition_panorama_views,
)

if TYPE_CHECKING:
    from navclaw.llm.client import LLMClient
    from navclaw.runtime.cache import RuntimeCache


@dataclass(frozen=True)
class VerticalTransitionStepDecision:
    thought: str
    transition_status: str
    waypoint_target: str
    selected_angle_deg: int | None

    def to_dict(self) -> dict[str, object]:
        return {
            "thought": str(self.thought),
            "transition_status": str(self.transition_status),
            "waypoint_target": str(self.waypoint_target),
            "selected_angle_deg": (
                None if self.selected_angle_deg is None else int(self.selected_angle_deg)
            ),
        }


def decide_vertical_transition_step(
    *,
    client: "LLMClient",
    cache: "RuntimeCache",
    instruction: str,
    visual_context: VisualActionContext,
    initial_observation: bool = False,
    movement_rgb_history_blocks: list[list[str]] | None = None,
    current_retry_feedback: dict[str, object] | None = None,
) -> VerticalTransitionStepDecision:
    allowed_angles = visual_context.available_angles
    system_prompt = """
You are guiding an embodied navigation agent up or down a staircase.
Decide whether the agent has reached the next floor. If not, choose the next visible and reachable place to continue.
Return JSON only.
""".strip()
    request_text = f"""
Requested movement:
{instruction}
""".strip()
    decision_text = f"""
Decision rules:
- Interpret upstairs and downstairs from the agent's current pose.
- `complete`: the agent has reached the first stable corridor or room floor beyond the staircase just traversed.
- A stair-turn landing or mid-stair platform that only connects parts of the same staircase is not complete, but may be selected as the next waypoint when needed to continue.
- Once a corridor or room floor is reached, another visible staircase belongs to a later navigation action.
- `continue`: another local move is needed to enter the staircase, continue along it, or reach the next-floor walking surface.
- For `continue`, choose the next waypoint using this priority:
  1. If the next-floor walking surface is visible and reachable, choose the nearest reachable point on the first stable floor immediately beyond the final stair tread, before any onward corridor or room travel.
  2. Otherwise, choose the farthest clearly reachable area that continues along the staircase in the requested direction.
  3. If the staircase has not yet been entered, choose a reachable area that approaches or enters its visible entrance.
- Identify the view containing the selected area and describe one matching reachable waypoint target. Select its direction from {allowed_directions_text(allowed_angles)}.
- `fail`: no reachable local movement in the current panorama can enter or continue in the requested direction.
- Briefly state the status judgment in thought. For `continue`, also state why the selected view and target best follow the waypoint priority.
- For `complete` or `fail`, use an empty waypoint_target and a null selected_direction.
""".strip()
    output_text = """
Return JSON only with exactly these fields in this order:
- thought: string
- transition_status: one of "complete", "continue", or "fail"
- waypoint_target: string
- selected_direction: one of "front", "back", "left", or "right"; null for complete or fail
""".strip()
    content = [
        {"type": "text", "text": request_text},
    ]
    if initial_observation:
        content.append(
            {
                "type": "text",
                "text": (
                    "Current state:\n"
                    "This is the initial observation before the agent starts the requested movement."
                ),
            }
        )
    _append_vertical_transition_rgb_history(
        content=content,
        cache=cache,
        blocks=[] if movement_rgb_history_blocks is None else movement_rgb_history_blocks,
    )
    content.extend(
        image_content_for_vertical_transition_panorama_views(
            cache=cache,
            views=visual_context.views,
        )
    )
    if isinstance(current_retry_feedback, dict) and current_retry_feedback != {}:
        feedback_text = _vertical_transition_retry_feedback_text(current_retry_feedback)
        if feedback_text != "":
            content.append({"type": "text", "text": feedback_text})
    content.extend(
        [
            {"type": "text", "text": decision_text},
            {"type": "text", "text": output_text},
        ]
    )
    parsed = client.decide_vertical_transition_step(system_prompt, content)
    return _normalize_vertical_transition_step(
        parsed,
        allowed_angles=allowed_angles,
    )


def _normalize_vertical_transition_step(
    payload: dict[str, object],
    *,
    allowed_angles: list[int] | None = None,
) -> VerticalTransitionStepDecision:
    valid_angles = VISUAL_ACTION_ANGLES if allowed_angles is None else [int(angle) for angle in allowed_angles]
    thought = str(payload.get("thought", "")).strip()
    waypoint_target = str(payload.get("waypoint_target", "")).strip()
    raw_transition_status = str(payload.get("transition_status", "")).strip().lower()
    transition_status = raw_transition_status
    if transition_status not in {"continue", "complete", "fail"}:
        raise ValueError(
            f"vertical transition step planner returned unsupported transition_status: {payload!r}"
        )
    if thought == "":
        raise ValueError(
            f"vertical transition step planner requires thought: {payload!r}"
        )
    selected_angle: int | None = None
    if transition_status in {"complete", "fail"}:
        waypoint_target = ""
    else:
        raw_direction = payload.get("selected_direction")
        raw_angle = payload.get("selected_angle_deg")
        if raw_direction is None and raw_angle is None:
            raise ValueError(
                f"vertical transition step planner requires selected_direction: {payload!r}"
            )
        if raw_direction is not None:
            try:
                selected_angle = angle_for_direction(raw_direction)
            except ValueError as exc:
                raise ValueError(
                    "vertical transition step planner selected unsupported "
                    f"direction: {payload!r}"
                ) from exc
        else:
            try:
                selected_angle = int(raw_angle)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "vertical transition step planner selected unsupported "
                    f"direction: {payload!r}"
                ) from exc
        if selected_angle not in valid_angles:
            raise ValueError(
                f"vertical transition step planner selected unsupported direction: {payload!r}"
            )
        if waypoint_target == "":
            raise ValueError(
                f"vertical transition step planner requires waypoint_target: {payload!r}"
            )
    return VerticalTransitionStepDecision(
        thought=thought,
        transition_status=transition_status,
        waypoint_target=waypoint_target,
        selected_angle_deg=selected_angle,
    )


def _append_vertical_transition_rgb_history(
    *,
    content: list[dict[str, object]],
    cache: "RuntimeCache",
    blocks: list[list[str]],
) -> None:
    rendered_blocks: list[tuple[int, dict[str, object]]] = []
    for movement_index, obs_ids in enumerate(blocks):
        image_content = image_content_for_movement_history_sheet(
            cache=cache,
            obs_ids=[str(obs_id) for obs_id in obs_ids],
        )
        if image_content is not None:
            rendered_blocks.append((movement_index, image_content))
    if rendered_blocks == []:
        return
    content.append(
        {
            "type": "text",
            "text": (
                "Movement history:\n"
                "- Movements are shown in execution order.\n"
                "- Within each movement, frames are ordered by time."
            ),
        }
    )
    for movement_index, image_content in rendered_blocks:
        content.extend(
            [
                {"type": "text", "text": f"Movement {movement_index + 1}:"},
                image_content,
            ]
        )


def _vertical_transition_retry_feedback_text(feedback: dict[str, object]) -> str:
    if str(feedback.get("type", "")).strip() == "completion_rejected":
        return (
            "System feedback:\n"
            "The agent has not yet been confirmed to have reached the next floor. "
            "Choose the next visible and reachable place to continue."
        )
    rejected_options = [
        item
        for item in list(feedback.get("rejected_options", []))
        if isinstance(item, dict)
        and item.get("angle_deg") is not None
        and str(item.get("waypoint_target", "")).strip() != ""
    ]
    if rejected_options == []:
        return ""
    lines = ["Rejected movement options for the current observation:"]
    for index, option in enumerate(rejected_options, start=1):
        lines.extend(
            [
                "",
                f"{index}.",
                f"- rejected view: {direction_for_angle(int(option['angle_deg']))}",
                "- rejected target in that view: "
                + str(option["waypoint_target"]).strip(),
            ]
        )
    return "\n".join(lines)
