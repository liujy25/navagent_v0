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
You are the Vertical Transition Step Planner in NavClaw.
For the fixed up- or down-stair action, decide whether the transition is complete; otherwise identify one visible reachable local region for the next movement.
Do not change the requested vertical direction or choose an image point.
Return only valid JSON matching the provided output contract.
""".strip()
    request_text = f"""
Fixed vertical-transition action:
{instruction}
""".strip()
    decision_text = f"""
Decision semantics:
- `complete`: the agent has reached a stable walking surface on the destination floor beyond the staircase being traversed.
- `continue`: one more local move is required to enter the staircase, continue on it, or reach the destination-floor walking surface.
- `fail`: the current panorama provides no reachable local movement that can enter or continue the requested transition.
- A turn landing or mid-stair platform that only connects parts of the same staircase is not completion.
- A different staircase visible after reaching the destination-floor corridor or room belongs to a later high-level action.

Evidence rules:
- Judge completion from the current fresh panorama and supplied movement history, not from a proposed point.
- Interpret `up` and `down` as elevation change from the agent's current physical pose.
- Treat rejected movement options as invalid for the current panorama unless new evidence changes their validity.

Waypoint-target priority for `continue`:
1. If the destination-floor walking surface is visible and reachable, choose the nearest safe region immediately beyond the final stair tread.
2. Otherwise choose the farthest clearly reachable region that continues along the current staircase in the requested vertical direction.
3. If the staircase has not been entered, choose a reachable region at or just inside its visible entrance.

Output rules:
- For `continue`, describe one concrete reachable region in `waypoint_target` and select its view from {allowed_directions_text(allowed_angles)}.
- For `complete` or `fail`, use an empty `waypoint_target` and JSON `null` for `selected_direction`.
- `thought` is a concise status basis and, for `continue`, states why the target follows the priority above; it is not a long reasoning trace.
""".strip()
    output_text = """
Output contract:
{
  "thought": "<concise status and target basis>",
  "transition_status": "complete|continue|fail",
  "waypoint_target": "<reachable local region, or empty string>",
  "selected_direction": "front|back|left|right|null"
}

For `complete` or `fail`, `selected_direction` is null:
{"thought":"<concise status basis>","transition_status":"complete","waypoint_target":"","selected_direction":null}
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
    content.extend(
        [
            {"type": "text", "text": decision_text},
            {"type": "text", "text": output_text},
        ]
    )
    if isinstance(current_retry_feedback, dict) and current_retry_feedback != {}:
        feedback_text = _vertical_transition_retry_feedback_text(current_retry_feedback)
        if feedback_text != "":
            content.append({"type": "text", "text": feedback_text})
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
        if waypoint_target != "":
            raise ValueError(
                "complete or failed vertical transition step requires an empty "
                f"waypoint_target: {payload!r}"
            )
        if (
            payload.get("selected_direction") is not None
            or payload.get("selected_angle_deg") is not None
        ):
            raise ValueError(
                "complete or failed vertical transition step requires a null "
                f"selected_direction: {payload!r}"
            )
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
                "- Movement blocks are shown in execution order.\n"
                "- Frames are ordered chronologically from left to right and then top to bottom."
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
    failure_detail = str(feedback.get("feedback", "")).strip()
    if failure_detail != "":
        lines.extend(["", "Grounding failure:", failure_detail])
    return "\n".join(lines)
