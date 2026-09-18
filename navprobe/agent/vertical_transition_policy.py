from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from navprobe.agent.visual_action_context import allowed_directions_text
from navprobe.agent.visual_action_context import angle_for_direction
from navprobe.agent.visual_action_context import direction_for_angle
from navprobe.agent.visual_action_context import VISUAL_ACTION_ANGLES
from navprobe.agent.visual_action_context import VisualActionContext
from navprobe.agent.visual_policy_prompt_images import image_content_for_array
from navprobe.agent.visual_policy_prompt_images import (
    image_content_for_movement_history_sheet,
    image_content_for_vertical_transition_panorama_views,
)

if TYPE_CHECKING:
    from navprobe.llm.client import LLMClient
    from navprobe.runtime.cache import RuntimeCache


@dataclass(frozen=True)
class VerticalTransitionStepDecision:
    transition_status: str
    waypoint_target: str
    selected_angle_deg: int | None
    reason: str
    destination_floor_reached: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "transition_status": str(self.transition_status),
            "waypoint_target": str(self.waypoint_target),
            "selected_angle_deg": (
                None if self.selected_angle_deg is None else int(self.selected_angle_deg)
            ),
            "reason": str(self.reason),
            "destination_floor_reached": self.destination_floor_reached,
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
    planning_reference: bool = False,
) -> VerticalTransitionStepDecision:
    allowed_angles = visual_context.available_angles
    request_text = f"Requested vertical objective:\n{instruction}"
    system_prompt = """
You are the VerticalMove local skill controller in NavProbe.
Use the requested objective, current observation, and executed movement history to assess the requested endpoint and identify a reachable next region.
Return valid JSON matching the output contract.
""".strip()
    decision_text = f"""
Decision semantics:
- `complete`: the robot physically satisfies the endpoint stated in the requested objective.
- `continue`: another local move is needed to reach that endpoint.
- `fail`: the observation provides no reachable continuation of this objective.
- Preserve the exact endpoint, including a specified turn landing or partial ascent/descent. A partial stair target is complete at its stated position; reaching a new floor is only required when the objective requests it.
- Use fresh observations and executed movement history to assess completion.
- For `continue`, identify a visible reachable stair tread, landing, or adjoining walking surface that advances the objective without passing its endpoint. Select its view from {allowed_directions_text(allowed_angles)}.
- `destination_floor_reached` is true only when the current robot is on a stable floor surface beyond the staircase; a stair tread or intermediate platform is false.
""".strip()
    output_text = """
Output contract:
{"transition_status":"complete|continue|fail", "waypoint_target":"<reachable local region for continue, otherwise empty>", "selected_direction":"front|back|left|right|null", "destination_floor_reached":false, "reason":"<observed endpoint or next-region basis>"}
""".strip()
    content = [
        {"type": "text", "text": request_text},
    ]
    if planning_reference:
        content.append({"type": "text", "text": (
            "Planning reference: these stored views describe the selected visited node. "
            "The robot has not moved there yet. Select a reachable next region from this reference; "
            "endpoint completion requires a fresh observation after physical movement."
        )})
    elif initial_observation:
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
    reason = str(payload.get("reason", "")).strip()
    waypoint_target = str(payload.get("waypoint_target", "")).strip()
    raw_transition_status = str(payload.get("transition_status", "")).strip().lower()
    transition_status = raw_transition_status
    if transition_status not in {"continue", "complete", "fail"}:
        raise ValueError(
            f"vertical transition step planner returned unsupported transition_status: {payload!r}"
        )
    if reason == "":
        raise ValueError(
            f"vertical transition step planner requires reason: {payload!r}"
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
    destination_floor_reached = payload.get("destination_floor_reached")
    if type(destination_floor_reached) is not bool:
        raise ValueError("NavProbe vertical status requires destination_floor_reached boolean")
    return VerticalTransitionStepDecision(
        transition_status=transition_status,
        waypoint_target=waypoint_target,
        selected_angle_deg=selected_angle,
        reason=reason,
        destination_floor_reached=destination_floor_reached,
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
    if feedback.get("reason") == "virtual_reference_is_not_physical_completion":
        return "System feedback: the stored planning reference is not the robot's current pose. Ground the requested movement before judging physical completion."
    return ""


def select_vertical_fss_waypoint(*, client, prepared, objective, local_target, task_context):
    """Choose the parameterized waypoint from shared RGB/BEV FSS labels."""
    content = [{"type": "text", "text": (
        f"Selected vertical objective:\n{objective}\n\n"
        f"Next local target:\n{local_target}\n\n{task_context}\n\n"
        "Choose a visible connected FSS label that advances the requested target without passing its endpoint. "
        "The same labels identify the same world points in RGB and BEV. "
        "Return {\"candidate_label\":<integer or null if none matches>,\"reason\":\"<evidence-based selection>\"}."
    )}]
    for view_set in prepared.view_candidate_sets:
        content.append({"type": "text", "text": f"{direction_for_angle(view_set.view.angle_deg)} view; labels: {[c.label for c in view_set.candidates]}"})
        content.append(image_content_for_array(view_set.rgb_overlay))
    if prepared.bev_overlay is not None:
        content.append({"type": "text", "text": "Local BEV: blue is detected stair support temporarily traversable; orange is the robot; numbered red dots are candidate endpoints."})
        content.append(image_content_for_array(prepared.bev_overlay))
    response = client._create_visual_json_completion(
        call_name="vertical_fss_waypoint_planner",
        system_prompt="You are NavProbe's VerticalMove waypoint selector. Select a labeled reachable waypoint for the supplied objective.",
        user_prompt=content, max_new_tokens=2048,
        token_field="max_completion_tokens", retry_count=2,
    )
    selected_label = response.get("candidate_label")
    if not str(response.get("reason", "")).strip():
        raise ValueError("vertical FSS selection requires reason")
    if selected_label is None:
        raise ValueError("vertical FSS has no candidate matching the requested target")
    if type(selected_label) is not int or selected_label not in {c.label for c in prepared.candidates}:
        raise ValueError("vertical FSS selected an unavailable candidate label")
    return selected_label, str(response["reason"]).strip()
