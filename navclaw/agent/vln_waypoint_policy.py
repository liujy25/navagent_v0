from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import json
from typing import TYPE_CHECKING

import numpy as np

from navclaw.agent.visual_action_context import angle_convention_text
from navclaw.agent.visual_action_context import VisualActionContext, VisualViewContext
from navclaw.agent.visual_grounding import VisualWaypoint
from navclaw.agent.visual_policy_decisions import LocalMovePlanDecision
from navclaw.agent.visual_policy_decisions import VisualActionPointDecision
from navclaw.agent.visual_policy_decisions import VisualWaypointVerificationDecision
from navclaw.agent.visual_policy_prompt_images import image_content_for_array
from navclaw.agent.vln_landmark_context import draw_vln_landmarks_on_view_image
from navclaw.agent.vln_landmark_context import VlnLandmarkEvidence
from navclaw.agent.vln_waypoint_sampling import draw_vln_sampled_waypoint_bev_overlay
from navclaw.agent.vln_waypoint_sampling import draw_vln_sampled_waypoint_rgb_overlay
from navclaw.agent.vln_waypoint_sampling import generate_vln_unified_sampled_waypoint_candidates
from navclaw.agent.vln_waypoint_sampling import visual_waypoint_from_sampled_candidate
from navclaw.agent.vln_waypoint_sampling import VlnSampledWaypointCandidate
from navclaw.agent.vln_waypoint_sampling import VLN_WAYPOINT_MAX_DISTANCE_M
from navclaw.agent.vln_waypoint_sampling import VLN_WAYPOINT_NODE_DEDUP_RADIUS_M
from navclaw.agent.vln_waypoint_sampling import VLN_WAYPOINT_SAMPLE_SPACING_M

if TYPE_CHECKING:
    from navclaw.llm.client import LLMClient
    from navclaw.mapping.exploration.manager import ExplorationManager
    from navclaw.mapping.exploration.bev_map import GlobalBEVMap
    from navclaw.runtime.cache import RuntimeCache
    from navclaw.types import LocalmapFrontierRecord
    from navclaw.visualization.action_mode_overlays import BevOverlayTransform


VLN_WAYPOINT_REPLAN_MAX_ATTEMPTS = 3
VLN_WAYPOINT_CANDIDATE_SELECTOR_SYSTEM_PROMPT = """
You are the VLN Waypoint Planner of an embodied navigation agent.
You select one labeled local waypoint candidate by following the updated task progress memory.
Use the current active progress item as the immediate waypoint objective.
""".strip()

VLN_WAYPOINT_INHERITED_CONTEXT_SYSTEM_PROMPT = """
You are the VLN Waypoint Planner of an embodied navigation agent.
Continue the Navigation step context by grounding its Navigation action to one available labeled waypoint candidate.
Use the progress updates, Navigation reasoning, waypoint reference context, and candidate overlays as grounding evidence.
Write concise grounding reasoning before the selected angle and candidate label.
""".strip()

VLN_WAYPOINT_ACTIVE_PROGRESS_SYSTEM_PROMPT = """
You are a navigation agent.
Continue the current navigation step by selecting one provided waypoint candidate and its clearest task-relevant RGB view based on the accumulated context.
""".strip()


@dataclass(frozen=True)
class VlnWaypointLoopResult:
    local_move_plan: LocalMovePlanDecision | None
    selected_view: VisualViewContext | None
    visual_action: VisualActionPointDecision | None
    waypoint: VisualWaypoint | None
    attempt_records: list[dict[str, object]] = field(default_factory=list)
    navigation_replan_feedback: list[dict[str, object]] = field(default_factory=list)
    failure_reason: str = ""


@dataclass(frozen=True)
class _VlnViewCandidateSet:
    view: VisualViewContext
    candidates: list[VlnSampledWaypointCandidate]
    rgb_overlay: np.ndarray


def _build_sampled_candidate_sets(
    *,
    exploration: "ExplorationManager",
    cache: "RuntimeCache",
    visual_context: VisualActionContext,
    robot_xy: np.ndarray,
    world_z: float,
    evidences: list[VlnLandmarkEvidence],
    frontier_records: dict[str, "LocalmapFrontierRecord"] | None,
    avoid_node_xys: list[tuple[float, float]] | None,
    node_dedup_map: "GlobalBEVMap | None" = None,
    node_dedup_radius_m: float = VLN_WAYPOINT_NODE_DEDUP_RADIUS_M,
    sample_spacing_m: float = VLN_WAYPOINT_SAMPLE_SPACING_M,
    max_distance_m: float = VLN_WAYPOINT_MAX_DISTANCE_M,
) -> tuple[list[_VlnViewCandidateSet], list[dict[str, object]]]:
    return _generate_all_view_sampled_candidates(
        exploration=exploration,
        cache=cache,
        visual_context=visual_context,
        robot_xy=robot_xy,
        world_z=float(world_z),
        evidences=evidences,
        avoid_node_xys=avoid_node_xys,
        frontier_records=frontier_records,
        node_dedup_map=node_dedup_map,
        node_dedup_radius_m=float(node_dedup_radius_m),
        sample_spacing_m=float(sample_spacing_m),
        max_distance_m=float(max_distance_m),
    )


def plan_vln_waypoint_loop(
    *,
    client: "LLMClient",
    cache: "RuntimeCache",
    goal_text: str,
    visual_context: VisualActionContext,
    task_progress_analysis: str,
    task_progress_text: str,
    exploration: "ExplorationManager",
    floor_height_m: float,
    goal_kind: str = "",
    planner_context_text: str = "",
    planner_context_images: list[tuple[str, np.ndarray]] | None = None,
    inherited_agent_context_content: list[dict[str, object]] | None = None,
    active_progress_item: str = "",
    candidate_bev_landmark_markers: list[dict[str, object]] | None = None,
    candidate_bev_base_image: object | None = None,
    candidate_bev_transform: "BevOverlayTransform | None" = None,
    candidate_bev_coordinate_exploration: "ExplorationManager | None" = None,
    candidate_bev_reference_node_marker: dict[str, object] | None = None,
    candidate_bev_context_text: str = "",
    landmark_evidences: list[VlnLandmarkEvidence] | None = None,
    frontier_records: dict[str, "LocalmapFrontierRecord"] | None = None,
    avoid_node_xys: list[tuple[float, float]] | None = None,
    node_dedup_map: "GlobalBEVMap | None" = None,
    node_dedup_radius_m: float = VLN_WAYPOINT_NODE_DEDUP_RADIUS_M,
    sample_spacing_m: float = VLN_WAYPOINT_SAMPLE_SPACING_M,
    max_distance_m: float = VLN_WAYPOINT_MAX_DISTANCE_M,
    show_cardinal_border: bool = False,
    heading_reference: str = "the current robot heading",
) -> VlnWaypointLoopResult:
    evidences: list[VlnLandmarkEvidence] = list(landmark_evidences or [])
    loop_events: list[dict[str, object]] = []
    attempt_records: list[dict[str, object]] = []
    navigation_replan_feedback: list[dict[str, object]] = []

    for _ in range(VLN_WAYPOINT_REPLAN_MAX_ATTEMPTS):
        robot_xy = _robot_xy_from_visual_context(cache=cache, visual_context=visual_context)
        view_candidate_sets, candidate_generation_failures = _build_sampled_candidate_sets(
            exploration=exploration,
            cache=cache,
            visual_context=visual_context,
            robot_xy=robot_xy,
            world_z=float(floor_height_m),
            evidences=evidences,
            frontier_records=frontier_records,
            avoid_node_xys=avoid_node_xys,
            node_dedup_map=node_dedup_map,
            node_dedup_radius_m=float(node_dedup_radius_m),
            sample_spacing_m=float(sample_spacing_m),
            max_distance_m=float(max_distance_m),
        )
        all_candidates = _all_view_candidates(view_candidate_sets)
        if all_candidates == []:
            loop_events.append(
                {
                    "event_type": "sampled_waypoint_candidates_unavailable",
                    "selected_angle_deg": None,
                    "selected_obs_id": "",
                    "decision": {},
                    "feedback": {
                        "candidate_count": 0,
                        "failure_reason": "no_projected_sampled_waypoint_candidates",
                        "view_failures": candidate_generation_failures,
                    },
                }
            )
            continue
        candidate_bev_overlay = draw_vln_sampled_waypoint_bev_overlay(
            exploration=exploration,
            robot_xy=robot_xy,
            candidates=all_candidates,
            landmark_markers=candidate_bev_landmark_markers,
            base_overlay=(
                None
                if candidate_bev_base_image is None
                else np.asarray(candidate_bev_base_image, dtype=np.uint8)
            ),
            base_transform=candidate_bev_transform,
            coordinate_exploration=candidate_bev_coordinate_exploration,
            reference_node_marker=candidate_bev_reference_node_marker,
            show_cardinal_border=bool(show_cardinal_border),
        )
        candidate_response = _run_vln_sampled_waypoint_candidate_selector(
            client=client,
            goal_text=goal_text,
            task_progress_analysis=task_progress_analysis,
            task_progress_text=task_progress_text,
            evidences=evidences,
            loop_events=loop_events,
            planner_context_text=planner_context_text,
            planner_context_images=planner_context_images,
            inherited_agent_context_content=inherited_agent_context_content,
            active_progress_item=active_progress_item,
            candidate_bev_landmark_markers=candidate_bev_landmark_markers,
            candidate_bev_context_text=candidate_bev_context_text,
            view_candidate_sets=view_candidate_sets,
            candidate_bev_overlay=candidate_bev_overlay,
            include_graph_context=visual_context.graph_context_visible,
            heading_reference=str(heading_reference),
        )
        selected_candidate, selected_view, candidate_failure_reason = _selected_sampled_candidate(
            response=candidate_response,
            view_candidate_sets=view_candidate_sets,
        )
        if selected_candidate is None or selected_view is None:
            loop_events.append(
                {
                    "event_type": "sampled_waypoint_selection_invalid",
                    "selected_angle_deg": candidate_response.get("selected_angle_deg"),
                    "selected_obs_id": "",
                    "decision": {},
                    "candidate_selection": deepcopy(candidate_response),
                    "feedback": {
                        "candidate_count": len(all_candidates),
                        "failure_reason": candidate_failure_reason,
                        "view_failures": candidate_generation_failures,
                    },
                    "overlay_images": {
                        "sampled_candidate_bev_overlay": candidate_bev_overlay,
                    },
                }
            )
            continue
        candidate_rgb_overlay = _rgb_overlay_for_view(
            view_candidate_sets=view_candidate_sets,
            selected_view=selected_view,
        )
        waypoint_target = f"sampled candidate label {int(selected_candidate.label)}"
        local_move_plan = LocalMovePlanDecision(
            selected_angle_deg=int(selected_view.angle_deg),
            waypoint_target=waypoint_target,
            reasoning=str(candidate_response.get("reasoning", "")).strip(),
            failure_reason="",
        )
        waypoint = visual_waypoint_from_sampled_candidate(
            selected_view=selected_view,
            candidate=selected_candidate,
            target=waypoint_target,
            raw_world_z=float(floor_height_m),
        )
        attempt_record = _attempt_record(
            decision=deepcopy(candidate_response),
            local_move_plan=local_move_plan,
            selected_view=selected_view,
            visual_action=None,
            verification=None,
            failure_reason="",
            overlay_images={
                "sampled_candidate_rgb_overlay": candidate_rgb_overlay,
                "sampled_candidate_bev_overlay": candidate_bev_overlay,
            },
            sampled_candidate=selected_candidate.to_dict(),
            candidate_selection=candidate_response,
            candidate_count=len(all_candidates),
        )
        attempt_records.append(attempt_record)
        return VlnWaypointLoopResult(
            local_move_plan=local_move_plan,
            selected_view=selected_view,
            visual_action=None,
            waypoint=waypoint,
            attempt_records=attempt_records,
            navigation_replan_feedback=navigation_replan_feedback,
        )

    candidate_generation_exhausted = (
        loop_events != []
        and all(
            str(item.get("event_type", ""))
            == "sampled_waypoint_candidates_unavailable"
            for item in loop_events
        )
    )
    return VlnWaypointLoopResult(
        local_move_plan=None,
        selected_view=None,
        visual_action=None,
        waypoint=None,
        attempt_records=attempt_records,
        navigation_replan_feedback=navigation_replan_feedback,
        failure_reason=(
            "no_projected_sampled_waypoint_candidates"
            if candidate_generation_exhausted
            else "vln_waypoint_loop_exhausted"
        ),
    )


def _generate_all_view_sampled_candidates(
    *,
    exploration: "ExplorationManager",
    cache: "RuntimeCache",
    visual_context: VisualActionContext,
    robot_xy: np.ndarray,
    world_z: float,
    evidences: list[VlnLandmarkEvidence],
    frontier_records: dict[str, "LocalmapFrontierRecord"] | None,
    avoid_node_xys: list[tuple[float, float]] | None,
    node_dedup_map: "GlobalBEVMap | None" = None,
    node_dedup_radius_m: float = VLN_WAYPOINT_NODE_DEDUP_RADIUS_M,
    sample_spacing_m: float = VLN_WAYPOINT_SAMPLE_SPACING_M,
    max_distance_m: float = VLN_WAYPOINT_MAX_DISTANCE_M,
) -> tuple[list[_VlnViewCandidateSet], list[dict[str, object]]]:
    view_candidate_sets: list[_VlnViewCandidateSet] = []
    failures: list[dict[str, object]] = []
    views = [
        visual_context.view_for_angle(int(angle))
        for angle in _waypoint_overlay_angles(visual_context.available_angles)
    ]
    try:
        projected_candidates = generate_vln_unified_sampled_waypoint_candidates(
            exploration=exploration,
            cache=cache,
            views=views,
            robot_xy=robot_xy,
            world_z=float(world_z),
            frontier_records=frontier_records,
            avoid_node_xys=avoid_node_xys,
            node_dedup_map=node_dedup_map,
            node_dedup_radius_m=float(node_dedup_radius_m),
            sample_spacing_m=float(sample_spacing_m),
            max_distance_m=float(max_distance_m),
        )
    except ValueError as exc:
        return [], [
            {
                "selected_angle_deg": None,
                "selected_obs_id": "",
                "failure_reason": f"unified_sampled_waypoint_generation_failed:{exc}",
            }
        ]
    candidates_by_angle: dict[int, list[VlnSampledWaypointCandidate]] = {}
    for item in projected_candidates:
        candidates_by_angle.setdefault(int(item.view.angle_deg), []).append(item.candidate)
    for selected_view in views:
        candidates = candidates_by_angle.get(int(selected_view.angle_deg), [])
        if candidates == []:
            failures.append(
                {
                    "selected_angle_deg": int(selected_view.angle_deg),
                    "selected_obs_id": str(selected_view.obs_id),
                    "failure_reason": "no_projected_sampled_waypoint_candidates_for_view",
                }
            )
        candidate_rgb_overlay = draw_vln_sampled_waypoint_rgb_overlay(
            cache=cache,
            selected_view=selected_view,
            candidates=candidates,
        )
        candidate_rgb_overlay = draw_vln_landmarks_on_view_image(
            image=candidate_rgb_overlay,
            cache=cache,
            view=selected_view,
            evidences=evidences,
        )
        view_candidate_sets.append(
            _VlnViewCandidateSet(
                view=selected_view,
                candidates=candidates,
                rgb_overlay=candidate_rgb_overlay,
            )
        )
    return view_candidate_sets, failures


def _all_view_candidates(view_candidate_sets: list[_VlnViewCandidateSet]) -> list[VlnSampledWaypointCandidate]:
    candidates_by_label: dict[int, VlnSampledWaypointCandidate] = {}
    for item in view_candidate_sets:
        for candidate in item.candidates:
            candidates_by_label.setdefault(int(candidate.label), candidate)
    return [
        candidates_by_label[label]
        for label in sorted(candidates_by_label)
    ]


def _rgb_overlay_for_view(
    *,
    view_candidate_sets: list[_VlnViewCandidateSet],
    selected_view: VisualViewContext,
) -> np.ndarray:
    for item in view_candidate_sets:
        if int(item.view.angle_deg) == int(selected_view.angle_deg):
            return np.asarray(item.rgb_overlay, dtype=np.uint8)
    raise ValueError(f"missing sampled waypoint overlay for angle_{int(selected_view.angle_deg)}")


def _run_vln_sampled_waypoint_candidate_selector(
    *,
    client: "LLMClient",
    goal_text: str,
    task_progress_analysis: str,
    task_progress_text: str,
    evidences: list[VlnLandmarkEvidence],
    loop_events: list[dict[str, object]],
    planner_context_text: str,
    planner_context_images: list[tuple[str, np.ndarray]] | None,
    inherited_agent_context_content: list[dict[str, object]] | None,
    active_progress_item: str = "",
    candidate_bev_landmark_markers: list[dict[str, object]] | None = None,
    candidate_bev_context_text: str = "",
    view_candidate_sets: list[_VlnViewCandidateSet],
    candidate_bev_overlay: np.ndarray,
    include_graph_context: bool,
    heading_reference: str = "the current robot heading",
) -> dict[str, object]:
    active_item = str(active_progress_item).strip()
    if active_item != "":
        prompt_text = _build_vln_active_progress_waypoint_prompt(
            loop_events=loop_events,
        )
    elif inherited_agent_context_content:
        prompt_text = _build_vln_inherited_waypoint_grounding_prompt(
            loop_events=loop_events,
        )
    else:
        prompt_text = _build_vln_sampled_waypoint_candidate_selector_prompt(
            goal_text=goal_text,
            task_progress_analysis=task_progress_analysis,
            task_progress_text=task_progress_text,
            evidences=evidences,
            loop_events=loop_events,
            planner_context_text=planner_context_text,
            view_candidate_sets=view_candidate_sets,
            include_graph_context=include_graph_context,
        )
    user_prompt: list[dict[str, object]] = []
    if not inherited_agent_context_content:
        user_prompt.append({"type": "text", "text": prompt_text})
    user_prompt.extend(deepcopy(list(inherited_agent_context_content or [])))
    _append_planner_context_blocks(
        user_prompt=user_prompt,
        planner_context_text=planner_context_text,
        planner_context_images=planner_context_images,
    )
    candidate_bev_content = [
        {
            "type": "text",
            "text": _candidate_bev_prompt_text(
                candidate_bev_landmark_markers,
                candidate_bev_context_text=candidate_bev_context_text,
            ),
        },
        image_content_for_array(candidate_bev_overlay),
    ]
    if inherited_agent_context_content:
        user_prompt.extend(candidate_bev_content)
    user_prompt.append(
        {
            "type": "text",
            "text": _waypoint_rgb_context_text(
                view_candidate_sets,
                evidences=evidences,
                include_graph_context=include_graph_context,
                heading_reference=str(heading_reference),
            ),
        }
    )
    for item in view_candidate_sets:
        user_prompt.append(
            {
                "type": "text",
                "text": _view_overlay_prompt_text(
                    item,
                    evidences=evidences,
                    include_graph_context=include_graph_context,
                ),
            }
        )
        user_prompt.append(
            image_content_for_array(np.asarray(item.rgb_overlay, dtype=np.uint8))
        )
    if not inherited_agent_context_content:
        user_prompt.extend(candidate_bev_content)
    if not inherited_agent_context_content:
        for block in _loop_context_blocks(loop_events):
            text = str(block.get("text", "")).strip()
            if text != "":
                user_prompt.append({"type": "text", "text": text})
            for label, image in list(block.get("images", [])):
                user_prompt.append({"type": "text", "text": str(label)})
                user_prompt.append(image_content_for_array(np.asarray(image, dtype=np.uint8)))
    if inherited_agent_context_content:
        user_prompt.append({"type": "text", "text": prompt_text})
    if active_item != "":
        system_prompt = VLN_WAYPOINT_ACTIVE_PROGRESS_SYSTEM_PROMPT
    elif inherited_agent_context_content:
        system_prompt = VLN_WAYPOINT_INHERITED_CONTEXT_SYSTEM_PROMPT
    else:
        system_prompt = VLN_WAYPOINT_CANDIDATE_SELECTOR_SYSTEM_PROMPT
    if active_item == "" and not inherited_agent_context_content and not (
        str(task_progress_analysis).strip() or str(task_progress_text).strip()
    ):
        system_prompt = """
You are the VLN Waypoint Planner of an embodied navigation agent.
You select one labeled local waypoint candidate that best advances the navigation instruction.
""".strip()
    return client._create_visual_json_completion(
        call_name="vln_waypoint_planner",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_new_tokens=4096,
        token_field="max_completion_tokens",
        retry_count=2,
        client_kind="la",
    )


def _append_planner_context_blocks(
    *,
    user_prompt: list[dict[str, object]],
    planner_context_text: str,
    planner_context_images: list[tuple[str, np.ndarray]] | None,
) -> None:
    context_text = str(planner_context_text).strip()
    images = list(planner_context_images or [])
    if context_text == "" and images == []:
        return
    if context_text != "":
        user_prompt.append({"type": "text", "text": f"Waypoint reference context:\n{context_text}"})
    for label, image in images:
        user_prompt.append({"type": "text", "text": str(label)})
        user_prompt.append(image_content_for_array(np.asarray(image, dtype=np.uint8)))


def _planner_context_reference_text(planner_context_text: str) -> str:
    if str(planner_context_text).strip() == "":
        return ""
    return "See the additional route context attached after this prompt text."


def _landmark_marker_labels_text(
    landmark_markers: list[dict[str, object]] | None,
) -> str:
    items = [
        f"{int(marker['label'])} {str(marker.get('class_name', '')).strip()}".strip()
        for marker in list(landmark_markers or [])
        if marker.get("label") is not None
    ]
    return ", ".join(items)


def _candidate_bev_prompt_text(
    landmark_markers: list[dict[str, object]] | None,
    *,
    candidate_bev_context_text: str = "",
) -> str:
    text = (
        "Candidate endpoint BEV. White numbered circles are the selectable waypoint "
        "endpoints and use the same labels as the RGB overlays."
    )
    graph_context = str(candidate_bev_context_text).strip()
    if graph_context != "":
        text += "\n" + graph_context
    landmark_text = _landmark_marker_labels_text(landmark_markers)
    if landmark_text != "":
        text += f"\nWhite numbered squares are landmarks: {landmark_text}."
    return text


def _build_vln_active_progress_waypoint_prompt(
    *,
    loop_events: list[dict[str, object]],
) -> str:
    retry_summary = _loop_context_summary(loop_events)
    retry_section = (
        f"Previous invalid selection:\n{retry_summary}\n\n"
        if retry_summary != "none"
        else ""
    )
    return f"""
{retry_section}
Rules:
- Choose exactly one candidate_label and one selected_angle_deg where that label is visible.
- candidate_label identifies one provided world-space waypoint.
- selected_angle_deg identifies the RGB view that provides the clearest task-relevant evidence for the selected candidate.
- Use the accumulated task progress, retrieval conclusions, and Navigation action reasoning to preserve the current navigation decision.
- Use the RGB overlays to compare semantic and route relevance.
- Use the candidate BEV to compare spatial direction and relative position.
- Select the candidate that best advances the current Navigation action.

Return exactly these JSON fields in the shown order:
{{
  "reasoning": "<why this candidate best advances the current Navigation action, based on the accumulated context and relevant RGB/BEV evidence>",
  "selected_angle_deg": 0,
  "candidate_label": 1
}}
""".strip()


def _build_vln_inherited_waypoint_grounding_prompt(
    *,
    loop_events: list[dict[str, object]],
) -> str:
    retry_summary = _loop_context_summary(loop_events)
    retry_section = (
        f"Previous invalid selection:\n{retry_summary}\n\n"
        if retry_summary != "none"
        else ""
    )
    grounding_rules = """
- Compare candidates across all attached directional RGB overlays.
- Compare candidate endpoint geometry in the candidate BEV.
- Use the clearest projection when one candidate appears in multiple views.
- Select visible connected walkable floor, stair tread or landing, doorway floor, corridor floor, or safe free space.
- Reject walls, ceilings, furniture or object surfaces, clutter, windows, mirrors, and door leaves.
""".strip()
    reasoning_schema = (
        "<compare candidate support for the inherited objective and cite the visible or "
        "inherited evidence for the selected candidate>"
    )
    selection_rules = """
- Choose exactly one candidate_label and one selected_angle_deg where that label is visible.
- candidate_label identifies the world-space waypoint; selected_angle_deg identifies its grounding view.
""".strip()
    output_schema = f"""
{{
  "reasoning": "{reasoning_schema}",
  "selected_angle_deg": 0,
  "candidate_label": 1
}}
""".strip()
    return f"""
{retry_section}
Rules:
{selection_rules}
{grounding_rules}

Return exactly these JSON fields in the shown order:
{output_schema}
""".strip()


def _build_vln_sampled_waypoint_candidate_selector_prompt(
    *,
    goal_text: str,
    task_progress_analysis: str,
    task_progress_text: str,
    evidences: list[VlnLandmarkEvidence],
    loop_events: list[dict[str, object]],
    planner_context_text: str,
    view_candidate_sets: list[_VlnViewCandidateSet],
    include_graph_context: bool,
) -> str:
    planner_context_reference = _planner_context_reference_text(planner_context_text)
    planner_context_section = ""
    if planner_context_reference != "":
        planner_context_section = f"""

Additional route context:
{planner_context_reference}
""".rstrip()
    has_progress = bool(
        str(task_progress_analysis).strip() or str(task_progress_text).strip()
    )
    progress_section = ""
    task_instruction = "Analyze the navigation task"
    if has_progress:
        progress_section = f"""
Route progress:
{task_progress_analysis}

Updated task progress memory:
{task_progress_text}
""".rstrip()
        task_instruction = "Analyze the current active VLN progress item"
    overlay_contents = "view angle, available waypoint labels, and visible landmarks"
    if include_graph_context:
        overlay_contents = (
            "view angle, available waypoint labels, visible visited nodes, and visible landmarks"
        )
    evidence_description = f"""
{len(view_candidate_sets)} sampled waypoint RGB overlays are attached separately after this prompt.
Each overlay is immediately preceded by its {overlay_contents}.
The same waypoint label may appear in multiple overlays; repeated labels refer to the same world-space waypoint.
A candidate BEV containing the same labeled endpoints is also attached.
""".strip()
    evidence_rules = """
- Compare candidates across the separately provided directional overlays.
- Compare candidate endpoint geometry in the candidate BEV.
- Use the navigation task, available route context, and RGB overlays to compare candidates.
- When one label appears in multiple views, use the view that provides the clearest task-relevant evidence for that waypoint.
- Base the decision on visible support for the navigation task.
- Choose a candidate on visible connected walkable floor, stair tread/landing, doorway floor, corridor floor, or safe free space.
- Reject candidates that appear on walls, ceilings, furniture/object surfaces, clutter, windows, mirrors, or door leaves.
""".strip()
    if has_progress:
        evidence_rules = """
- Compare candidates across the separately provided directional overlays.
- Compare candidate endpoint geometry in the candidate BEV.
- Use the updated task progress memory and RGB overlays to compare candidates.
- When one label appears in multiple views, use the view that provides the clearest task-relevant evidence for that waypoint.
- Base the decision on visible support for the current active progress item. Use later progress items only as a tie-breaker among candidates that already support the current active item.
- Choose a candidate on visible connected walkable floor, stair tread/landing, doorway floor, corridor floor, or safe free space.
- Reject candidates that appear on walls, ceilings, furniture/object surfaces, clutter, windows, mirrors, or door leaves.
""".strip()
        progress_evidence_schema = (
            "<visible RGB overlay evidence that supports the current active progress item>"
        )
        reasoning_schema = (
            "<identify the current active progress item, compare candidate support for that "
            "item across the provided views, and explain why the selected waypoint best advances it>"
        )
    else:
        progress_evidence_schema = (
            "<visible RGB overlay evidence that supports the navigation task>"
        )
        reasoning_schema = (
            "<compare candidate support for the navigation task across the provided views "
            "and explain why the selected waypoint best advances it>"
        )
    return f"""
Navigation task:
{goal_text}
{progress_section}

Waypoint retry context:
{_loop_context_summary(loop_events)}
{planner_context_section}

{evidence_description}

Task:
{task_instruction} and select the sampled waypoint candidate that best advances it.

Rules:
- Choose exactly one candidate_label and one selected_angle_deg where that label is visible.
- candidate_label determines the world-space waypoint; selected_angle_deg determines which RGB projection of that waypoint is used.
{evidence_rules}
- Do not choose a label merely because it is closest to the agent.
- Choose the best candidate from the provided labels; leave failure_reason empty after a valid selection.

Return JSON only:
{{
  "progress_alignment_evidence": "{progress_evidence_schema}",
  "reasoning": "{reasoning_schema}",
  "selected_angle_deg": 0,
  "candidate_label": 1,
  "failure_reason": ""
}}
""".strip()


def _waypoint_overlay_angles(available_angles: list[int]) -> list[int]:
    available = {int(angle) for angle in available_angles}
    preferred = [0, 90, 180, 270]
    ordered = [angle for angle in preferred if angle in available]
    ordered.extend(sorted(angle for angle in available if angle not in set(preferred)))
    return ordered


def _waypoint_rgb_context_text(
    view_candidate_sets: list[_VlnViewCandidateSet],
    *,
    evidences: list[VlnLandmarkEvidence],
    include_graph_context: bool,
    heading_reference: str = "the current robot heading",
) -> str:
    lines = [
        f"Waypoint RGB overlays: {len(view_candidate_sets)} directional views.",
        angle_convention_text(
            [int(item.view.angle_deg) for item in view_candidate_sets],
            heading_reference=str(heading_reference),
        ),
        "- Numbered circles are waypoint candidates.",
        "- The same label in multiple views is one world-space waypoint.",
    ]
    if include_graph_context:
        lines.append("- Blue numbered circles are visited place nodes.")
    if evidences != []:
        lines.append(
            "- Landmark boxes display the numeric suffix of each canonical landmark ref."
        )
    return "\n".join(lines)


def _robot_xy_from_visual_context(
    *,
    cache: "RuntimeCache",
    visual_context: VisualActionContext,
) -> np.ndarray:
    current_view = visual_context.view_for_angle(0)
    observation = cache.get_observation(str(current_view.obs_id)).observation
    if observation.T_odom_base is not None:
        transform = np.asarray(observation.T_odom_base, dtype=np.float64)
        return np.asarray([float(transform[0, 3]), float(transform[1, 3])], dtype=np.float64)
    return np.asarray([float(observation.pose.x), float(observation.pose.y)], dtype=np.float64)


def _candidate_labels_text(candidates: list[VlnSampledWaypointCandidate]) -> str:
    labels = [str(int(candidate.label)) for candidate in candidates]
    return ", ".join(labels) if labels != [] else "none"


def _visible_node_labels_text(view: "VisualViewContext") -> str:
    items: list[str] = []
    for node_id in list(getattr(view, "visible_visited_nodes", [])):
        node_id_text = str(node_id).strip()
        if node_id_text == "":
            continue
        items.append(node_id_text)
    return ", ".join(items) if items != [] else "none"


def _view_overlay_prompt_text(
    view_candidate_set: _VlnViewCandidateSet,
    *,
    evidences: list[VlnLandmarkEvidence],
    include_graph_context: bool,
) -> str:
    angle = int(view_candidate_set.view.angle_deg)
    lines = [
        f"Sampled waypoint RGB overlay for angle_{angle}.",
        f"- Available waypoint labels: {_candidate_labels_text(view_candidate_set.candidates)}.",
    ]
    if include_graph_context:
        visible_nodes = _visible_node_labels_text(view_candidate_set.view)
        if visible_nodes != "none":
            lines.append(f"- Visible visited nodes: {visible_nodes}.")
    visible_landmarks = _landmark_labels_text(evidences, angle_deg=angle)
    if visible_landmarks != "none":
        lines.append(f"- Visible landmarks: {visible_landmarks}.")
    return "\n".join(lines)


def _landmark_labels_text(evidences: list[VlnLandmarkEvidence], *, angle_deg: int) -> str:
    items: list[str] = []
    for evidence in evidences:
        if int(evidence.angle_deg) != int(angle_deg):
            continue
        class_name = str(evidence.class_name).strip()
        landmark_id = str(evidence.landmark_id).strip()
        if class_name == "" or landmark_id == "":
            continue
        item = f"{landmark_id} {class_name} confidence={float(evidence.score):.2f}"
        if item not in items:
            items.append(item)
    return ", ".join(items) if items != [] else "none"


def _selected_sampled_candidate(
    *,
    response: dict[str, object],
    view_candidate_sets: list[_VlnViewCandidateSet],
) -> tuple[VlnSampledWaypointCandidate | None, VisualViewContext | None, str]:
    reasoning = str(response.get("reasoning", "")).strip()
    if reasoning == "":
        return None, None, "candidate_selector_missing_reasoning"
    response_fields = list(response)
    if "reasoning" in response_fields:
        reasoning_index = response_fields.index("reasoning")
        selection_indices = [
            response_fields.index(field_name)
            for field_name in ("selected_angle_deg", "candidate_label")
            if field_name in response_fields
        ]
        if selection_indices and reasoning_index > min(selection_indices):
            return None, None, "candidate_selector_reasoning_must_precede_selection"
    raw_label = response.get("candidate_label")
    if raw_label is None:
        return None, None, str(response.get("failure_reason", "candidate_selector_returned_null_label")).strip()
    raw_angle = response.get("selected_angle_deg")
    if raw_angle is None:
        return None, None, "candidate_selector_missing_selected_angle_deg"
    try:
        selected_angle = int(raw_angle)
    except (TypeError, ValueError):
        return None, None, f"candidate_selector_invalid_selected_angle_deg:{raw_angle!r}"
    try:
        label = int(raw_label)
    except (TypeError, ValueError):
        return None, None, f"candidate_selector_invalid_label:{raw_label!r}"
    available_angles: list[int] = []
    for item in view_candidate_sets:
        for candidate in item.candidates:
            if int(candidate.label) != label:
                continue
            angle = int(item.view.angle_deg)
            available_angles.append(angle)
            if angle == selected_angle:
                return candidate, item.view, ""
    if available_angles != []:
        angles_text = "_".join(str(angle) for angle in available_angles)
        return None, None, (
            f"candidate_selector_angle_label_mismatch:"
            f"angle_{selected_angle}_label_{label}_available_at_angles_{angles_text}"
        )
    return None, None, f"candidate_selector_unavailable_label:{label}"


def _attempt_record(
    *,
    decision: dict[str, object],
    local_move_plan: LocalMovePlanDecision,
    selected_view: VisualViewContext,
    visual_action: VisualActionPointDecision | None,
    verification: VisualWaypointVerificationDecision | None,
    failure_reason: str,
    overlay_images: dict[str, np.ndarray] | None = None,
    sampled_candidate: dict[str, object] | None = None,
    candidate_selection: dict[str, object] | None = None,
    candidate_count: int | None = None,
) -> dict[str, object]:
    return {
        "decision": deepcopy(decision),
        "selected_angle_deg": int(selected_view.angle_deg),
        "selected_obs_id": str(selected_view.obs_id),
        "waypoint_target": str(local_move_plan.waypoint_target),
        "visual_action": None if visual_action is None else visual_action,
        "verification": verification,
        "failure_reason": str(failure_reason),
        "overlay_images": {} if overlay_images is None else dict(overlay_images),
        "sampled_candidate": {} if sampled_candidate is None else deepcopy(sampled_candidate),
        "candidate_selection": {} if candidate_selection is None else deepcopy(candidate_selection),
        "candidate_count": None if candidate_count is None else int(candidate_count),
    }


def _loop_event_from_attempt(attempt: dict[str, object]) -> dict[str, object]:
    visual_action = attempt.get("visual_action")
    verification = attempt.get("verification")
    return {
        "event_type": "sampled_waypoint_selection",
        "selected_angle_deg": attempt.get("selected_angle_deg"),
        "selected_obs_id": attempt.get("selected_obs_id"),
        "waypoint_target": attempt.get("waypoint_target"),
        "decision": deepcopy(attempt.get("decision", {})),
        "visual_action": None if visual_action is None else visual_action.to_dict(),
        "verification": None if verification is None else verification.to_dict(),
        "failure_reason": str(attempt.get("failure_reason", "")),
        "overlay_images": dict(attempt.get("overlay_images", {})),
        "sampled_candidate": deepcopy(attempt.get("sampled_candidate", {})),
        "candidate_selection": deepcopy(attempt.get("candidate_selection", {})),
        "candidate_count": attempt.get("candidate_count"),
    }


def _loop_context_summary(loop_events: list[dict[str, object]]) -> str:
    if loop_events == []:
        return "none"
    lines: list[str] = []
    for index, event in enumerate(loop_events, start=1):
        event_type = str(event.get("event_type", "")).strip()
        if event_type in {"sampled_waypoint_candidates", "sampled_waypoint_candidates_unavailable"}:
            feedback = event.get("feedback", {})
            lines.append(
                f"Previous attempt {index}: no projected sampled waypoint candidates were available. "
                f"Feedback: {json.dumps(feedback, ensure_ascii=False)}"
            )
            continue
        if event_type == "sampled_waypoint_selection_invalid":
            feedback = event.get("feedback", {})
            lines.append(
                f"Previous planner response {index}: the response did not identify a valid sampled waypoint candidate. "
                f"Feedback: {json.dumps(feedback, ensure_ascii=False)}"
            )
            continue
        if event_type == "sampled_waypoint_selection":
            selection = event.get("candidate_selection", {})
            lines.append(
                f"Previous attempt {index}: selected angle_{event.get('selected_angle_deg')} "
                f"with candidate details {json.dumps(selection, ensure_ascii=False)}. "
                f"Failure: {event.get('failure_reason', '')}"
            )
            continue
        verification = event.get("verification")
        lines.append(
            f"Previous attempt {index}: angle_{event.get('selected_angle_deg')} "
            f"target={event.get('waypoint_target')} validation={json.dumps(verification, ensure_ascii=False)} "
            f"failure={event.get('failure_reason', '')}"
        )
    return "\n".join(lines)


def _loop_context_blocks(loop_events: list[dict[str, object]]) -> list[dict[str, object]]:
    blocks: list[dict[str, object]] = []
    for index, event in enumerate(loop_events, start=1):
        text = _loop_context_summary([event])
        images: list[tuple[str, np.ndarray]] = []
        overlays = event.get("overlay_images", {})
        if isinstance(overlays, dict):
            for key in (
                "sampled_candidate_rgb_overlay",
                "sampled_candidate_bev_overlay",
            ):
                image = overlays.get(key)
                if isinstance(image, np.ndarray):
                    images.append((f"Loop context event {index} {key}", image))
        blocks.append({"text": text, "images": images})
    return blocks
