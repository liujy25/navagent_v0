from __future__ import annotations

from copy import deepcopy
import json
from typing import TYPE_CHECKING

import numpy as np

from navclaw.agent.episodic_retrieval import RetrieveRequest
from navclaw.agent.episodic_retrieval import normalize_retrieve_request
from navclaw.agent.visual_action_context import direction_for_angle
from navclaw.agent.visual_action_context import VisualActionContext, VisualViewContext
from navclaw.agent.visual_policy_decisions import NavigationModeDecision
from navclaw.agent.visual_policy_decisions import StopConfirmationDecision
from navclaw.agent.visual_policy_decisions import TaskProgressDecision
from navclaw.agent.visual_policy_decisions import VisualActionPointDecision
from navclaw.agent.visual_policy_decisions import VisualWaypointVerificationDecision
from navclaw.agent.visual_policy_decisions import normalize_stop_confirmation
from navclaw.agent.visual_policy_decisions import normalize_visual_action_point
from navclaw.agent.visual_policy_decisions import normalize_vertical_transition_visual_action_point
from navclaw.agent.visual_policy_decisions import normalize_visual_waypoint_verification
from navclaw.agent.visual_policy_prompt_images import current_panorama_prompt_text
from navclaw.agent.visual_policy_prompt_images import image_content_for_array
from navclaw.agent.visual_policy_prompt_images import image_content_for_movement_history_sheet
from navclaw.agent.visual_policy_prompt_images import image_content_for_current_panorama_views
from navclaw.agent.visual_policy_prompt_images import image_content_for_stop_waypoint_overlay
from navclaw.agent.visual_policy_prompt_images import image_content_for_view
from navclaw.agent.visual_policy_prompt_images import selected_view_prompt_text
from navclaw.llm.image_preprocessing import LLM_CAMERA_IMAGE_MAX_SIZE
from navclaw.llm.image_preprocessing import resize_rgb_to_fit
from navclaw.memory.task_progress import TaskProgressItem, TaskProgressMemory
from navclaw.perception.goal_identity import GOAL_KIND_VLN_INSTRUCTION
from navclaw.visualization.waypoint_overlay import normalized_point_to_pixel

if TYPE_CHECKING:
    from navclaw.llm.client import LLMClient
    from navclaw.runtime.cache import RuntimeCache


TRAJECTORY_RGB_KEYFRAME_DISTANCE_M = 0.8
TRAJECTORY_RGB_KEYFRAME_YAW_DEG = 45.0
VLN_PROGRESS_NAVIGATION_SEMANTIC_MAX_ATTEMPTS = 3
VLN_HORIZONTAL_DIRECTIONS = {"front", "back", "left", "right"}


def ensure_task_progress_memory(
    *,
    client: "LLMClient",
    task_progress: TaskProgressMemory,
    goal_text: str,
    task_type: str = "",
    cache: "RuntimeCache | None" = None,
    visual_context: VisualActionContext | None = None,
) -> dict[str, object]:
    current_node_id = (
        str(visual_context.current_node_id)
        if visual_context is not None
        else ""
    )
    if task_progress.items != []:
        task_progress.ensure_current_task_started(current_node_id)
        return {"initialized": False, "source": "existing", "task_progress": task_progress.to_dict()}
    normalized_task_type = str(task_type).strip()
    if normalized_task_type not in {"", GOAL_KIND_VLN_INSTRUCTION}:
        raise ValueError("task progress supports VLN instructions only")
    system_prompt = """
You are the Initial Task Progress Generator in NavClaw.
Convert the navigation instruction into an ordered list of route-level subtasks and a final stopping requirement when one is stated.
Do not assess visual evidence, create online progress conditions, retrieve history, or choose a navigation action.
Return only valid JSON matching the provided output contract.
""".strip()
    user_prompt = f"""
Navigation instruction:
{goal_text}

Decision objective:
Create the initial ordered task-progress items.

Task-item semantics:
- Each item is one route-level subtask whose completion can later be assessed from navigation evidence, or one final stopping requirement.
- TPU creates and maintains online evidence conditions during navigation.
- The system maintains node associations outside item content.

Decomposition rules:
- Preserve instruction order.
- Create a new item when a distinct movement stage must be completed before the next stage should guide navigation.
- Keep one movement and the landmarks or spatial relations that define that stage together.
- Separate a prerequisite stage from a later target when the prerequisite must be completed before that target guides navigation.
- Preserve route-defining relations such as pass, enter, exit, through, between, first/next, left/right of a landmark, and final stop/wait relations.
- Keep descriptive modifiers with the target they identify.
- Do not create a direction-only item for panorama reorientation such as "look left" or "face the door"; retain directional wording in the route stage it constrains.
- Do not invent landmarks, rooms, turns, or intermediate actions.
- Use concise action-oriented item content.

Initialization rules:
- Set every item status to "active".
- Set every item result to an empty string.

Output contract:
{{
  "progress_items": [
    {{
      "content": "<route-level subtask or final stopping requirement>",
      "status": "active",
      "result": ""
    }}
  ]
}}
""".strip()
    prompt_payload: str | list[dict[str, object]] = user_prompt
    source = "llm"
    try:
        parsed = client.generate_task_progress_memory(system_prompt, prompt_payload)
        if not isinstance(parsed, dict) or set(parsed) != {"progress_items"}:
            raise ValueError(
                "VLN initial task-progress output must contain only progress_items: "
                f"{parsed!r}"
            )
        raw_items = parsed.get("progress_items", [])
        if not isinstance(raw_items, list) or raw_items == []:
            raise ValueError(f"visual task progress generator returned no progress_items: {parsed!r}")
        expected_fields = {"content", "status", "result"}
        for item in raw_items:
            if not isinstance(item, dict) or set(item) != expected_fields:
                raise ValueError(
                    "VLN initial task-progress items must contain exactly "
                    f"{sorted(expected_fields)!r}: {item!r}"
                )
            if (
                str(item.get("content", "")).strip() == ""
                or str(item.get("status", "")).strip() != "active"
                or str(item.get("result", "")).strip() != ""
            ):
                raise ValueError(
                    "VLN initial task-progress items require non-empty content, "
                    f"status='active', and result='': {item!r}"
                )
        task_progress.items = [
            TaskProgressItem.from_dict(item)
            for item in raw_items
            if isinstance(item, dict)
        ]
        if task_progress.items == []:
            raise ValueError(f"visual task progress generator returned invalid progress_items: {parsed!r}")
    except Exception as exc:
        source = "fallback"
        task_progress.items = TaskProgressMemory.fallback(goal_text).items
        task_progress.ensure_current_task_started(current_node_id)
        return {
            "initialized": True,
            "source": source,
            "error": str(exc),
            "task_progress": task_progress.to_dict(),
        }
    task_progress.ensure_current_task_started(current_node_id)
    return {"initialized": True, "source": source, "task_progress": task_progress.to_dict()}






_VLN_TASK_PROGRESS_TOOLS = {
    "retrieve",
    "update_progress",
}

_VLN_NAVIGATION_TOOLS = {
    "backtrack",
    "go_to_waypoint",
    "approach_to_stop",
    "vertical_transition",
}


def _vln_progress_update_rules(
    node_binding_rule: str,
    *,
    goal_kind: str,
) -> str:
    if str(goal_kind).strip() == GOAL_KIND_VLN_INSTRUCTION:
        completion_rules = """
- Assess task-progress items in instruction order and interpret each item in the context of the ordered route.
- Evidence may complete several consecutive items only when it independently establishes each one.
- Mark an item `done` only when its required action and spatial relations have been completed and the resulting state is consistent with any adjacent route stage that constrains its interpretation.
- Use the preceding item to interpret the route context from which the current item is executed.
- Use the following item when it disambiguates which execution or route transition of the current item is intended.
- Do not require the following item itself to be completed before marking the current item `done`.
- If the current item's local action has occurred but its consistency with the surrounding route remains unresolved, keep the item `active`.
- Reopen a `done` item when later evidence shows that its previously accepted execution was inconsistent with the ordered route.
""".strip()
        condition_creation_rules = """
- Add a transition condition only when an adjacent task-progress item provides information needed to disambiguate the valid completion of the parent item.
- The condition should express the route relation that must be established, rather than restating either subtask.
- Do not add a transition condition when the parent item's completion can already be determined from its own execution evidence.
- Add any other condition only for a distinct partial-completion fact needed to assess its parent subtask.
- Do not add unrelated scene notes, duplicates, or a restatement of the full subtask.
""".strip()
    else:
        completion_rules = """
- Assess task-progress items in instruction order. Evidence may complete several consecutive items only when it independently establishes each one.
- Mark a subtask `done` only when its required action and spatial relations have been completed. Keep it `active` when the target is only visible, only part of the relation is established, or decisive evidence is missing.
- Reopen a `done` item only when current evidence directly invalidates the earlier completion judgment.
""".strip()
        condition_creation_rules = """
- Add a condition only for a distinct partial-completion fact needed to assess its parent subtask. Do not add unrelated scene notes, duplicates, or a restatement of the full subtask.
""".strip()
    return f"""
Progress update rules:
{completion_rules}
{node_binding_rule}
{condition_creation_rules}
- `update` changes only a condition state. `rewrite` refines or corrects the same underlying fact. `remove` deletes only a duplicate, irrelevant, or invalid condition; missing evidence does not justify removing an unconfirmed condition.
- Determine item status from the full subtask semantics and execution evidence, not by counting confirmed conditions.
- If the evidence supports no condition change, omit `update_progress_conditions`. If it supports no item change, use an empty `progress_updates` list.
""".strip()








def decide_vln_task_progress_step(
    *,
    client: "LLMClient",
    cache: "RuntimeCache",
    goal_kind: str = GOAL_KIND_VLN_INSTRUCTION,
    visual_context: VisualActionContext,
    task_progress: TaskProgressMemory,
    memory_index_text: str,
    retrieval_workspace_content: list[dict[str, object]],
    retrieve_max_rounds: int,
    retrieve_completed_rounds: int,
    retrieve_fields_by_ref: dict[str, list[str]],
    allow_retrieve: bool,
    allow_update_progress: bool,
    require_retrieval_conclusion: bool,
    retrieve_provided_fields_by_ref: dict[str, list[str]] | None = None,
    memory_index_context_only: bool = False,
    progress_context_text: str = "",
    backtrack_context_text: str = "",
    landmark_panorama_views: dict[int, np.ndarray] | None = None,
    detected_landmarks_text: str = "",
    terminal_check_context: dict[str, object] | None = None,
    task_progress_bev_overlay: np.ndarray | None = None,
    planning_reference_panorama: bool = False,
) -> TaskProgressDecision | RetrieveRequest:
    decision = _run_vln_task_module_prompt(
        client=client,
        cache=cache,
        goal_kind=goal_kind,
        visual_context=visual_context,
        task_progress=task_progress,
        memory_index_text=memory_index_text,
        retrieval_workspace_content=retrieval_workspace_content,
        retrieve_max_rounds=retrieve_max_rounds,
        retrieve_completed_rounds=retrieve_completed_rounds,
        retrieve_fields_by_ref=retrieve_fields_by_ref,
        allow_retrieve=allow_retrieve,
        allow_update_progress=allow_update_progress,
        allow_navigation_actions=False,
        require_retrieval_conclusion=require_retrieval_conclusion,
        retrieve_provided_fields_by_ref=retrieve_provided_fields_by_ref,
        memory_index_context_only=memory_index_context_only,
        progress_context_text=progress_context_text,
        backtrack_context_text=backtrack_context_text,
        landmark_panorama_views=landmark_panorama_views,
        detected_landmarks_text=detected_landmarks_text,
        terminal_check_context=terminal_check_context,
        task_progress_bev_overlay=task_progress_bev_overlay,
        planning_reference_panorama=planning_reference_panorama,
    )
    if not isinstance(decision, (TaskProgressDecision, RetrieveRequest)):
        raise TypeError("Task Progress Updater returned a navigation decision")
    return decision


def decide_vln_navigation_step(
    *,
    client: "LLMClient",
    cache: "RuntimeCache",
    goal_kind: str = GOAL_KIND_VLN_INSTRUCTION,
    visual_context: VisualActionContext,
    task_progress: TaskProgressMemory,
    latest_task_progress: TaskProgressDecision,
    retrieval_workspace_content: list[dict[str, object]],
    progress_context_text: str = "",
    backtrack_context_text: str = "",
    navigation_replan_feedback_text: str = "",
    landmark_panorama_views: dict[int, np.ndarray] | None = None,
    detected_landmarks_text: str = "",
    allowed_backtrack_node_ids: set[str] | None = None,
    require_backtrack: bool = False,
    system_owned_waypoint_objective: bool = False,
    task_progress_bev_overlay: np.ndarray | None = None,
    planning_reference_panorama: bool = False,
) -> NavigationModeDecision:
    decision = _run_vln_task_module_prompt(
        client=client,
        cache=cache,
        goal_kind=goal_kind,
        visual_context=visual_context,
        task_progress=task_progress,
        memory_index_text="",
        retrieval_workspace_content=retrieval_workspace_content,
        retrieve_max_rounds=0,
        retrieve_completed_rounds=0,
        retrieve_fields_by_ref={},
        allow_retrieve=False,
        allow_update_progress=False,
        allow_navigation_actions=True,
        require_retrieval_conclusion=False,
        latest_task_progress=latest_task_progress,
        progress_context_text=progress_context_text,
        backtrack_context_text=backtrack_context_text,
        navigation_replan_feedback_text=navigation_replan_feedback_text,
        landmark_panorama_views=landmark_panorama_views,
        detected_landmarks_text=detected_landmarks_text,
        allowed_backtrack_node_ids=allowed_backtrack_node_ids,
        require_backtrack=require_backtrack,
        system_owned_waypoint_objective=(
            str(goal_kind).strip() == GOAL_KIND_VLN_INSTRUCTION
            or system_owned_waypoint_objective
        ),
        task_progress_bev_overlay=task_progress_bev_overlay,
        planning_reference_panorama=planning_reference_panorama,
    )
    if not isinstance(decision, NavigationModeDecision):
        raise TypeError("PCNP returned a task-progress decision")
    return decision


def _run_vln_task_module_prompt(
    *,
    client: "LLMClient",
    cache: "RuntimeCache",
    goal_kind: str = GOAL_KIND_VLN_INSTRUCTION,
    visual_context: VisualActionContext,
    task_progress: TaskProgressMemory,
    memory_index_text: str,
    retrieval_workspace_content: list[dict[str, object]],
    retrieve_max_rounds: int,
    retrieve_completed_rounds: int,
    retrieve_fields_by_ref: dict[str, list[str]],
    allow_retrieve: bool,
    allow_update_progress: bool,
    allow_navigation_actions: bool,
    require_retrieval_conclusion: bool,
    retrieve_provided_fields_by_ref: dict[str, list[str]] | None = None,
    memory_index_context_only: bool = False,
    latest_task_progress: TaskProgressDecision | None = None,
    progress_context_text: str = "",
    backtrack_context_text: str = "",
    navigation_replan_feedback_text: str = "",
    landmark_panorama_views: dict[int, np.ndarray] | None = None,
    detected_landmarks_text: str = "",
    allowed_backtrack_node_ids: set[str] | None = None,
    require_backtrack: bool = False,
    system_owned_waypoint_objective: bool = False,
    terminal_check_context: dict[str, object] | None = None,
    task_progress_bev_overlay: np.ndarray | None = None,
    planning_reference_panorama: bool = False,
) -> TaskProgressDecision | RetrieveRequest | NavigationModeDecision:
    if str(goal_kind).strip() != GOAL_KIND_VLN_INSTRUCTION:
        raise ValueError("progress-navigation supports VLN instructions only")
    if not any((allow_retrieve, allow_update_progress, allow_navigation_actions)):
        raise ValueError("VLN task module has no available operation")
    if allow_navigation_actions and (allow_retrieve or allow_update_progress):
        raise ValueError(
            "Task Progress Updater and Navigation Planner operations cannot share a request"
        )
    if allow_navigation_actions and latest_task_progress is None:
        raise ValueError("navigation action tools require current task progress")
    if require_backtrack and (
        not allow_navigation_actions or not allowed_backtrack_node_ids
    ):
        raise ValueError("required `backtrack` needs at least one available anchor")
    terminal_check_required = isinstance(terminal_check_context, dict)
    if terminal_check_required and not allow_update_progress:
        raise ValueError("post-approach terminal check requires `update_progress`")
    tool_schemas: list[str] = []
    if allow_retrieve or allow_update_progress:
        tool_schemas.append(
            """### `update_progress_conditions`
Purpose:
- Apply evidence-backed changes to the dynamic condition sets of task-progress items.

Use when:
- At least one condition must be added, have its state updated, be rewritten, or be removed using evidence already in context.

Arguments:
- `condition_updates`: a non-empty list of condition operations.

Operations:
- `add` requires `op`, `item_index`, `content`, `status`, and `reason`.
- `update` requires `op`, `item_index`, `condition_id`, `status`, and `reason`; it changes only the state.
- `rewrite` requires `op`, `item_index`, `condition_id`, `content`, `status`, and `reason`; it preserves the underlying proposition while refining its wording.
- `remove` requires `op`, `item_index`, `condition_id`, and `reason`.

Constraints:
- This call is optional. If present, it is the first entry in `tool_calls`.
- Every operation states its decisive evidence in `reason`.
- When the final call is `retrieve`, these operations use only evidence available before the requested fields are loaded.

JSON:
{"name":"update_progress_conditions","arguments":{"condition_updates":[{"op":"update","item_index":0,"condition_id":"pc0","status":"confirmed","reason":"<decisive evidence>"}]}}"""
        )
    if allow_retrieve:
        tool_schemas.append(
            """### `retrieve`
Purpose:
- Load selected historical entity fields needed to resolve one task-progress verification query.

Use when:
- Current evidence is insufficient for a concrete progress judgment and the memory index exposes evidence that can resolve it.

Arguments:
- `query`: one focused verification question.
- `items`: a non-empty list of memory refs and requested available fields.

Constraints:
- Every ref and field exists in the supplied memory index.
- Exclude fields already marked as provided.
- Request the minimum sufficient evidence for the query.
- This is the final entry in `tool_calls` and continues TPU reasoning in a later round.

JSON:
{"name":"retrieve","arguments":{"query":"<focused verification question>","items":[{"ref":"<memory ref>","fields":["<available field>"]}]}}"""
        )
    if allow_update_progress:
        terminal_check_schema = (
            ',"terminal_check":{'
            '"decision":"done|continue",'
            '"reason":"<endpoint judgment>",'
            '"missing_constraints":[]}'
            if terminal_check_required
            else ""
        )
        tool_schemas.append(
            """### `update_progress`
Purpose:
- Commit item-level status and result changes and end TPU reasoning for this decision step or planning reference.

Arguments:
- `progress_updates`: a list of item updates; use an empty list when no item change is supported.

Operations:
- Only `update` is allowed.
- `update` requires `op`, `index`, `status`, `result`, and `reason`.
- Item content, order, and count are immutable after initialization.

Constraints:
- `status` is `active` or `done`.
- `result` is empty for `active` and a concise past-tense completion result for `done`.
- Do not add, insert, rewrite, remove, reorder, or rename task-progress items.
- A contradicted completion may be changed from `done` to `active` with an empty result and evidence-backed reason.
- This is the final entry in `tool_calls` and ends the TPU retrieval loop.

JSON:
{"name":"update_progress","arguments":{"progress_updates":[{"op":"update","index":0,"status":"done","result":"Passed the pool.","reason":"<decisive execution evidence>"}]"""
            + terminal_check_schema
            + "}}"
        )
    if allow_navigation_actions:
        if allowed_backtrack_node_ids:
            allowed_backtrack_refs = ", ".join(
                sorted(str(node_id) for node_id in allowed_backtrack_node_ids)
            )
            required_anchor_constraint = (
                "- This is the only available action in the current state."
                if require_backtrack
                else ""
            )
            tool_schemas.append(
                f"""### `backtrack`
Purpose:
- Re-evaluate navigation from a previously visited anchor node by switching the planning reference to that node.

Use when:
- Historical evidence suggests that a required branch, turn, opening, or route choice may have been missed at an earlier visited location.
- The committed task-progress state conflicts with the executed route and an earlier viewpoint must be reconsidered.
- An earlier visited viewpoint can resolve a concrete route-recovery need better than the current planning reference.

Arguments:
- `anchor_node_id`: one of {allowed_backtrack_refs}.
- `objective`: the route feature, branch, opening, or task relation to reassess from that anchor.
- `reason`: the evidence-based reason to reconsider that viewpoint now.

Constraints:
{required_anchor_constraint}
- Selecting `backtrack` does not physically move the robot.
- TPU and PCNP rerun with the anchor observation and graph context as their planning reference.
- The physical robot node remains unchanged until a later grounded movement is executed.
- The anchor panorama is stored planning-reference evidence, not a fresh observation from the robot's physical pose.

JSON:
{{"backtrack":{{"anchor_node_id":"<one of: {allowed_backtrack_refs}>","objective":"<route relation to reassess>","reason":"<why this anchor is needed>"}}}}"""
            )
        if not require_backtrack:
            tool_schemas.extend(
                [
                    """### `go_to_waypoint`
Purpose:
- Continue instruction following toward a locally groundable route or region.

Use when:
- The next step is neither a final stopping approach nor a floor transition.

Arguments:
- `direction`: the local panorama direction in which the world-space waypoint should advance the route.
- `reason`: the visible route or region and its relation to the active task stage.

Constraints:
- Select the route direction from current evidence; do not copy an instruction direction that referred to an earlier pose.
- The Waypoint Planner chooses the exact candidate.

JSON:
{"go_to_waypoint":{"direction":"front|back|left|right","reason":"<semantic local route intent>"}}""",
                    """### `approach_to_stop`
Purpose:
- Perform the final local approach toward the expected stopping region. TPU performs the terminal check at the next decision step.

Use when:
- The expected stopping region is visible and the remaining task requires local positioning within or near it.

Arguments:
- `movement`: `move` for one local movement or `stay` when the observed pose already occupies the stopping region.
- `direction`: required only for `move`; the local panorama direction of the approach.
- `stop_objective`: the terminal spatial relation to establish.
- `reason`: the visible evidence supporting the approach or stay proposal.

Constraints:
- This action does not declare the episode complete.
- Target visibility at a distance is insufficient.

JSON for `move`:
{"approach_to_stop":{"movement":"move","direction":"front|back|left|right","stop_objective":"<terminal spatial relation>","reason":"<visible approach evidence>"}}

JSON for `stay`:
{"approach_to_stop":{"movement":"stay","stop_objective":"<terminal spatial relation at the observed pose>","reason":"<visible endpoint evidence>"}}""",
                    """### `vertical_transition`
Purpose:
- Begin an up- or down-stair transition required by the current task stage.

Use when:
- The task requires a floor change and the corresponding stair route is visible and locally accessible.

Arguments:
- `vertical_direction`: `up` or `down`.
- `reason`: the visible stair route, its accessibility, and why it is required now.

Constraints:
- A visible staircase unrelated to the active task stage does not justify this action.
- The vertical-transition grounding modules choose the local movement points.

JSON:
{"vertical_transition":{"vertical_direction":"up|down","reason":"<visible required stair route>"}}""",
                ]
            )

    retrieve_remaining = max(
        0,
        int(retrieve_max_rounds) - int(retrieve_completed_rounds),
    )
    retrieval_catalog_section = ""
    retrieval_semantics = ""
    if allow_retrieve:
        single_retrieval_instruction = (
            "\n\nSingle-retrieval requirement:\n"
            "- This is the only RETRIEVE call available for this navigation step.\n"
            "- Use this one request to retrieve every memory entity and available field "
            "needed to resolve the current historical evidence question.\n"
            "- Include all required items together in the single `items` list."
            if int(retrieve_max_rounds) == 1
            and int(retrieve_completed_rounds) == 0
            else ""
        )
        retrieval_catalog_section = f"""

Episodic-memory text index:
{memory_index_text if str(memory_index_text).strip() != "" else "none"}

Retrieval budget for this navigation step:
- maximum rounds: {int(retrieve_max_rounds)}
- completed rounds: {int(retrieve_completed_rounds)}
- remaining rounds: {int(retrieve_remaining)}
{single_retrieval_instruction}
""".rstrip()
        retrieval_semantics = """
Memory field semantics:
- `node.rgb`: the node's stored multi-direction panorama.
- `node.landmarks`: all landmark detections annotated in that panorama.
- `edge.rgb`: the stored start view with the executed trajectory and endpoint overlaid.
- `edge.trajectory`: the executed edge trajectory highlighted on the shared-floor BEV.
- `edge.movement_rgb`: traversal RGB keyframes ordered chronologically from left to right and then top to bottom.
- `landmark.rgb`: a local historical crop around the landmark with its bounding box.
- `provided_fields` are already present in this context; `available_fields` may be requested.
- Landmark refs use `landmark_<global index>`; image boxes display only the numeric suffix.
""".strip()
    elif memory_index_context_only:
        retrieval_catalog_section = f"""

Episodic-memory text index:
{memory_index_text if str(memory_index_text).strip() != "" else "none"}

Only the compact text index is available in this condition. No stored RGB, trajectory,
landmark overlay, or graph BEV evidence is attached.
""".rstrip()
    progress_context = str(progress_context_text).strip()
    progress_context_section = (
        f"\n\nCurrent-step context:\n{progress_context}"
        if progress_context != ""
        else ""
    )
    backtrack_context = str(backtrack_context_text).strip()
    backtrack_context_section = (
        f"Backtrack context:\n{backtrack_context}"
        if backtrack_context != ""
        and not backtrack_context.startswith("Backtrack context:")
        else backtrack_context
    )
    latest_progress_section = ""
    if allow_navigation_actions and latest_task_progress is not None:
        latest_progress_lines: list[str] = []
        if latest_task_progress.progress_updates:
            latest_progress_lines.extend(
                [
                    "Latest progress updates:",
                    json.dumps(
                        latest_task_progress.progress_updates,
                        ensure_ascii=False,
                        indent=2,
                    ),
                ]
            )
        if str(latest_task_progress.terminal_check_decision).strip() != "":
            latest_progress_lines.extend(
                [
                    f"terminal_check: {latest_task_progress.terminal_check_decision}",
                    f"terminal_reason: {latest_task_progress.terminal_check_reasoning}",
                    "terminal_missing_constraints: "
                    + json.dumps(
                        latest_task_progress.terminal_check_missing_constraints,
                        ensure_ascii=False,
                    ),
                ]
            )
        if latest_progress_lines:
            latest_progress_section = "\n\n" + "\n".join(latest_progress_lines)
    replan_feedback = str(navigation_replan_feedback_text).strip()
    replan_feedback_section = (
        f"\n\nRejected local navigation attempt:\n{replan_feedback}"
        if replan_feedback != ""
        else ""
    )
    landmark_text = str(detected_landmarks_text).strip()
    conclusion_rules = (
        """
Latest unresolved retrieval round:
- Raw evidence for the latest retrieval round is attached and has no conclusion yet.
- `retrieval_conclusion` directly answers that query, names the supporting refs, and states any remaining evidence gap.
""".strip()
        if require_retrieval_conclusion
        else ""
    )
    if str(goal_kind).strip() == GOAL_KIND_VLN_INSTRUCTION:
        task_progress_semantics = """
Task-progress semantics:
- A task-progress item is one stage of the ordered navigation route or the final stopping requirement. Its content and order are fixed after initialization.
- `done` means that the item's action and spatial relations have been completed consistently with the surrounding route context.
- The preceding and following task-progress items may constrain how the current item should be interpreted and whether its execution constitutes valid completion.
- Completion of a following item is not required to complete the current item; it is used only when it helps disambiguate the intended route transition.
- A progress condition is an evidence-checkable fact used to assess the completion of its parent item.
- A condition may describe progress within the parent item or a route relation connecting the parent item to an adjacent task-progress item.
- A transition condition captures whether the state produced by the parent item is consistent with the route stage before or after it.
- Conditions are supporting state, not separate subtasks.
- TPU creates and revises conditions online to preserve partial progress and unresolved historical-verification needs.
- `confirmed` means the available evidence establishes the exact condition.
- `unconfirmed` means the condition is not yet established; it does not mean the condition is false.
- The condition set is dynamic and is not a fixed completion checklist.
- Determine the parent status from the full item semantics and execution evidence, not by counting confirmed conditions.
- `result` is empty for an active item and a concise past-tense completion summary for a done item.
- Start and completion node associations are system-owned and do not belong in item content, result, or condition content.
""".strip()
    else:
        task_progress_semantics = """
Task-progress semantics:
- A task-progress item is one ordered route-level subtask or final stopping requirement. Its content and order are fixed after initialization.
- A progress condition is one evidence-checkable partial-completion fact under its parent subtask; it is supporting state, not a separate subtask.
- TPU creates and revises conditions online to preserve partial progress and unresolved historical-verification needs.
- `confirmed` means the available evidence establishes the exact condition.
- `unconfirmed` means the condition is not established; it does not mean the condition is false.
- Determine the parent status from the full subtask semantics and execution evidence; conditions are not a fixed completion checklist.
- `result` is empty for an active item and a concise past-tense completion summary for a done item.
- Start and completion node associations are system-owned and do not belong in item content, result, or condition content.
""".strip()
    evidence_retrieval_policy = """
Evidence and retrieval policy:
- Judge each progress relation only from evidence that establishes it. Seeing a target does not prove that the required movement, passage, turn, entry, approach, or stop relation was executed.
- Distinguish visibility from room membership. An object visible through a doorway or opening may belong to an adjacent room. Determine the agent's current room from spatial boundaries and entry/exit evidence, not from object visibility alone.
- Use entity knowledge as a compact prior. Retrieve raw historical fields only when they can resolve a concrete subtask-status or condition judgment.
- Do not retrieve for generic scene understanding, waypoint comparison, or unspecified additional context.
- Each retrieval query asks one focused progress-verification question and requests only the entity fields needed to answer it.
- After inspecting new evidence, answer the query, name the supporting refs, and state any remaining evidence gap.
- A later query targets the remaining gap instead of repeating a resolved question.
- When current evidence is sufficient, no suitable historical field exists, or the budget is exhausted, commit the evidence-backed progress state without inventing facts.
- A retrieval request is not evidence; updates emitted with `retrieve` use only evidence available before the requested fields are loaded.
""".strip()
    navigation_state_semantics = """
State semantics:
- `confirmed` conditions are established by the available evidence.
- `unconfirmed` conditions remain unresolved and must not be treated as completed facts.
- Task-progress memory has already been committed by TPU. Do not modify or reinterpret its structure in this module.
- Retrieval conclusions summarize inspected historical evidence. Use them as historical decision evidence; do not request raw fields from this module.
""".strip()
    navigation_action_rules = (
        """
Action-selection rules:
- Choose the action that best advances the earliest active route stage while remaining consistent with completed stages and confirmed conditions.
- Use the current panorama to identify a visible and actionable local route, stopping region, or stair transition.
- Use retrieval conclusions and the cumulative BEV to account for historical route evidence, visited locations, and traversed geometry.
- Do not use an action choice to conceal unresolved task progress; TPU has already committed the evidence-supported state for this step.
- Action parameters express semantic route intent, not a waypoint label, metric coordinate, or image point.
- The action `reason` identifies the decisive visible route or historical evidence and explains how the action advances the current task stage.
- Interpret directional task language in the current route context instead of copying a left/right word tied to an earlier pose.
- After rejected local grounding, choose a newly supported objective rather than repeating the rejected route without new evidence.
""".strip()
        if allow_navigation_actions
        else ""
    )
    node_binding_rule = (
        "- `node span` is maintained by the system; do not copy it into `content` or `result`."
        if visual_context.graph_context_visible
        else ""
    )
    terminal_check_section = ""
    if terminal_check_required:
        stop_objective = str(terminal_check_context.get("stop_objective", "")).strip()
        approach_movement = str(
            terminal_check_context.get("movement", "move")
        ).strip()
        approach_result = (
            "The `approach_to_stop` decision retained the current pose."
            if approach_movement == "stay"
            else "The latest `approach_to_stop` movement reached the current pose."
        )
        terminal_check_section = f"""

Post-approach terminal check:
- {approach_result}
- Reassess the ordered task and final stopping constraints at this endpoint.
- `done` requires every task item to be done after the proposed updates and no missing terminal constraint.
- `continue` requires an active task after the proposed updates and names the remaining constraints.

Approach objective:
{stop_objective}
""".rstrip()
    system_prompt = (
        """
You are the Progress-Conditioned Navigation Planner (PCNP) in NavClaw.
Choose exactly one high-level navigation action from the updated task-progress state, current observation, retrieval conclusions, and cumulative BEV when available.
Do not update task progress, retrieve historical fields, or choose a metric waypoint or image point.
Return only valid JSON matching one of the provided action contracts.
""".strip()
        if allow_navigation_actions
        else """
You are the Task Progress Updater (TPU) in NavClaw.
Assess task progress from the available evidence.
If a specific progress judgment remains unresolved, retrieve the historical evidence needed to verify it; otherwise commit the progress update.
Do not choose navigation actions or metric waypoints.
Return only valid JSON matching the response protocol.
""".strip()
    )
    task_progress_section = (
        ("Updated task-progress memory:" if allow_navigation_actions else "Task progress memory:")
        + "\n"
        + task_progress.format_for_prompt(
            include_node_bindings=visual_context.graph_context_visible,
            show_empty_progress_conditions=True,
        )
    )
    response_protocol = (
        ("Action contracts:\n\n" + "\n\n".join(tool_schemas))
        if allow_navigation_actions
        else (
            "Tool definitions:\n\n"
            + "\n\n".join(tool_schemas)
            + "\n\nResponse protocol:\n"
            + "- `tool_calls` contains one or two calls.\n"
            + "- `update_progress_conditions` is optional; if present, it is first and contains at least one operation.\n"
            + "- The final call is exactly one of `retrieve` or `update_progress`.\n"
            + "- Include a non-empty top-level `retrieval_conclusion` only when raw evidence for the latest unresolved retrieval round is attached.\n"
            + "- Otherwise omit `retrieval_conclusion`.\n"
            + "- Include no other top-level fields.\n\n"
            + "Without a new retrieval conclusion:\n"
            + '{"tool_calls":[{"name":"<final tool name>","arguments":{}}]}\n\n'
            + "With a new retrieval conclusion:\n"
            + '{"retrieval_conclusion":"<direct answer to the latest verification query>","tool_calls":[{"name":"<final tool name>","arguments":{}}]}'
        )
    )
    decision_text = "\n\n".join(
        section
        for section in (
            navigation_state_semantics if allow_navigation_actions else task_progress_semantics,
            retrieval_semantics,
            evidence_retrieval_policy
            if allow_retrieve or allow_update_progress
            else "",
            _vln_progress_update_rules(
                node_binding_rule,
                goal_kind=goal_kind,
            )
            if allow_retrieve or allow_update_progress
            else "",
            conclusion_rules,
            navigation_action_rules,
            response_protocol,
        )
        if section != ""
    )
    content: list[dict[str, object]] = [
        {"type": "text", "text": task_progress_section}
    ]
    if progress_context_section != "":
        content.append({"type": "text", "text": progress_context_section.strip()})
    if not allow_navigation_actions and terminal_check_section != "":
        content.append({"type": "text", "text": terminal_check_section.strip()})
    if allow_navigation_actions and latest_progress_section != "":
        content.append({"type": "text", "text": latest_progress_section.strip()})
    if not allow_navigation_actions and backtrack_context_section != "":
        content.append({"type": "text", "text": backtrack_context_section})
    observation_reference_lines = [
        (
            f"Planning reference node: {visual_context.current_node_id}"
            if planning_reference_panorama
            else f"Observation reference node: {visual_context.current_node_id}"
        )
    ]
    if planning_reference_panorama:
        observation_reference_lines.append(
            "The attached panorama is stored planning-reference evidence, not a fresh observation from the robot's physical pose."
        )
    if landmark_text != "":
        observation_reference_lines.append(landmark_text)
    content.append({"type": "text", "text": "\n".join(observation_reference_lines)})
    panorama_text = current_panorama_prompt_text(
        visual_context,
        include_visited_nodes=visual_context.graph_context_visible,
        planning_reference=bool(planning_reference_panorama),
    )
    if landmark_text != "":
        panorama_text += (
            "\n- Detected landmark boxes display the numeric suffix of each canonical "
            "landmark ref."
        )
    content.append({"type": "text", "text": panorama_text})
    content.extend(
        image_content_for_current_panorama_views(
            cache=cache,
            views=visual_context.views,
            include_visited_nodes=visual_context.graph_context_visible,
            image_overrides_by_angle=landmark_panorama_views,
            planning_reference=bool(planning_reference_panorama),
        )
    )
    if task_progress_bev_overlay is not None:
        landmark_bev_text = (
            " Numbered squares are current detected landmarks."
            if landmark_text != ""
            else ""
        )
        content.extend(
            [
                {
                    "type": "text",
                    "text": (
                        "Current-step BEV memory. Blue numbered circles are "
                        "visited graph nodes; magenta lines are visited move "
                        f"edges/agent path.{landmark_bev_text} No frontier or "
                        "waypoint candidates are drawn."
                    ),
                },
                image_content_for_array(
                    np.asarray(task_progress_bev_overlay, dtype=np.uint8)
                ),
            ]
        )
    if terminal_check_required:
        raw_movement_obs_ids = terminal_check_context.get("rgb_history_obs_ids", [])
        movement_obs_ids = (
            [str(obs_id) for obs_id in raw_movement_obs_ids]
            if isinstance(raw_movement_obs_ids, (list, tuple))
            else []
        )
        movement_history_content = image_content_for_movement_history_sheet(
            cache=cache,
            obs_ids=movement_obs_ids,
        )
        if movement_history_content is not None:
            content.append(
                {
                    "type": "text",
                    "text": (
                        "Post-approach movement RGB. Frames are ordered "
                        "chronologically from left to right and then top to bottom."
                    ),
                }
            )
            content.append(movement_history_content)
    content.extend(deepcopy(retrieval_workspace_content))
    if retrieval_catalog_section != "":
        content.append(
            {"type": "text", "text": retrieval_catalog_section.strip()}
        )
    if allow_navigation_actions and replan_feedback_section != "":
        content.append({"type": "text", "text": replan_feedback_section.strip()})
    if allow_navigation_actions and backtrack_context_section != "":
        content.append({"type": "text", "text": backtrack_context_section})
    content.append({"type": "text", "text": decision_text})
    retry_feedback = ""
    for attempt_index in range(VLN_PROGRESS_NAVIGATION_SEMANTIC_MAX_ATTEMPTS):
        request_content = list(content)
        if retry_feedback != "":
            request_content.append({"type": "text", "text": retry_feedback})
        parsed = (
            client.decide_vln_navigation_step(system_prompt, request_content)
            if allow_navigation_actions
            else client.decide_vln_task_progress_step(
                system_prompt,
                request_content,
            )
        )
        try:
            if allow_navigation_actions:
                if latest_task_progress is None:
                    raise ValueError(
                        "PCNP requires current task progress"
                    )
                return normalize_vln_navigation_step(
                    parsed,
                    latest_task_progress=latest_task_progress,
                    allowed_backtrack_node_ids=allowed_backtrack_node_ids,
                    require_backtrack=require_backtrack,
                    system_owned_waypoint_objective=(
                        system_owned_waypoint_objective
                    ),
                )
            return normalize_vln_task_progress_step(
                parsed,
                retrieve_fields_by_ref=retrieve_fields_by_ref,
                retrieve_provided_fields_by_ref=retrieve_provided_fields_by_ref,
                allow_retrieve=allow_retrieve,
                allow_update_progress=allow_update_progress,
                require_retrieval_conclusion=require_retrieval_conclusion,
                require_terminal_check=terminal_check_required,
            )
        except ValueError as exc:
            if (
                attempt_index + 1
                >= VLN_PROGRESS_NAVIGATION_SEMANTIC_MAX_ATTEMPTS
            ):
                raise
            retry_feedback = (
                "Validation feedback from the previous response:\n"
                f"{json.dumps(parsed, ensure_ascii=False, indent=2)}\n\n"
                "Validation error:\n"
                f"{exc}\n\n"
                "Return one corrected response using the same output contract."
            )
    raise RuntimeError(
        "VLN progress-navigation semantic retry loop exited unexpectedly"
    )


def _normalize_vln_progress_condition_updates(
    raw_updates: object,
) -> list[dict[str, object]]:
    if not isinstance(raw_updates, list):
        raise ValueError("progress_condition_updates must be a list")
    normalized: list[dict[str, object]] = []
    for raw_update in raw_updates:
        if not isinstance(raw_update, dict):
            raise ValueError(f"progress condition update must be an object: {raw_update!r}")
        op = str(raw_update.get("op", "")).strip().lower()
        reason_field = _operation_reason_field(raw_update)
        raw_id_field = (
            "condition_id"
            if "condition_id" in raw_update
            else "knowledge_id"
            if "knowledge_id" in raw_update
            else "condition_id"
        )
        expected_fields = {
            "add": {"op", "item_index", "content", "status", reason_field},
            "update": {"op", "item_index", raw_id_field, "status", reason_field},
            "rewrite": {
                "op",
                "item_index",
                raw_id_field,
                "content",
                "status",
                reason_field,
            },
            "remove": {"op", "item_index", raw_id_field, reason_field},
        }.get(op)
        if expected_fields is None or set(raw_update) != expected_fields:
            raise ValueError(f"invalid progress condition update fields: {raw_update!r}")
        reason = str(raw_update.get(reason_field, "")).strip()
        if reason == "":
            raise ValueError(f"progress condition update requires reason: {raw_update!r}")
        item_index = raw_update.get("item_index")
        if not isinstance(item_index, int) or isinstance(item_index, bool):
            raise ValueError(
                f"progress condition item_index must be an integer: {raw_update!r}"
            )
        item: dict[str, object] = {
            "op": op,
            "item_index": int(item_index),
        }
        if raw_id_field in expected_fields:
            condition_id = str(raw_update.get(raw_id_field, "")).strip()
            if condition_id == "":
                raise ValueError(f"{op} progress condition update requires condition_id")
            item["condition_id"] = condition_id
        if "content" in expected_fields:
            content = str(raw_update.get("content", "")).strip()
            if content == "":
                raise ValueError(f"{op} progress condition update requires content")
            item["content"] = content
        if "status" in expected_fields:
            status = str(raw_update.get("status", "")).strip().lower()
            status = {"missing": "unconfirmed", "satisfied": "confirmed"}.get(
                status,
                status,
            )
            if status not in {"unconfirmed", "confirmed"}:
                raise ValueError(f"invalid progress condition status: {raw_update!r}")
            item["status"] = status
        item["reason"] = reason
        normalized.append(item)
    return normalized


def _normalize_vln_progress_updates(raw_updates: object) -> list[dict[str, object]]:
    if not isinstance(raw_updates, list):
        raise ValueError("update_progress progress_updates must be a list")
    normalized: list[dict[str, object]] = []
    for raw_update in raw_updates:
        if not isinstance(raw_update, dict):
            raise ValueError(f"progress update must be an object: {raw_update!r}")
        op = str(raw_update.get("op", "")).strip().lower()
        if op != "update":
            raise ValueError(
                "task-progress item content and order are immutable; only "
                f"op='update' is allowed: {raw_update!r}"
            )
        reason_field = _operation_reason_field(raw_update)
        reason = str(raw_update.get(reason_field, "")).strip()
        if reason == "":
            raise ValueError(f"progress update requires reason: {raw_update!r}")
        expected_fields = {"op", "index", "status", "result", reason_field}
        if set(raw_update) != expected_fields:
            raise ValueError(
                f"{op} progress update fields must be exactly "
                f"{sorted(expected_fields)!r}: {raw_update!r}"
            )
        if "index" in expected_fields and (
            not isinstance(raw_update.get("index"), int)
            or isinstance(raw_update.get("index"), bool)
        ):
            raise ValueError(f"progress update index must be an integer: {raw_update!r}")
        status = str(raw_update.get("status", "")).strip().lower()
        if status not in {"active", "done"}:
            raise ValueError(f"invalid progress item status: {raw_update!r}")
        item: dict[str, object] = {
            "op": op,
            "index": int(raw_update["index"]),
            "status": status,
        }
        if status == "done":
            if str(raw_update.get("result", "")).strip() == "":
                raise ValueError(f"done progress update requires result: {raw_update!r}")
            item["result"] = str(raw_update.get("result", "")).strip()
        elif str(raw_update.get("result", "")).strip() != "":
            raise ValueError(f"active progress update requires empty result: {raw_update!r}")
        else:
            item["result"] = ""
        item["reason"] = reason
        normalized.append(item)
    return normalized


def _operation_reason_field(raw_update: dict[str, object]) -> str:
    fields = [
        field_name
        for field_name in ("reason", "thought", "reasoning")
        if field_name in raw_update
    ]
    if len(fields) != 1:
        raise ValueError(
            "operation must contain exactly one reason field; canonical output "
            f"uses reason: {raw_update!r}"
        )
    return fields[0]


def normalize_vln_task_progress_step(
    payload: dict[str, object],
    *,
    retrieve_fields_by_ref: dict[str, list[str]],
    allow_retrieve: bool,
    allow_update_progress: bool,
    require_retrieval_conclusion: bool,
    retrieve_provided_fields_by_ref: dict[str, list[str]] | None = None,
    require_terminal_check: bool = False,
) -> TaskProgressDecision | RetrieveRequest:
    decision = _normalize_vln_task_module_step(
        payload,
        retrieve_fields_by_ref=retrieve_fields_by_ref,
        retrieve_provided_fields_by_ref=retrieve_provided_fields_by_ref,
        allow_retrieve=allow_retrieve,
        allow_update_progress=allow_update_progress,
        allow_navigation_actions=False,
        require_retrieval_conclusion=require_retrieval_conclusion,
        latest_task_progress=None,
        require_terminal_check=require_terminal_check,
    )
    if not isinstance(decision, (TaskProgressDecision, RetrieveRequest)):
        raise TypeError("Task Progress Updater normalized a navigation decision")
    return decision


def normalize_vln_navigation_step(
    payload: dict[str, object],
    *,
    latest_task_progress: TaskProgressDecision,
    allowed_backtrack_node_ids: set[str] | None = None,
    require_backtrack: bool = False,
    system_owned_waypoint_objective: bool = False,
) -> NavigationModeDecision:
    decision = _normalize_vln_task_module_step(
        payload,
        retrieve_fields_by_ref={},
        allow_retrieve=False,
        allow_update_progress=False,
        allow_navigation_actions=True,
        require_retrieval_conclusion=False,
        latest_task_progress=latest_task_progress,
        allowed_backtrack_node_ids=allowed_backtrack_node_ids,
        require_backtrack=require_backtrack,
        system_owned_waypoint_objective=system_owned_waypoint_objective,
    )
    if not isinstance(decision, NavigationModeDecision):
        raise TypeError("Navigation Planner normalized a task-progress decision")
    return decision


def _normalize_vln_task_module_step(
    payload: dict[str, object],
    *,
    retrieve_fields_by_ref: dict[str, list[str]],
    allow_retrieve: bool,
    allow_update_progress: bool,
    allow_navigation_actions: bool,
    require_retrieval_conclusion: bool,
    latest_task_progress: TaskProgressDecision | None,
    retrieve_provided_fields_by_ref: dict[str, list[str]] | None = None,
    allowed_backtrack_node_ids: set[str] | None = None,
    require_backtrack: bool = False,
    system_owned_waypoint_objective: bool = False,
    require_terminal_check: bool = False,
) -> TaskProgressDecision | RetrieveRequest | NavigationModeDecision:
    module_name = (
        "Progress-Conditioned Navigation Planner (PCNP)"
        if allow_navigation_actions
        else "Task Progress Updater"
    )
    module_tools = (
        _VLN_NAVIGATION_TOOLS
        if allow_navigation_actions
        else _VLN_TASK_PROGRESS_TOOLS
    )
    if not allow_navigation_actions and "tool_calls" in payload:
        payload = _normalize_vln_tpu_tool_calls(
            payload,
            require_retrieval_conclusion=require_retrieval_conclusion,
        )
    selected_tools = [name for name in module_tools if name in payload]
    if len(selected_tools) != 1:
        raise ValueError(
            f"{module_name} must select exactly one available operation: "
            f"{payload!r}"
        )
    selected_tool = selected_tools[0]
    if require_backtrack and selected_tool != "backtrack":
        raise ValueError(
            f"only backtrack is available in this state: {payload!r}"
        )
    expected_top_fields = {selected_tool}
    if require_retrieval_conclusion:
        expected_top_fields.add("retrieval_conclusion")
    condition_updates_field = ""
    if selected_tool in {"retrieve", "update_progress"}:
        condition_update_fields = [
            field_name
            for field_name in (
                "progress_condition_updates",
                "progress_knowledge_updates",
            )
            if field_name in payload
        ]
        if len(condition_update_fields) != 1:
            raise ValueError(
                "Task Progress Updater must return exactly one "
                "progress_condition_updates field"
            )
        condition_updates_field = condition_update_fields[0]
        expected_top_fields.add(condition_updates_field)
    if set(payload) != expected_top_fields:
        raise ValueError(
            f"{module_name} returned unexpected top-level fields: "
            f"{payload!r}"
        )
    retrieval_conclusion = str(payload.get("retrieval_conclusion", "")).strip()
    if require_retrieval_conclusion and retrieval_conclusion == "":
        raise ValueError(
            "Task Progress Updater must conclude the latest retrieval "
            f"before its next tool: {payload!r}"
        )

    if selected_tool == "retrieve":
        if not allow_retrieve:
            raise ValueError(f"retrieve is not available in this state: {payload!r}")
        progress_condition_updates = _normalize_vln_progress_condition_updates(
            payload.get(condition_updates_field)
        )
        request = normalize_retrieve_request(
            payload,
            fields_by_ref=retrieve_fields_by_ref,
            provided_fields_by_ref=retrieve_provided_fields_by_ref,
            require_retrieval_conclusion=require_retrieval_conclusion,
        )
        return RetrieveRequest(
            query=request.query,
            items=request.items,
            already_provided_items=request.already_provided_items,
            retrieval_conclusion=request.retrieval_conclusion,
            progress_condition_updates=tuple(progress_condition_updates),
        )

    raw_tool = payload.get(selected_tool)
    if not isinstance(raw_tool, dict):
        raise ValueError(f"{selected_tool} tool payload must be an object: {payload!r}")
    if selected_tool == "update_progress":
        if not allow_update_progress:
            raise ValueError(f"update_progress is not available in this state: {payload!r}")
        expected_fields = {"progress_updates"}
        if require_terminal_check:
            expected_fields.add("terminal_check")
        if set(raw_tool) != expected_fields:
            raise ValueError(
                "update_progress fields must be exactly "
                f"{sorted(expected_fields)!r}: {payload!r}"
            )
        try:
            progress_updates = _normalize_vln_progress_updates(
                raw_tool.get("progress_updates")
            )
            progress_condition_updates = _normalize_vln_progress_condition_updates(
                payload.get(condition_updates_field)
            )
        except ValueError as exc:
            raise ValueError(f"{exc}: {payload!r}") from exc
        terminal_check_decision = ""
        terminal_check_reasoning = ""
        terminal_check_missing_constraints: list[str] = []
        if require_terminal_check:
            raw_terminal_check = raw_tool.get("terminal_check")
            if not isinstance(raw_terminal_check, dict):
                raise ValueError(f"UPDATE_PROGRESS terminal_check must be an object: {payload!r}")
            terminal_reason_field = _operation_reason_field(raw_terminal_check)
            expected_terminal_fields = {
                "decision",
                terminal_reason_field,
                "missing_constraints",
            }
            if set(raw_terminal_check) != expected_terminal_fields:
                raise ValueError(
                    "terminal_check fields must be exactly "
                    f"{sorted(expected_terminal_fields)!r}: {payload!r}"
                )
            terminal_check_decision = str(
                raw_terminal_check.get("decision", "")
            ).strip()
            terminal_check_reasoning = str(
                raw_terminal_check.get(terminal_reason_field, "")
            ).strip()
            raw_missing_constraints = raw_terminal_check.get("missing_constraints")
            if terminal_check_decision not in {"done", "continue"}:
                raise ValueError(f"terminal_check decision must be done or continue: {payload!r}")
            if terminal_check_reasoning == "":
                raise ValueError(f"terminal_check requires reason: {payload!r}")
            if not isinstance(raw_missing_constraints, list):
                raise ValueError(f"terminal_check missing_constraints must be a list: {payload!r}")
            terminal_check_missing_constraints = [
                str(item).strip()
                for item in raw_missing_constraints
                if str(item).strip() != ""
            ]
            if terminal_check_decision == "done" and terminal_check_missing_constraints:
                raise ValueError(f"done terminal_check cannot have missing constraints: {payload!r}")
            if terminal_check_decision == "continue" and not terminal_check_missing_constraints:
                raise ValueError(f"continue terminal_check requires missing constraints: {payload!r}")
        return TaskProgressDecision(
            progress_analysis="",
            progress_reasoning="",
            retrieval_conclusion=retrieval_conclusion,
            route_status="",
            status_reasoning="",
            progress_updates=progress_updates,
            progress_condition_updates=progress_condition_updates,
            terminal_check_decision=terminal_check_decision,
            terminal_check_reasoning=terminal_check_reasoning,
            terminal_check_missing_constraints=terminal_check_missing_constraints,
        )

    if not allow_navigation_actions or latest_task_progress is None:
        raise ValueError(f"{selected_tool} is not available before update_progress: {payload!r}")
    progress_fields = {
        "progress_analysis": "",
        "progress_reasoning": "",
        "progress_updates": [],
    }
    if selected_tool == "backtrack":
        expected_fields = {"anchor_node_id", "objective", "reason"}
        if set(raw_tool) != expected_fields:
            raise ValueError(f"backtrack fields must be exactly {sorted(expected_fields)!r}: {payload!r}")
        anchor_node_id = str(raw_tool.get("anchor_node_id", "")).strip()
        objective = str(raw_tool.get("objective", "")).strip()
        reason = str(raw_tool.get("reason", "")).strip()
        if allowed_backtrack_node_ids is not None and anchor_node_id not in allowed_backtrack_node_ids:
            raise ValueError(f"backtrack selected unavailable anchor node: {payload!r}")
        if anchor_node_id == "" or objective == "" or reason == "":
            raise ValueError(f"backtrack requires anchor, objective, and reason: {payload!r}")
        return NavigationModeDecision(
            action_mode="backtrack",
            stop_objective="",
            reasoning_action=objective,
            route_status="",
            action_objective=objective,
            action_reason=reason,
            backtrack_reason=reason,
            backtrack_anchor_node_id=anchor_node_id,
            backtrack_objective=objective,
            **progress_fields,
        )
    if selected_tool == "go_to_waypoint":
        expected_fields = (
            {"direction", "reason"}
            if system_owned_waypoint_objective
            else {"objective", "reason"}
        )
        if set(raw_tool) != expected_fields:
            raise ValueError(f"go_to_waypoint fields must be exactly {sorted(expected_fields)!r}: {payload!r}")
        objective = str(raw_tool.get("objective", "")).strip()
        direction = str(raw_tool.get("direction", "")).strip()
        reason = str(raw_tool.get("reason", "")).strip()
        if (
            reason == ""
            or (system_owned_waypoint_objective and direction not in VLN_HORIZONTAL_DIRECTIONS)
            or (not system_owned_waypoint_objective and objective == "")
        ):
            raise ValueError(f"go_to_waypoint requires its shown fields: {payload!r}")
        return NavigationModeDecision(
            action_mode="go_to_waypoint",
            stop_objective="",
            reasoning_action=reason if system_owned_waypoint_objective else objective,
            direction=direction,
            route_status="",
            action_objective=objective,
            action_reason=reason,
            **progress_fields,
        )
    if selected_tool == "approach_to_stop":
        movement = str(raw_tool.get("movement", "")).strip()
        expected_fields = {"movement", "stop_objective", "reason"}
        if system_owned_waypoint_objective and movement == "move":
            expected_fields.add("direction")
        if set(raw_tool) != expected_fields:
            raise ValueError(
                "approach_to_stop fields must be exactly "
                f"{sorted(expected_fields)!r}: {payload!r}"
            )
        direction = str(raw_tool.get("direction", "")).strip()
        stop_objective = str(raw_tool.get("stop_objective", "")).strip()
        reason = str(raw_tool.get("reason", "")).strip()
        if movement not in {"move", "stay"}:
            raise ValueError(
                f"approach_to_stop movement must be move or stay: {payload!r}"
            )
        if stop_objective == "" or reason == "":
            raise ValueError(
                f"approach_to_stop requires stop_objective and reason: {payload!r}"
            )
        if (
            system_owned_waypoint_objective
            and movement == "move"
            and direction not in VLN_HORIZONTAL_DIRECTIONS
        ):
            raise ValueError(
                f"moving approach_to_stop requires a horizontal direction: {payload!r}"
            )
        return NavigationModeDecision(
            action_mode="approach_to_stop",
            approach_movement=movement,
            stop_objective=stop_objective,
            reasoning_action=reason,
            direction=direction,
            route_status="",
            action_objective=stop_objective,
            action_reason=reason,
            **progress_fields,
        )
    if selected_tool == "vertical_transition":
        expected_fields = {"vertical_direction", "reason"}
        if set(raw_tool) != expected_fields:
            raise ValueError(f"vertical_transition fields must be exactly {sorted(expected_fields)!r}: {payload!r}")
        vertical_direction = str(raw_tool.get("vertical_direction", "")).strip()
        reason = str(raw_tool.get("reason", "")).strip()
        if vertical_direction not in {"up", "down"} or reason == "":
            raise ValueError(f"vertical_transition requires direction and reason: {payload!r}")
        return NavigationModeDecision(
            action_mode="vertical_transition",
            stop_objective="",
            reasoning_action=reason,
            vertical_direction=vertical_direction,
            route_status="",
            action_objective=f"Take the {vertical_direction} vertical transition.",
            action_reason=reason,
            **progress_fields,
        )
    raise ValueError(f"unsupported VLN progress-navigation tool: {payload!r}")


def _normalize_vln_tpu_tool_calls(
    payload: dict[str, object],
    *,
    require_retrieval_conclusion: bool,
) -> dict[str, object]:
    expected_top_fields = {"tool_calls"}
    if require_retrieval_conclusion:
        expected_top_fields.add("retrieval_conclusion")
    if set(payload) != expected_top_fields:
        raise ValueError(
            "Task Progress Updater returned unexpected top-level fields: "
            f"{payload!r}"
        )
    if require_retrieval_conclusion:
        if str(payload.get("retrieval_conclusion", "")).strip() == "":
            raise ValueError(
                "Task Progress Updater must conclude the latest retrieval "
                f"before its next tool: {payload!r}"
            )
    raw_calls = payload.get("tool_calls")
    if not isinstance(raw_calls, list) or not (1 <= len(raw_calls) <= 2):
        raise ValueError("tool_calls must contain one or two calls")
    calls: list[tuple[str, dict[str, object]]] = []
    for raw_call in raw_calls:
        if not isinstance(raw_call, dict) or set(raw_call) != {"name", "arguments"}:
            raise ValueError(f"invalid TPU tool call: {raw_call!r}")
        name = str(raw_call.get("name", "")).strip()
        arguments = raw_call.get("arguments")
        if not isinstance(arguments, dict):
            raise ValueError(f"TPU tool arguments must be an object: {raw_call!r}")
        calls.append((name, arguments))
    final_name, final_arguments = calls[-1]
    if final_name not in {"retrieve", "update_progress"}:
        raise ValueError(
            "the final TPU call must be exactly retrieve or update_progress"
        )
    condition_updates: list[object] = []
    if len(calls) == 2:
        condition_name, condition_arguments = calls[0]
        if condition_name != "update_progress_conditions":
            raise ValueError(
                "update_progress_conditions is the only optional first TPU call"
            )
        if set(condition_arguments) != {"condition_updates"}:
            raise ValueError(
                "update_progress_conditions arguments must contain exactly "
                "condition_updates"
            )
        raw_condition_updates = condition_arguments.get("condition_updates")
        if not isinstance(raw_condition_updates, list) or raw_condition_updates == []:
            raise ValueError(
                "update_progress_conditions requires non-empty condition_updates"
            )
        condition_updates = raw_condition_updates
    elif calls[0][0] == "update_progress_conditions":
        raise ValueError(
            "update_progress_conditions cannot be the final TPU call"
        )
    normalized: dict[str, object] = {}
    if require_retrieval_conclusion:
        normalized["retrieval_conclusion"] = str(
            payload.get("retrieval_conclusion", "")
        ).strip()
    normalized["progress_condition_updates"] = condition_updates
    normalized[final_name] = final_arguments
    return normalized


































def decide_visual_action_point(
    *,
    client: "LLMClient",
    cache: "RuntimeCache",
    goal_text: str,
    selected_view: VisualViewContext,
    waypoint_target: str,
    revision_feedback: str = "",
    rejected_point_2d: tuple[float, float] | None = None,
    include_graph_context: bool = True,
) -> VisualActionPointDecision:
    del goal_text
    feedback = str(revision_feedback).strip()
    revision_section = ""
    if feedback:
        rejected_point = (
            ""
            if rejected_point_2d is None
            else "\nPrevious rejected point_2d: "
            f"[{float(rejected_point_2d[0]):.4f}, "
            f"{float(rejected_point_2d[1]):.4f}]"
        )
        revision_section = (
            "Revision feedback from the previous rejected point:\n"
            f"{feedback}{rejected_point}\n"
            "Avoid repeating the rejected image region."
        )
    overlay_rule = (
        "If a node-label overlay is visible, choose the actual reachable surface, "
        "not the drawn label."
        if include_graph_context
        else ""
    )
    system_prompt = (
        "You are the Visual Action module of a navigation agent. Choose one "
        "reachable waypoint in the supplied RGB image. Return JSON only."
    )
    text = f"""
Waypoint target:
{waypoint_target}

Select exactly one normalized image point `point_2d: [x, y]`, with values in
[0, 1] from image top-left to bottom-right.

Rules:
- Choose reachable floor, stair tread or landing, doorway floor, corridor floor,
  or safe free space.
- Do not choose walls, ceilings, objects, clutter, windows, mirrors, or door leaves.
- If no safe waypoint is visible, return a non-empty failure_reason.
- {overlay_rule}
{revision_section}

Return JSON only:
{{
  "reasoning": "<brief reachability explanation>",
  "point_2d": [0.0, 0.0],
  "target": "<intended waypoint>",
  "failure_reason": ""
}}
""".strip()
    content = [
        {"type": "text", "text": text},
        {"type": "text", "text": selected_view_prompt_text(selected_view)},
        image_content_for_view(cache=cache, view=selected_view),
    ]
    return normalize_visual_action_point(
        client.decide_visual_action(system_prompt, content)
    )


def decide_vertical_transition_visual_action_point(
    *,
    client: "LLMClient",
    cache: "RuntimeCache",
    selected_view: VisualViewContext,
    waypoint_target: str,
    revision_feedback: str = "",
    rejected_point_2d: tuple[float, float] | None = None,
) -> VisualActionPointDecision:
    feedback_text = str(revision_feedback).strip()
    revision_section = ""
    if feedback_text != "":
        rejected_point_text = (
            ""
            if rejected_point_2d is None
            else (
                "\nRejected point_2d: "
                f"[{float(rejected_point_2d[0]):.4f}, "
                f"{float(rejected_point_2d[1]):.4f}]"
            )
        )
        revision_section = f"""
Revision feedback:
{feedback_text}{rejected_point_text}
""".strip()
    system_prompt = """
You are the Vertical Transition Visual Action Grounder in NavClaw.
Ground the fixed waypoint target in the supplied RGB view by either selecting one normalized image point on the intended reachable walking surface or explicitly reporting that no valid point is available.
Do not change the waypoint target or decide whether the overall floor transition is complete.
Return only valid JSON matching the provided output contract.
""".strip()
    text = f"""
Waypoint target:
{waypoint_target}

Decision objective:
Ground the fixed waypoint target in this RGB view.

Point semantics:
- `point_2d` uses normalized image coordinates `[x, y]`, with `[0,0]` at the top-left and `[1,1]` at the bottom-right.
- A successful point lies on the walking surface that realizes the waypoint target, not on a referenced object, door leaf, railing, wall, or overlay.
- Revision feedback rejects the previous point or image region for this target.
- A corrected region or image-local direction in revision feedback applies only when it remains consistent with the fixed waypoint target.

Grounding rules:
- Select reachable floor, stair tread or landing, doorway floor, corridor floor, or other safe walking surface that matches the target.
- Reject walls, ceilings, furniture or object surfaces, clutter, windows, mirrors, railings, and door leaves.
- For an object or fixture reference, ground to nearby reachable floor.
- For a doorway or corridor target, ground to the open walking passage.
- Report failure when no safe matching region is visible.
""".strip()
    output_text = """
Output contract:

Success:
{
  "status": "success",
  "reasoning": "<brief visible evidence that the point safely grounds the fixed target>",
  "point_2d": [0.42, 0.81],
  "target": "<grounded target region>",
  "failure_reason": ""
}

Failure:
{
  "status": "failure",
  "reasoning": "<brief visible evidence that no safe matching point is available>",
  "point_2d": null,
  "target": "",
  "failure_reason": "<why no valid point can be selected>"
}
""".strip()
    content: list[dict[str, object]] = [{"type": "text", "text": text}]
    if revision_section != "":
        content.append({"type": "text", "text": revision_section})
    content.extend(
        [
            {
                "type": "text",
                "text": (
                    "Selected RGB image for "
                    f"{direction_for_angle(selected_view.angle_deg)}:"
                ),
            },
            image_content_for_view(cache=cache, view=selected_view),
            {"type": "text", "text": output_text},
        ]
    )
    parsed = client.decide_visual_action(system_prompt, content)
    return normalize_vertical_transition_visual_action_point(parsed)


def verify_vertical_transition_waypoint(
    *,
    client: "LLMClient",
    cache: "RuntimeCache",
    instruction: str,
    selected_view: VisualViewContext,
    waypoint_target: str,
    visual_action: VisualActionPointDecision,
    geometry_warning: str = "",
    allow_revise_verdict: bool = True,
) -> VisualWaypointVerificationDecision:
    if visual_action.point_2d is None:
        raise ValueError("vertical transition waypoint verification requires visual_action.point_2d")
    geometry_warning_text = str(geometry_warning).strip()
    geometry_warning_section = ""
    if geometry_warning_text != "":
        geometry_warning_section = f"""

Geometry evidence:
{geometry_warning_text}
""".rstrip()
    verdicts = ["execute"]
    if bool(allow_revise_verdict):
        verdicts.append("revise_same_view")
        revise_rule = """
- Return "revise_same_view" if the red dot is reachable but not a good match to the waypoint target, and a better matching point is visible in the same image.
- Return "revise_same_view" if the selected view contains a better reachable waypoint for the requested vertical direction.
""".strip()
        critique_rule = """
- If you return "revise_same_view", critique must be exactly two short sentences.
- The first sentence says why the current red dot is invalid.
- The second sentence names the corrected requested-direction reachable region using visible anchors.
""".strip()
    else:
        revise_rule = """
- Do not return "revise_same_view"; this is the final validation attempt for this waypoint target.
- If the red dot does not match the waypoint target or requested vertical direction, return "fallback_navigation".
""".strip()
        critique_rule = ""
    verdicts.extend(["fallback_navigation", "fail"])
    verdict_values = "|".join(verdicts)
    observation = cache.get_observation(str(selected_view.obs_id)).observation
    rgb = np.asarray(observation.rgb)
    resized = resize_rgb_to_fit(rgb, max_size=LLM_CAMERA_IMAGE_MAX_SIZE).image
    height, width = int(resized.shape[0]), int(resized.shape[1])
    point_pixel = normalized_point_to_pixel(
        visual_action.point_2d,
        image_width=width,
        image_height=height,
    )
    overlay = _draw_vt_waypoint_red_dot_rgb(
        resized,
        point_pixel=point_pixel,
    )
    system_prompt = """
You are the Vertical Transition Waypoint Verifier in NavClaw.
Validate whether the marked image point is a reachable next local movement point that matches the fixed waypoint target and requested vertical direction.
Do not decide whether the overall floor transition is complete.
Return only valid JSON matching the provided output contract.
""".strip()
    text = f"""
Requested transition:
{instruction}

Fixed waypoint target:
{waypoint_target}
{geometry_warning_section}

Verdict semantics:
- `execute`: the red dot matches the waypoint target, lies on reachable walking surface, and is a valid next local move for the requested transition.
- `revise_same_view`: this view contains a better reachable point for the same target and vertical direction.
- `fallback_navigation`: this view or target does not admit a valid point for the requested transition; return control to the Vertical Transition Step Planner.
- `fail`: the supplied image or overlay is unusable for validation.

Validation rules:
- A reachable point is insufficient when it does not match the waypoint target.
- The point must lie on walking surface, not on a wall, furniture, object, railing, door leaf, window, mirror, or clutter.
- For a destination-floor target, the point must be on the first safe floor region immediately beyond the final tread rather than farther along the corridor or room.
- Do not predict whether executing this point completes the floor transition; the Step Planner assesses completion from the next fresh observation.
{revise_rule}
{critique_rule}
- For verdicts other than `revise_same_view`, `critique` is one concise evidence-based sentence.
""".strip()
    output_text = f"""
Output contract:
{{
  "critique": "<concise evidence; two short sentences for revise_same_view>",
  "verdict": "{verdict_values}"
}}
""".strip()
    content = [
        {"type": "text", "text": text},
        {
            "type": "text",
            "text": (
                "Overlay image for stair-transition waypoint validation. The red dot is the selected "
                "waypoint; judge whether the red dot matches the waypoint target "
                "and is executable for the requested direction."
            ),
        },
        image_content_for_array(overlay),
        {"type": "text", "text": output_text},
    ]
    parsed = client.verify_vertical_transition_waypoint(system_prompt, content)
    return normalize_visual_waypoint_verification(parsed)


def _draw_vt_waypoint_red_dot_rgb(
    image: np.ndarray,
    *,
    point_pixel: tuple[float, float],
) -> np.ndarray:
    canvas = np.asarray(image, dtype=np.uint8).copy()
    height, width = int(canvas.shape[0]), int(canvas.shape[1])
    cx = int(round(float(point_pixel[0])))
    cy = int(round(float(point_pixel[1])))
    cx = max(0, min(width - 1, cx))
    cy = max(0, min(height - 1, cy))
    radius = max(5, int(round(float(min(width, height)) * 0.018)))
    outline_radius = radius + 2
    yy, xx = np.ogrid[:height, :width]
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
    canvas[dist2 <= outline_radius**2] = np.array([0, 0, 0], dtype=np.uint8)
    canvas[dist2 <= radius**2] = np.array([255, 0, 0], dtype=np.uint8)
    return canvas




def decide_stop_confirmation(
    *,
    client: "LLMClient",
    cache: "RuntimeCache",
    goal_text: str,
    goal_kind: str = "",
    visual_context: VisualActionContext,
    pending_stop: dict[str, object],
) -> StopConfirmationDecision:
    if str(goal_kind).strip() != GOAL_KIND_VLN_INSTRUCTION:
        raise ValueError("stop confirmation supports VLN instructions only")
    system_prompt = """
You are the Stop Confirmation module of an embodied navigation agent.
You decide whether the current endpoint satisfies the terminal stop condition.
Use only the supplied endpoint evidence.
Return JSON only.
""".strip()
    terminal_objective = str(pending_stop.get("stop_objective", "")).strip()
    text = f"""
Navigation task:
{goal_text}

Terminal stop objective:
{terminal_objective}

Decision rules:
- Treat the pending stop request as the previous module's intent, not as proof that the terminal stop condition is satisfied.
- Return "stop" only when the current endpoint satisfies the final stop condition in the navigation task.
- For route-following tasks, the ordered route requirements must be complete before stopping.
- Use the endpoint image, fresh panorama, and stop-approach movement RGB sequence as evidence.
- Return "continue" when one more meaningful move is still needed; continue_objective should state that next move.
""".strip()
    output_text = """
Return JSON only:
{
  "reasoning": "<brief explanation based on the fresh panorama>",
  "decision": "stop|continue",
  "continue_objective": "<required only when decision is continue>"
}
"""
    output_text = output_text.strip()
    content = [{"type": "text", "text": text}]
    waypoint_overlay_content = image_content_for_stop_waypoint_overlay(
        cache=cache,
        pending_stop=pending_stop,
    )
    if waypoint_overlay_content is not None:
        content.append(
            {
                "type": "text",
                "text": (
                    "Endpoint image from the stop-approach move:\n"
                    "- The red dot marks the selected waypoint.\n"
                    "- The move to this waypoint succeeded, so treat the red-dot waypoint "
                    "as the agent's current physical endpoint for this confirmation."
                ),
            }
        )
        content.append(waypoint_overlay_content)
    content.append(
        {
            "type": "text",
            "text": "Fresh panorama after the stop-approach move.",
        }
    )
    content.append(
        {
            "type": "text",
            "text": current_panorama_prompt_text(
                visual_context,
                include_visited_nodes=False,
            ),
        }
    )
    content.extend(
        image_content_for_current_panorama_views(
            cache=cache,
            views=visual_context.views,
            include_visited_nodes=False,
        )
    )
    raw_movement_obs_ids = pending_stop.get("rgb_history_obs_ids")
    if not isinstance(raw_movement_obs_ids, (list, tuple)):
        raw_movement_obs_ids = visual_context.recent_edge_rgb_history_obs_ids
    movement_obs_ids = [str(obs_id) for obs_id in list(raw_movement_obs_ids)]
    movement_history_content = image_content_for_movement_history_sheet(
        cache=cache,
        obs_ids=movement_obs_ids,
    )
    if movement_history_content is not None:
        content.append(
            {
                "type": "text",
                "text": (
                    "Last movement RGB sequence from the stop-approach move, "
                    "ordered left-to-right/top-to-bottom by time."
                ),
            }
        )
        content.append(movement_history_content)
    content.append({"type": "text", "text": output_text})
    parsed = client.decide_stop_confirmation(system_prompt, content)
    return normalize_stop_confirmation(parsed)
