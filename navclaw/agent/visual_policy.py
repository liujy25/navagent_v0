from __future__ import annotations

from copy import deepcopy
import json
from typing import TYPE_CHECKING

import numpy as np

from navclaw.agent.episodic_retrieval import RetrieveRequest
from navclaw.agent.episodic_retrieval import normalize_retrieve_request
from navclaw.agent.visual_action_context import VisualActionContext, VisualViewContext
from navclaw.agent.visual_policy_decisions import NavigationModeDecision
from navclaw.agent.visual_policy_decisions import StopConfirmationDecision
from navclaw.agent.visual_policy_decisions import TaskProgressDecision
from navclaw.agent.visual_policy_decisions import VisualActionPointDecision
from navclaw.agent.visual_policy_decisions import VisualWaypointVerificationDecision
from navclaw.agent.visual_policy_decisions import normalize_stop_confirmation
from navclaw.agent.visual_policy_decisions import normalize_visual_action_point
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
    system_prompt = """
You are a task progress planner for an embodied navigation agent.
You create concise task progress memory that helps the agent make stable navigation decisions.
Return JSON only.
""".strip()
    normalized_task_type = str(task_type).strip()
    if normalized_task_type not in {"", GOAL_KIND_VLN_INSTRUCTION}:
        raise ValueError("task progress supports VLN instructions only")
    user_prompt = f"""
Navigation task:
{goal_text}

Create ordered progress items for this VLN instruction.

Rules:
- Each item should contain one action-driving route constraint, one route progress stage, or one final stopping condition from the instruction.
- Preserve ordered route constraints such as "past X", "left/right of Y", "first door", "exit", "enter", "wait", and "stop".
- If satisfying one constraint is a prerequisite before a later landmark, opening, room, or stop target should guide waypoint choice, split them into separate ordered items.
- Keep descriptive target modifiers together when they identify the same target, such as "the door near X" or "the chair beside Y"; split only when the instruction requires completing one route constraint before pursuing the next.
- Because the agent receives an all-around panorama, do not create pure orientation progress items such as "turn around", "look left", or "face X".
- Use status "active" for every initial progress item.
- Use result "" for active progress items.

Return JSON only:
{{
  "progress_items": [
    {{
      "content": "<route progress item>",
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
        # Legacy input compatibility for older LLM traces or cached responses.
        raw_items = parsed.get("progress_items", parsed.get("items", parsed.get("todos", [])))
        if not isinstance(raw_items, list) or raw_items == []:
            raise ValueError(f"visual task progress generator returned no progress_items: {parsed!r}")
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


def _vln_progress_update_rules(node_binding_rule: str) -> str:
    return f"""
Progress update rules:
- Review ordered progress items and preserve explicit prerequisites.
{node_binding_rule}
- Create and maintain progress conditions as independently verifiable atomic conditions required by their parent item, using the current observation and stored retrieval conclusions as evidence. `unconfirmed` means a condition is not yet confirmed; `confirmed` means evidence confirms that exact condition.
- Represent target visibility and the instructed agent-target relation as separate conditions. Visibility may confirm a visibility condition, but the instructed relation requires executed evidence.
- Progress conditions persist evidence-backed partial completion across navigation steps. `progress_condition_updates` is independent of item updates and may accompany either RETRIEVE or UPDATE_PROGRESS. With RETRIEVE, it records only evidence already available before the newly requested evidence is returned.
- Set each item's `status` from the complete executed evidence for the item: `active` means its instructed state is not yet complete, and `done` means the complete instructed state is confirmed. Progress conditions are supporting reference memory; they need not enumerate every condition and do not determine the item status.
- Each emitted item or condition update starts with a concise `thought` explaining why that operation is supported.
- Leave `progress_updates` empty when no item memory change is supported. Otherwise each entry uses the fields for its operation:
  - update: `thought,op,index,status,result`; it preserves existing task content.
  - rewrite: `thought,op,index,content,status,result`.
  - add: `thought,op,content,status,result`.
  - insert: `thought,op,index,content,status,result`.
  - remove: `thought,op,index`.
- Every non-remove item update includes `result`: use an empty string for `active` and a non-empty completion summary for `done`.
- Leave `progress_condition_updates` empty when no condition change is supported. Every entry identifies its parent with `item_index` and uses one operation: add=`thought,item_index,op,content,status`; update=`thought,item_index,op,condition_id,status`; rewrite=`thought,item_index,op,condition_id,content,status`; remove=`thought,item_index,op,condition_id`. In UPDATE_PROGRESS, `item_index` refers to the task memory after `progress_updates` are applied.
- For multi-floor instructions, a stair landing or mid-stair platform is not a completed floor transition.
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
    progress_context_text: str = "",
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
        progress_context_text=progress_context_text,
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
        navigation_replan_feedback_text=navigation_replan_feedback_text,
        landmark_panorama_views=landmark_panorama_views,
        detected_landmarks_text=detected_landmarks_text,
        allowed_backtrack_node_ids=allowed_backtrack_node_ids,
        require_backtrack=require_backtrack,
        system_owned_waypoint_objective=system_owned_waypoint_objective,
        task_progress_bev_overlay=task_progress_bev_overlay,
        planning_reference_panorama=planning_reference_panorama,
    )
    if not isinstance(decision, NavigationModeDecision):
        raise TypeError("Navigation Planner returned a task-progress decision")
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
    latest_task_progress: TaskProgressDecision | None = None,
    progress_context_text: str = "",
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
        raise ValueError("required BACKTRACK needs at least one available anchor")
    terminal_check_required = isinstance(terminal_check_context, dict)
    if terminal_check_required and not allow_update_progress:
        raise ValueError("post-approach terminal check requires UPDATE_PROGRESS")
    tool_schemas: list[str] = []
    conclusion_prefix = (
        '"retrieval_conclusion":"<direct answer to the latest retrieval query; confirmed and missing evidence>",'
        if require_retrieval_conclusion
        else ""
    )
    progress_condition_field = '"progress_condition_updates":[],'
    if allow_retrieve:
        tool_schemas.append(
            "RETRIEVE:\n"
            "{"
            f"{conclusion_prefix}"
            f"{progress_condition_field}"
            '"retrieve":{"query":"<question to answer from stored episode evidence>",'
            '"items":[{"ref":"<memory index ref>","fields":["<available field>"]}]}}'
        )
    if allow_update_progress:
        terminal_check_schema = (
            ',"terminal_check":{'
            '"decision":"done|continue",'
            '"reasoning":"<why the reached endpoint completes the task or what remains>",'
            '"missing_constraints":["<unconfirmed instruction constraint>"]}'
            if terminal_check_required
            else ""
        )
        tool_schemas.append(
            "UPDATE_PROGRESS:\n"
            "{"
            f"{conclusion_prefix}"
            f"{progress_condition_field}"
            '"update_progress":{'
            '"progress_updates":[]'
            f"{terminal_check_schema}" "}}"
        )
    if allow_navigation_actions:
        if allowed_backtrack_node_ids:
            allowed_backtrack_refs = ", ".join(
                sorted(str(node_id) for node_id in allowed_backtrack_node_ids)
            )
            tool_schemas.append(
                "BACKTRACK:\n"
                f'{{"backtrack":{{"anchor_node_id":"<one of: {allowed_backtrack_refs}>",'
                '"objective":"<what route choice to attempt from that node>",'
                '"reason":"<why replanning from that reference node is needed>"}}'
            )
        if not require_backtrack:
            go_to_waypoint_schema = (
                'GO_TO_WAYPOINT:\n{"go_to_waypoint":{"reason":"<the concrete visible region or '
                'relative route direction to enter next, and why it advances the current navigation intent>"}}'
                if system_owned_waypoint_objective
                else (
                    "GO_TO_WAYPOINT:\n"
                    '{"go_to_waypoint":{"objective":"<same-floor navigation objective>",'
                    '"reason":"<why this is the next action>"}}'
                )
            )
            tool_schemas.extend(
                [
                    go_to_waypoint_schema,
                    "APPROACH_TO_STOP:\n"
                    '{"approach_to_stop":{"movement":"move|stay",'
                    '"stop_objective":"<final locally reachable stopping region>",'
                    '"reason":"<why this stopping region should be approached or retained>"}}',
                    "VERTICAL_TRANSITION:\n"
                    '{"vertical_transition":{"vertical_direction":"up|down",'
                    '"reason":"<where the stairs are, why they are accessible from the current position, '
                    'and why taking them up or down is the required next route step>"}}',
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

Episode memory index (text-only lookup):
{memory_index_text if str(memory_index_text).strip() != "" else "none"}

Retrieval budget for this navigation step:
- maximum rounds: {int(retrieve_max_rounds)}
- completed rounds: {int(retrieve_completed_rounds)}
- remaining rounds: {int(retrieve_remaining)}
{single_retrieval_instruction}
""".rstrip()
        retrieval_semantics = """
RETRIEVE tool:
- The memory index is a compact text overview of the episode history.
- Use RETRIEVE to inspect selected visual and movement evidence from that history.
- Based on the evidence obtained so far, each subsequent RETRIEVE query must target the historical evidence still needed to determine or justify the current task-progress update.
- `provided_fields` are already attached in the current planning observation; `available_fields` can be loaded with RETRIEVE.
- Landmark refs use landmark_<global index>; landmark boxes display only the numeric suffix.
- node.rgb is a stored node panorama; node.landmarks are detections on that panorama.
- edge.rgb is the start-node view toward the endpoint, with the executed path and endpoint node overlaid.
- edge.trajectory is the executed trajectory on the floor BEV; edge.movement_rgb is chronological traversal RGB.
- landmark.rgb is stored detection RGB with its bounding box.
""".strip()
    progress_context = str(progress_context_text).strip()
    progress_context_section = (
        f"\n\nCurrent-step context:\n{progress_context}"
        if progress_context != ""
        else ""
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
                    f"terminal_reasoning: {latest_task_progress.terminal_check_reasoning}",
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
    landmark_section = f"\n\n{landmark_text}" if landmark_text != "" else ""
    conclusion_rules = (
        """
Latest retrieval conclusion:
- Write `retrieval_conclusion` before the selected tool.
- Directly answer the latest retrieval query, distinguishing confirmed facts from missing evidence and naming the relevant refs concisely.
- Previous retrieval conclusions and the cumulative BEV remain available; raw evidence is attached only for the latest unresolved retrieval.
""".strip()
        if require_retrieval_conclusion
        else ""
    )
    navigation_action_rules = (
        f"""
Navigation action tools:
{("- BACKTRACK switches the current planning node to a prior node. Progress, retrieval, and navigation then continue from that node's stored context; the resulting action is executed from the physical robot pose." if allowed_backtrack_node_ids else "")}
{("- The current planning node has no local exploration frontier. Select BACKTRACK to one of the available anchors." if require_backtrack else "- GO_TO_WAYPOINT continues same-floor waypoint planning from the current node.")}
{("" if require_backtrack else "- APPROACH_TO_STOP prepares a locally reachable stopping pose: use `stay` when the current pose is that stopping region, or `move` when one local approach is needed.")}
{("" if require_backtrack else "- VERTICAL_TRANSITION is used when the required upstairs or downstairs route is immediately visible and accessible from the current position.")}
""".strip()
        if allow_navigation_actions
        else ""
    )
    node_binding_rule = (
        "- Existing `node span: <start> -> <completion or pending>` annotations identify where "
        "an item first became the current active item and where completion was first confirmed. The system maintains "
        "these anchors; do not copy the annotation into task content."
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
            "The APPROACH_TO_STOP decision retained the current pose."
            if approach_movement == "stay"
            else "The latest APPROACH_TO_STOP movement reached the current pose."
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
You are the Progress-Conditioned Navigation Planner of an embodied navigation agent.
Use the current observation, updated task-progress state, and retrieval conclusions to select the next high-level navigation action.
Return JSON only.
""".strip()
        if allow_navigation_actions
        else """
You are the Task Progress Updater of an embodied navigation agent.
Use the current observation, task-progress memory, and retrieved episode evidence to resolve progress-relevant evidence gaps and update task progress.
Return JSON only.
""".strip()
    )
    context_text = "\n\n".join(
        section
        for section in (
            "Task progress memory:\n"
            + task_progress.format_for_prompt(
                include_node_bindings=visual_context.graph_context_visible,
                show_empty_progress_conditions=True,
            ),
            progress_context_section.strip(),
            replan_feedback_section.strip(),
            terminal_check_section.strip(),
            latest_progress_section.strip(),
        )
        if section != ""
    )
    decision_text = "\n\n".join(
        section
        for section in (
            retrieval_semantics,
            _vln_progress_update_rules(node_binding_rule) if allow_update_progress else "",
            conclusion_rules,
            navigation_action_rules,
            (
                "Return exactly one of the following JSON objects. "
                "Use exactly the shown fields:\n\n"
                + chr(10).join(tool_schemas)
            ),
        )
        if section != ""
    )
    content: list[dict[str, object]] = [
        {"type": "text", "text": context_text}
    ]
    if retrieval_catalog_section != "":
        content.append(
            {"type": "text", "text": retrieval_catalog_section.strip()}
        )
    if landmark_section != "":
        content.append({"type": "text", "text": landmark_section.strip()})
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
                        "Latest APPROACH_TO_STOP movement RGB, ordered "
                        "left-to-right/top-to-bottom by time."
                    ),
                }
            )
            content.append(movement_history_content)
    content.extend(deepcopy(retrieval_workspace_content))
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
                        "Navigation Planner requires current task progress"
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
                "Previous invalid response:\n"
                f"{json.dumps(parsed, ensure_ascii=False, indent=2)}\n\n"
                "Validation error:\n"
                f"{exc}\n\n"
                "Return one corrected complete JSON response using the same "
                "output schema."
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
        raw_id_field = (
            "condition_id"
            if "condition_id" in raw_update
            else "knowledge_id"
            if "knowledge_id" in raw_update
            else "condition_id"
        )
        expected_fields = {
            "add": {"thought", "item_index", "op", "content", "status"},
            "update": {"thought", "item_index", "op", raw_id_field, "status"},
            "rewrite": {
                "thought",
                "item_index",
                "op",
                raw_id_field,
                "content",
                "status",
            },
            "remove": {"thought", "item_index", "op", raw_id_field},
        }.get(op)
        if expected_fields is None or set(raw_update) != expected_fields:
            raise ValueError(f"invalid progress condition update fields: {raw_update!r}")
        thought = str(raw_update.get("thought", "")).strip()
        if thought == "":
            raise ValueError(f"progress condition update requires thought: {raw_update!r}")
        if next(iter(raw_update), None) != "thought":
            raise ValueError(
                f"progress condition update must write thought first: {raw_update!r}"
            )
        item_index = raw_update.get("item_index")
        if not isinstance(item_index, int) or isinstance(item_index, bool):
            raise ValueError(
                f"progress condition item_index must be an integer: {raw_update!r}"
            )
        item: dict[str, object] = {
            "thought": thought,
            "item_index": int(item_index),
            "op": op,
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
        normalized.append(item)
    return normalized


def _normalize_vln_progress_updates(raw_updates: object) -> list[dict[str, object]]:
    if not isinstance(raw_updates, list):
        raise ValueError("UPDATE_PROGRESS progress_updates must be a list")
    normalized: list[dict[str, object]] = []
    for raw_update in raw_updates:
        if not isinstance(raw_update, dict):
            raise ValueError(f"progress update must be an object: {raw_update!r}")
        op = str(raw_update.get("op", "")).strip().lower()
        if op not in {"update", "rewrite", "add", "insert", "remove"}:
            raise ValueError(f"progress update has unsupported op: {raw_update!r}")
        if str(raw_update.get("thought", "")).strip() == "":
            raise ValueError(f"progress update requires thought: {raw_update!r}")
        if next(iter(raw_update), None) != "thought":
            raise ValueError(f"progress update must write thought first: {raw_update!r}")
        expected_fields = {"thought", "op"}
        if op in {"update", "rewrite", "insert", "remove"}:
            expected_fields.add("index")
        if op == "rewrite":
            expected_fields.add("content")
        if op != "remove":
            expected_fields.update({"status", "result"})
        if op in {"add", "insert"}:
            expected_fields.add("content")
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
        if "content" in expected_fields and str(raw_update.get("content", "")).strip() == "":
            raise ValueError(f"progress update requires content: {raw_update!r}")
        item = dict(raw_update)
        if "status" in expected_fields:
            status = str(raw_update.get("status", "")).strip().lower()
            if status not in {"active", "done"}:
                raise ValueError(f"invalid progress item status: {raw_update!r}")
            item["status"] = status
        if op != "remove":
            if item["status"] == "done":
                if str(raw_update.get("result", "")).strip() == "":
                    raise ValueError(f"done progress update requires result: {raw_update!r}")
                item["result"] = str(raw_update.get("result", "")).strip()
            elif str(raw_update.get("result", "")).strip() != "":
                raise ValueError(f"active progress update requires empty result: {raw_update!r}")
            else:
                item["result"] = ""
        normalized.append(item)
    return normalized


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
        "Navigation Planner"
        if allow_navigation_actions
        else "Task Progress Updater"
    )
    module_tools = (
        _VLN_NAVIGATION_TOOLS
        if allow_navigation_actions
        else _VLN_TASK_PROGRESS_TOOLS
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
            f"only BACKTRACK is available in this state: {payload!r}"
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
    if require_retrieval_conclusion and next(iter(payload)) != "retrieval_conclusion":
        raise ValueError(
            "Task Progress Updater must write retrieval_conclusion before "
            f"its next tool: {payload!r}"
        )
    retrieval_conclusion = str(payload.get("retrieval_conclusion", "")).strip()
    if require_retrieval_conclusion and retrieval_conclusion == "":
        raise ValueError(
            "Task Progress Updater must conclude the latest retrieval "
            f"before its next tool: {payload!r}"
        )

    if selected_tool == "retrieve":
        if not allow_retrieve:
            raise ValueError(f"RETRIEVE is not available in this state: {payload!r}")
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
            raise ValueError(f"UPDATE_PROGRESS is not available in this state: {payload!r}")
        expected_fields = {"progress_updates"}
        if require_terminal_check:
            expected_fields.add("terminal_check")
        if set(raw_tool) != expected_fields:
            raise ValueError(
                "UPDATE_PROGRESS fields must be exactly "
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
            expected_terminal_fields = {
                "decision",
                "reasoning",
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
                raw_terminal_check.get("reasoning", "")
            ).strip()
            raw_missing_constraints = raw_terminal_check.get("missing_constraints")
            if terminal_check_decision not in {"done", "continue"}:
                raise ValueError(f"terminal_check decision must be done or continue: {payload!r}")
            if terminal_check_reasoning == "":
                raise ValueError(f"terminal_check requires reasoning: {payload!r}")
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
        raise ValueError(f"{selected_tool} is not available before UPDATE_PROGRESS: {payload!r}")
    progress_fields = {
        "progress_analysis": "",
        "progress_reasoning": "",
        "progress_updates": [],
    }
    if selected_tool == "backtrack":
        expected_fields = {"anchor_node_id", "objective", "reason"}
        if set(raw_tool) != expected_fields:
            raise ValueError(f"BACKTRACK fields must be exactly {sorted(expected_fields)!r}: {payload!r}")
        anchor_node_id = str(raw_tool.get("anchor_node_id", "")).strip()
        objective = str(raw_tool.get("objective", "")).strip()
        reason = str(raw_tool.get("reason", "")).strip()
        if allowed_backtrack_node_ids is not None and anchor_node_id not in allowed_backtrack_node_ids:
            raise ValueError(f"BACKTRACK selected unavailable anchor node: {payload!r}")
        if anchor_node_id == "" or objective == "" or reason == "":
            raise ValueError(f"BACKTRACK requires anchor, objective, and reason: {payload!r}")
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
            {"reason"}
            if system_owned_waypoint_objective
            else {"objective", "reason"}
        )
        if set(raw_tool) != expected_fields:
            raise ValueError(f"GO_TO_WAYPOINT fields must be exactly {sorted(expected_fields)!r}: {payload!r}")
        objective = str(raw_tool.get("objective", "")).strip()
        reason = str(raw_tool.get("reason", "")).strip()
        if reason == "" or (not system_owned_waypoint_objective and objective == ""):
            raise ValueError(f"GO_TO_WAYPOINT requires its shown fields: {payload!r}")
        return NavigationModeDecision(
            action_mode="go_to_waypoint",
            stop_objective="",
            reasoning_action=reason if system_owned_waypoint_objective else objective,
            route_status="",
            action_objective=objective,
            action_reason=reason,
            **progress_fields,
        )
    if selected_tool == "approach_to_stop":
        expected_fields = {"movement", "stop_objective", "reason"}
        if set(raw_tool) != expected_fields:
            raise ValueError(
                "APPROACH_TO_STOP fields must be exactly "
                f"{sorted(expected_fields)!r}: {payload!r}"
            )
        movement = str(raw_tool.get("movement", "")).strip()
        stop_objective = str(raw_tool.get("stop_objective", "")).strip()
        reason = str(raw_tool.get("reason", "")).strip()
        if movement not in {"move", "stay"}:
            raise ValueError(
                f"APPROACH_TO_STOP movement must be move or stay: {payload!r}"
            )
        if stop_objective == "" or reason == "":
            raise ValueError(
                f"APPROACH_TO_STOP requires stop_objective and reason: {payload!r}"
            )
        return NavigationModeDecision(
            action_mode="approach_to_stop",
            approach_movement=movement,
            stop_objective=stop_objective,
            reasoning_action=reason,
            route_status="",
            action_objective=stop_objective,
            action_reason=reason,
            **progress_fields,
        )
    if selected_tool == "vertical_transition":
        expected_fields = {"vertical_direction", "reason"}
        if set(raw_tool) != expected_fields:
            raise ValueError(f"VERTICAL_TRANSITION fields must be exactly {sorted(expected_fields)!r}: {payload!r}")
        vertical_direction = str(raw_tool.get("vertical_direction", "")).strip()
        reason = str(raw_tool.get("reason", "")).strip()
        if vertical_direction not in {"up", "down"} or reason == "":
            raise ValueError(f"VERTICAL_TRANSITION requires direction and reason: {payload!r}")
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
You judge one marked local waypoint for a stair transition.
The agent wants to go upstairs or downstairs.
Decide whether the red dot matches the waypoint target and is a valid next movement point.
Return JSON only.
""".strip()
    text = f"""
Requested transition:
{instruction}

Waypoint target:
{waypoint_target}
{geometry_warning_section}

Rules:
- The red dot marks the selected waypoint.
- The red dot must match the waypoint target; a reachable but mismatched floor point is not enough for "execute".
- Return "execute" only if the red dot matches the waypoint target, is reachable, and moving to it is a valid next step for the requested up/down stair transition.
- On the next-floor surface, the red dot must be on the first stable floor immediately beyond the final tread, not farther along the corridor or room.
{revise_rule}
- Return "fallback_navigation" if the red dot is in the wrong vertical direction or this view does not provide a valid next point.
- Return "fail" only if the image is unusable.
- A valid point must be on reachable walking surface, not on walls, furniture, railings, doors, windows, or clutter.
{critique_rule}
- If you return "execute", critique must briefly state why the red dot matches the waypoint target and requested vertical direction.
- If you return "fallback_navigation" or "fail", critique must briefly state the blocking reason.
""".strip()
    output_text = f"""
Return JSON only:
{{
  "critique": "<brief reason; for revise_same_view use two short sentences>",
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
