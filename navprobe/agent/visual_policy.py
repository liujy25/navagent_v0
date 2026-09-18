from __future__ import annotations

from copy import deepcopy
import json
from typing import TYPE_CHECKING

import numpy as np

from navprobe.agent.episodic_retrieval import RetrieveRequest
from navprobe.agent.episodic_retrieval import normalize_retrieve_request
from navprobe.agent.visual_action_context import VisualActionContext
from navprobe.agent.visual_policy_decisions import NavigationModeDecision
from navprobe.agent.visual_policy_decisions import TaskProgressDecision
from navprobe.agent.visual_policy_prompt_images import current_panorama_prompt_text
from navprobe.agent.visual_policy_prompt_images import image_content_for_array
from navprobe.agent.visual_policy_prompt_images import image_content_for_movement_history_sheet
from navprobe.agent.visual_policy_prompt_images import image_content_for_current_panorama_views
from navprobe.memory.task_progress import TaskProgressItem, TaskProgressMemory
from navprobe.perception.goal_identity import GOAL_KIND_OBJECT_CATEGORY
from navprobe.perception.goal_identity import GOAL_KIND_VLN_INSTRUCTION

if TYPE_CHECKING:
    from navprobe.llm.client import LLMClient
    from navprobe.runtime.cache import RuntimeCache


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
    if task_progress.agenda_initialized:
        task_progress.initialize_agenda(goal_text, [])
        return {"initialized": False, "source": "existing", "task_progress": task_progress.to_dict()}
    if (
        str(task_type).strip() == GOAL_KIND_OBJECT_CATEGORY or task_progress.task_constraints
    ):
        task_progress.initialize_agenda(goal_text, [])
        return {"initialized": True, "source": "online_objectives", "task_progress": task_progress.to_dict()}
    if task_progress.items != []:
        task_progress.initialize_agenda(goal_text, task_progress.items)
        return {"initialized": False, "source": "existing", "task_progress": task_progress.to_dict()}
    normalized_task_type = str(task_type).strip()
    system_prompt = """
You are the Initial Task Executive in NavProbe.
Initialize intermediate objectives from the explicit navigation instruction clauses, preserving its original route order and stopping requirements.
Return only JSON matching the provided output contract.
""".strip()
    user_prompt = f"""
Original navigation instruction:
{goal_text}

Create initial route-level objectives in instruction order. Keep each required movement together with its defining landmarks and spatial relations, preserving negation, ordinal references, and before/after dependencies. Preserve all stated stopping constraints, including stopping partway up or down a staircase; never replace a partial stair objective with a full-floor transition. Include only objectives supported by the instruction; do not strengthen movement relations or add precision, centering, or facing requirements. Retain ambiguities that require observation for online interpretation.
These initial objectives may later be refined, reprioritized, completed, abandoned, or reopened as evidence arrives. The original instruction and its ordering constraints remain authoritative. Evidence predicates are initially empty and are created online.

Output contract:
{{"agenda":[{{"content":"<instruction-derived objective>","status":"active","result":""}}]}}
""".strip()
    if task_progress.task_constraints:
        user_prompt += (
            "\n\nFixed task constraints:\n"
            + task_progress.task_constraints
            + "\nPreserve the target-validity and stopping requirements in the relevant task items."
        )
    prompt_payload: str | list[dict[str, object]] = user_prompt
    source = "llm"
    try:
        parsed = client.generate_task_progress_memory(system_prompt, prompt_payload)
        if not isinstance(parsed, dict) or set(parsed) != {"agenda"}:
            raise ValueError("NavProbe initialization requires only the agenda field")
        raw_items = parsed["agenda"]
        if not isinstance(raw_items, list) or raw_items == []:
            raise ValueError(f"visual task progress generator returned no progress_items: {parsed!r}")
        if normalized_task_type == GOAL_KIND_VLN_INSTRUCTION:
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
        initial_items = [
            TaskProgressItem.from_dict(item)
            for item in raw_items
            if isinstance(item, dict)
        ]
        if initial_items == []:
            raise ValueError(f"visual task progress generator returned invalid progress_items: {parsed!r}")
        task_progress.initialize_agenda(goal_text, initial_items)
    except Exception as exc:
        source = "fallback"
        task_progress.initialize_agenda(goal_text, [TaskProgressItem(content=goal_text)])
        return {
            "initialized": True,
            "source": source,
            "error": str(exc),
            "task_progress": task_progress.to_dict(),
        }
    return {"initialized": True, "source": source, "task_progress": task_progress.to_dict()}


_VLN_TASK_PROGRESS_TOOLS = {
    "retrieve",
    "update_task_state",
}

_VLN_NAVIGATION_TOOLS = {
    "backtrack",
    "go_to_waypoint",
    "approach_to_stop",
    "vertical_transition",
}


def _navprobe_executive_tool_schemas(
    *, allow_retrieve: bool, allow_update_progress: bool,
    require_terminal_check: bool = False,
) -> list[str]:
    schemas = ["""### `update_predicates`
Apply evidence-backed predicate changes to an active subgoal using its stable `subgoal_id`.
Arguments: `predicate_updates`, a non-empty list of these operations:
- add: {"op":"add","subgoal_id":"sg0","content":"<evidence-checkable proposition>","status":"confirmed|unconfirmed"}
- update: {"op":"update","subgoal_id":"sg0","predicate_id":"pc0","status":"confirmed|unconfirmed"}
- rewrite: {"op":"rewrite","subgoal_id":"sg0","predicate_id":"pc0","content":"<revised proposition>","status":"confirmed|unconfirmed"}
- remove: {"op":"remove","subgoal_id":"sg0","predicate_id":"pc0"}
Conditions on active objectives are applied before agenda operations. You may also reference a historical objective that is reopened in this same response: its predicate edits apply to the new attempt immediately after reopening, while the historical snapshot is preserved. Newly added objectives receive IDs on commit and can be referenced in subsequent assessments.
JSON:
{"name":"update_predicates","arguments":{"predicate_updates":[{"op":"add","subgoal_id":"sg0","content":"<supported proposition>","status":"confirmed"}]}}"""]
    if allow_update_progress:
        terminal_argument = ""
        agenda_example = '{"agenda_updates":[{"op":"complete","subgoal_id":"sg0","result":"<observed outcome>"}]}'
        if require_terminal_check:
            terminal_argument = """
Additional argument: `terminal_check`, an object with exactly `decision` (done|continue) and `missing_constraints` (a list of original-task constraints).
Use an empty list for done and a non-empty list for continue. Explain the decision in `task_state_assessment`, without a separate reason field.
"""
            terminal_argument += (
                "When calling retrieve, omit terminal_check from the entire response. Otherwise, call update_task_state with terminal_check, even if agenda_updates is empty.\n"
                if allow_retrieve else
                "Call update_task_state with terminal_check in this response, even if agenda_updates is empty.\n"
            )
            agenda_example = '{"agenda_updates":[],"terminal_check":{"decision":"continue","missing_constraints":["<unmet or unestablished original-task constraint>"]}}'
        schemas.append("""### `update_task_state`
Revise the agenda and preserve outcomes in execution history.
Arguments: `agenda_updates`, a list of these operations; an empty list is allowed:
- add: {"op":"add","content":"<new intermediate objective>","position":0}
- rewrite: {"op":"rewrite","subgoal_id":"sg0","content":"<refined objective>"}
- reorder: {"op":"reorder","subgoal_ids":["sg1","sg0"]}
- complete: {"op":"complete","subgoal_id":"sg0","result":"<execution summary and outcome>"}
- abandon: {"op":"abandon","subgoal_id":"sg0","result":"<attempt outcome and why it is no longer useful>"}
- reopen: {"op":"reopen","subgoal_id":"sg0","position":0}
Positions are zero-based pursuit priorities at the time of each operation; `reorder` lists every currently active ID exactly once. The system assigns new IDs on add. Complete and abandon remove the objective from the active agenda and archive this attempt with its evidence and spatial references. Reopen returns the same historical objective as a new attempt while retaining its previous outcome. Operations apply in list order as one validated transaction.
All agenda entries, including instruction-initialized objectives, support these operations. The original goal, required route order, and stopping constraints remain authoritative. Abandoning or rewriting an objective does not remove a requirement of the original task.
""" + terminal_argument + 'JSON:\n{"name":"update_task_state","arguments":' + agenda_example + '}')
    if allow_retrieve:
        schemas.append("""### `retrieve`
Inspect historical records that could resolve a concrete uncertainty affecting task state or the next action.
Arguments: `query`, one decision-relevant question, and `items`, a non-empty list of entity refs and fields exposed by the memory index. Batch sufficient complementary fields for that question; exclude fields marked as provided.
This is the last call in `tool_calls`; its evidence is interpreted in the next assessment. Updates in this response rely on evidence already in context.
JSON:
{"name":"retrieve","arguments":{"query":"<unresolved task-state question>","items":[{"ref":"<memory ref>","fields":["<available field>"]}]}}""")
    return schemas


def _navprobe_navigation_tool_schemas(
    *, allowed_backtrack_node_ids: set[str] | None, require_backtrack: bool,
    system_owned_waypoint_objective: bool,
    planning_reference_panorama: bool = False,
) -> list[str]:
    schemas = ["""Action objective reference:
For every action, set `subgoal_id` to the selected active objective's stable ID. When the agenda is empty, use JSON null and ground the action directly in the original task goal and its constraints. An empty agenda does not establish task completion. The examples with `<active ID>` use null in this case."""]
    schemas.append("""Grounding handoff:
The waypoint grounder receives the selected action with candidate RGB/BEV overlays. Make the action self-contained: in `reason`, specify the immediate visible route or target region, the evidence that identifies it, and the relevant route-order, target-validity, or stopping constraint. Include any retrieved correction needed to interpret that region. Express these facts directly rather than referring to an agenda ID, predicate ID, assessment, or retrieval round for their meaning.""")
    if allowed_backtrack_node_ids:
        schemas.append("""### `backtrack`
Switch the planning reference to one of the listed visited nodes to reconsider a missed route or search region. This action does not physically move the robot. The executive and skill policy rerun on the stored anchor observation. Ground the next waypoint from that reference and execute from the physical pose; switching back to the physical view is not a prerequisite for movement. Select another reference when it adds relevant evidence or useful grounding options.
Available anchors: """ + ", ".join(sorted(allowed_backtrack_node_ids)) + """
JSON:
{"backtrack":{"subgoal_id":"<active ID>","anchor_node_id":"<available anchor>","objective":"<relation or region to reassess>","reason":"<evidence supporting recovery>"}}""")
    if require_backtrack:
        return schemas
    waypoint_field = (
        '"direction":"front|back|left|right"'
        if system_owned_waypoint_objective else '"objective":"<local navigation objective>"'
    )
    stay_rule = (
        "In this historical-reference mode, `stay` requests physical navigation to the planning node's position before terminal assessment; it does not keep the robot at its current physical pose. Use it only to intentionally return to a supported final stopping position at that node, not for ordinary recovery or an intermediate objective. The resulting terminal context records movement=move; the stored image does not prove arrival."
        if planning_reference_panorama and system_owned_waypoint_objective else
        "In this physical-reference mode, `stay` keeps the robot at its current pose without movement. Use it only when physical-pose evidence supports already occupying the final stopping region."
    )
    schemas.extend([
        """### `go_to_waypoint`
Advance the selected objective, or the original task when the agenda is empty, toward a visible route or region. Specify the immediate local target and its constraints in `reason`; the waypoint grounder selects an FSS candidate for that target. Interpret direction in the current observation's frame.
JSON:
{"go_to_waypoint":{"subgoal_id":"<active ID>",""" + waypoint_field + """, "reason":"<visible route and why it advances this objective>"}}""",
        """### `approach_to_stop`
Propose final local positioning toward the original task's stopping region. Use `move` to ground a reachable local approach. """ + stay_rule + """ The Task Executive verifies the original task and terminal constraints at the next decision step before declaring completion. Preserve the original stopping relation without adding an exact historical pose, centering, or facing requirement.
Use the selected active subgoal ID, or null only if the agenda is empty. The examples show the empty-agenda case; replace null with the selected ID when an objective remains. Visibility at a distance does not establish arrival.
JSON for stay:
{"approach_to_stop":{"subgoal_id":null,"movement":"stay","stop_objective":"<original task's stopping relation>","reason":"<endpoint evidence>"}}
JSON for move:
{"approach_to_stop":{"subgoal_id":null,"movement":"move",""" + (
            '"direction":"front|back|left|right",' if system_owned_waypoint_objective else ""
        ) + """"stop_objective":"<original task's stopping relation>","reason":"<reachable approach evidence>"}}""",
        """### `vertical_transition`
Advance the selected subgoal, or the original task when the agenda is empty, along a visible, locally accessible staircase in the required direction. The waypoint grounder applies FSS with the detected stair region temporarily traversable.
`waypoint_target` states the full selected objective (the original task if no agenda objective exists) and its endpoint, preserving partial-stair stopping, turn landings, and any original route constraints. A partial stair objective is not a request for a complete floor transition.
JSON:
{"vertical_transition":{"subgoal_id":"<active ID>","vertical_direction":"up|down","waypoint_target":"<full selected objective including its required endpoint>","reason":"<visible accessible staircase and why it advances this objective>"}}""",
    ])
    return schemas


def _object_navigation_progress_rules(*, goal_kind: str) -> str:
    if str(goal_kind).strip() != GOAL_KIND_OBJECT_CATEGORY:
        return ""
    return """
ObjectNav progress rules:
- For `Find a <category>.`, task completion means reaching navigable floor close to a valid target instance; target visibility or recognition alone is only partial evidence.
- The reached-target condition requires the agent and target to be in the same accessible local region with connected reachable floor.
- A target seen only through a window, glass panel, mirror, or closed door, or outside the building in disconnected space, cannot satisfy the reached-target condition.
""".strip()


def _object_navigation_action_rules(
    *, goal_kind: str, planning_reference_panorama: bool = False,
) -> str:
    if str(goal_kind).strip() != GOAL_KIND_OBJECT_CATEGORY:
        return ""
    stay_rule = (
        "Use `approach_to_stop` with `stay` only to intentionally return to a planning node supported as near-target reachable floor; verify actual arrival after execution."
        if planning_reference_panorama else
        "Use `approach_to_stop` with `stay` only when the current pose is already on that near-target reachable floor; image proximity alone is insufficient."
    )
    return f"""
ObjectNav action rules:
- Use `go_to_waypoint` to keep searching when a target is only visible at a distance or has no visibly connected reachable approach.
- Use `approach_to_stop` with `move` only when the objective is reachable floor adjacent to the target in the same accessible local region.
- {stay_rule}
- Never approach or stop for an exterior target seen through glass or a window, or for a target separated by a mirror, closed door, enclosure, height difference, or disconnected free space.
""".strip()


def _object_navigation_terminal_rules(*, goal_kind: str) -> str:
    if str(goal_kind).strip() != GOAL_KIND_OBJECT_CATEGORY:
        return ""
    return """
- For ObjectNav, `done` requires the current endpoint to be on navigable floor close to the target in the same accessible local region.
- Seeing the target, including seeing an exterior target through glass or a window, does not prove physical approach or task completion.
- Return `continue` when reachability or physical proximity remains unconfirmed.
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
    original_instruction: str | None = None,
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
        original_instruction=original_instruction,
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
        raise TypeError("Task Executive returned a navigation decision")
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
    original_instruction: str | None = None,
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
        original_instruction=original_instruction,
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
        raise TypeError("Semantic Skill Selector returned a task-progress decision")
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
    original_instruction: str | None = None,
) -> TaskProgressDecision | RetrieveRequest | NavigationModeDecision:
    if not task_progress.agenda_initialized:
        raise ValueError("NavProbe requires an initialized task agenda")
    if not any((allow_retrieve, allow_update_progress, allow_navigation_actions)):
        raise ValueError("VLN task module has no available operation")
    if allow_navigation_actions and (allow_retrieve or allow_update_progress):
        raise ValueError(
            "Task Executive and Navigation Planner operations cannot share a request"
        )
    if allow_navigation_actions and latest_task_progress is None:
        raise ValueError("navigation action tools require current task progress")
    if require_backtrack and (
        not allow_navigation_actions or not allowed_backtrack_node_ids
    ):
        raise ValueError("required `backtrack` needs at least one available anchor")
    terminal_check_required = isinstance(terminal_check_context, dict)
    if terminal_check_required and not allow_update_progress:
        raise ValueError("post-approach terminal check requires `update_task_state`")
    tool_schemas = (
        _navprobe_navigation_tool_schemas(
            allowed_backtrack_node_ids=allowed_backtrack_node_ids,
            require_backtrack=require_backtrack,
            system_owned_waypoint_objective=system_owned_waypoint_objective,
            planning_reference_panorama=planning_reference_panorama,
        )
        if allow_navigation_actions else _navprobe_executive_tool_schemas(
            allow_retrieve=allow_retrieve, allow_update_progress=allow_update_progress,
            require_terminal_check=terminal_check_required,
        )
    )
    retrieve_remaining = max(
        0,
        int(retrieve_max_rounds) - int(retrieve_completed_rounds),
    )
    retrieval_catalog_section = ""
    retrieval_semantics = ""
    if allow_retrieve or (((not allow_navigation_actions) and (not memory_index_context_only))):
        single_retrieval_instruction = (
            "\n\nSingle-retrieval requirement:\n"
            "- This is the only RETRIEVE call available for this navigation step.\n"
            "- Use this one request to retrieve every memory entity and available field "
            "needed to resolve the current historical evidence question.\n"
            "- Include all required items together in the single `items` list."
            if allow_retrieve and int(retrieve_max_rounds) == 1
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
- `retrieve` is {"available" if allow_retrieve else "unavailable"} for this call; remaining budget alone does not grant tool access.
{single_retrieval_instruction}
""".rstrip()
        retrieval_semantics = """
Memory field semantics:
- `node.rgb`: the node's stored multi-direction panorama.
- `node.landmarks`: all landmark detections annotated in that panorama.
- `edge.rgb`: the stored start view with the smoothed executed path and endpoint node overlaid. Blue arrows show the executed travel direction.
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
        if str(latest_task_progress.progress_analysis).strip() != "":
            latest_progress_lines.extend(
                [
                    "Latest Task Executive assessment:",
                    latest_task_progress.progress_analysis,
                ]
            )
        if str(latest_task_progress.terminal_check_decision).strip() != "":
            latest_progress_lines.extend(
                [
                    f"terminal_check: {latest_task_progress.terminal_check_decision}",
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
""".strip()
        if require_retrieval_conclusion
        else ""
    )
    task_progress_semantics = """
Task-state semantics:
- The original goal defines movement, route order, and stopping requirements. The agenda is a revisable plan; generated helper objectives do not become additional original-task requirements. Preserve the original relations without adding passage, proximity, centering, or facing requirements.
- Revise only state that needs revision. Pursuit priority does not change the original instruction's required route order. Resolve objectives from their full meaning and evidence, not by counting confirmed predicates. An empty agenda or a negative search does not establish overall completion; formulate a supported next objective when work remains.
- Predicates are initially empty, dynamic, and non-exhaustive. `confirmed` records an evidence-supported claim that can be corrected; `unconfirmed` means unestablished, not false. Revise invalid or irrelevant predicates as needed.
- Use grounded hypotheses to choose useful investigation while preserving uncertainty. A negative search may resolve an inspection objective. Hypotheses, detector labels, requested actions, and visibility alone do not establish task completion; movement relations require spatial boundaries and execution evidence.
- Distinguish past execution events from current-position relations. Later movement does not erase a supported past event; contradictory evidence can change its interpretation. Historical nodes are spatial references, not exact required stopping poses unless the original task requires them. Different node IDs alone do not disprove arrival.
""".strip()
    evidence_retrieval_policy = """
Evidence and retrieval policy:
- Use sufficient current evidence directly. Retrieve for an uncertainty that historical records could resolve and whose answer could change task interpretation, progress, or action. Batch complementary records; further rounds need a decision-relevant gap.
- Conclusions remain usable when earlier raw evidence is no longer attached. Re-read for a specific gap, lost detail, or contradiction, not merely an absent image.
- The budget is a limit, not a quota. End retrieval when the next decision is supported or remaining uncertainty needs movement or a fresh observation; state that need and retain uncertainty. History cannot verify a future endpoint, and ending retrieval does not establish completion.
- A request is not evidence. Updates use evidence already supplied. Conclude every pending batch before handoff, even at zero budget; leave missing evidence unconfirmed.
""".strip()
    navigation_state_semantics = """
Task-state semantics:
- Interpret committed state using the original goal, route order, and stopping relation; helper objectives and historical nodes cannot strengthen them. Agenda order is pursuit priority.
- Use the Executive assessment and evidence conclusions. Carry contradictions and material uncertainty in `reason` without rewriting state. Confirmed predicates can be corrected; unconfirmed means unknown, not false. An empty agenda or negative search does not establish completion; the Executive assesses completion.
""".strip()
    navigation_action_rules = ""
    if allow_navigation_actions:
        navigation_action_rules = """
Semantic skill selection:
Choose one available skill to advance the task or inspect a grounded hypothesis. An intermediate visible route or inspection region is useful even when its destination is unknown; preserve required turns, passages, and stopping boundaries without inventing unseen contents. Express the local intent in the panorama's reference frame with a concise, self-contained reason as specified below. Use grounding rejection feedback to revise the local target.
""".strip()
        object_action_rules = _object_navigation_action_rules(
            goal_kind=goal_kind,
            planning_reference_panorama=(
                planning_reference_panorama and system_owned_waypoint_objective
            ),
        )
        if object_action_rules:
            navigation_action_rules += "\n\n" + object_action_rules
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
        terminal_call_rule = (
            "If this response calls `retrieve`, omit `terminal_check` until the returned evidence has been assessed. Otherwise, call `update_task_state` with `terminal_check`, even when `agenda_updates` is empty."
            if allow_retrieve else
            "Call `update_task_state` with `terminal_check` in this response, even when `agenda_updates` is empty."
        )
        terminal_check_section = f"""

Post-approach terminal check:
- {approach_result}
- Judge the final stopping relation at the actual physical endpoint and required route events from supported execution history. The approach objective is a proposed plan, not an additional task requirement.
- `done` requires evidence that the original goal and all required route-order and stopping constraints are satisfied. Complete supported objectives and abandon only obsolete helper objectives in this transaction; abandonment does not satisfy an original requirement.
- `continue` names the unmet or unestablished original-task constraints, including when the agenda is empty. Distinguish a known violation from missing evidence; add a supported objective if further pursuit is clear.
- Do not require exact reproduction of a historical pose, centering, orientation, or extra proximity unless the original task requires it. Distinct node IDs alone do not disprove arrival; assess the physical stopping relation.
- Agenda exhaustion and completion of a negative search are insufficient for `done`.
- {terminal_call_rule}
- Explain the terminal decision in `task_state_assessment` using the current endpoint evidence and final stopping requirements. Return `decision` and `missing_constraints` in `terminal_check`, without a separate `reason`.
{_object_navigation_terminal_rules(goal_kind=goal_kind)}

Proposed approach objective (interpret against the original task):
{stop_objective}
""".rstrip()
    system_prompt = (
        "You are the NavProbe skill policy's semantic navigation planner.\n"
        "Select one parameterized skill from the original goal, current agenda and history, executive assessment, observation, and retrieved evidence conclusions.\n"
        "Return JSON matching one action contract."
        if allow_navigation_actions else
        "You are the NavProbe task executive.\n"
        "Formulate, revise, and resolve intermediate objectives and verification predicates from the original goal, observation, task state, and episodic-memory index.\n"
        "Probe historical evidence when an unresolved task-state question requires it, interpret returned evidence, and preserve outcomes for reconsideration.\n"
        "Return JSON matching the response protocol."
    )
    task_progress_section = (
        ("Task state (agenda and execution history):"
        + "\n"
        + task_progress.format_for_prompt(
            include_node_bindings=visual_context.graph_context_visible,
            show_empty_progress_conditions=True,
        ))
    )
    response_example: dict[str, object] = {}
    if require_retrieval_conclusion:
        response_example["retrieval_conclusion"] = (
            "<direct answer to the latest retrieval query, supporting refs, and unresolved evidence>"
        )
    response_example["task_state_assessment"] = (
        "<supported progress, required updates, and any decision-relevant gap or needed action/observation>"
    )
    response_example["tool_calls"] = []
    conclusion_protocol = (
        "- This response must output `retrieval_conclusion`, then `task_state_assessment`, then `tool_calls`.\n"
        "- `retrieval_conclusion` must directly answer the latest retrieval query, naming supporting refs and what remains unconfirmed.\n"
        if require_retrieval_conclusion
        else "- This response must output `task_state_assessment`, then `tool_calls`; omit `retrieval_conclusion` because no retrieval round awaits a conclusion.\n"
    )
    response_examples = "Response example for this call (no changes or retrieval needed):\n" + json.dumps(response_example)
    if terminal_check_required:
        terminal_example = deepcopy(response_example)
        terminal_example["tool_calls"] = [{
            "name": "update_task_state",
            "arguments": {
                "agenda_updates": [],
                "terminal_check": {
                    "decision": "continue",
                    "missing_constraints": ["<remaining endpoint requirement>"],
                },
            },
        }]
        response_examples = (
            "Terminal assessment without retrieval: call update_task_state with terminal_check, even when agenda_updates is empty. "
            "Use decision=done with missing_constraints=[] only when all task items are done and the endpoint requirements are satisfied; "
            "otherwise use continue and name the remaining constraints.\n"
            + json.dumps(terminal_example)
        )
        response_examples = response_examples.replace(
            "all task items are done and the endpoint requirements are satisfied",
            "the original task and its ordering and endpoint requirements are established and the remaining agenda has been resolved",
        )
        if allow_retrieve:
            retrieval_example = deepcopy(response_example)
            retrieval_example["tool_calls"] = [
                {"name": "update_task_state", "arguments": {"agenda_updates": []}},
                {"name": "retrieve", "arguments": {
                    "query": "<decision-relevant question answerable from historical records>",
                    "items": [{"ref": "<retrievable ref>", "fields": ["<available field>"]}],
                }},
            ]
            response_examples = (
                "If historical records could resolve the remaining question, optionally update progress, then call retrieve. "
                "Omit terminal_check from this entire response; assess the endpoint after the evidence returns.\n"
                + json.dumps(retrieval_example) + "\n\n" + response_examples
            )
    executive_tool_order = ["`update_predicates`"]
    if allow_update_progress:
        executive_tool_order.append("`update_task_state`")
    if allow_retrieve:
        executive_tool_order.append("`retrieve`")
    retrieval_call_protocol = (
        "- If `retrieve` is present, the system applies the updates, loads the requested evidence, and calls Task Executive again.\n"
        "- If `retrieve` is absent, Task Executive reasoning ends after the updates are applied.\n"
        if allow_retrieve else
        "- `retrieve` is unavailable for this call. Interpret any pending evidence before handing off; Task Executive reasoning ends after the supported updates and any required terminal check.\n"
    )
    response_protocol = (
        (
            "Action contracts:\n\n" + "\n\n".join(tool_schemas)
        )
        if allow_navigation_actions
        else (
            "Tool definitions:\n\n"
            + "\n\n".join(tool_schemas)
            + "\n\nResponse protocol:\n"
            + "- Include a non-empty `task_state_assessment` in every response, including when `tool_calls` is empty.\n"
            + conclusion_protocol
            + "- `task_state_assessment` is a concise evidence summary of progress, updates, and the next objective or material uncertainty, including any needed action or observation; no step-by-step reasoning transcript is required.\n"
            + "- Put the reasons for item updates and condition operations in `task_state_assessment`. Their tool arguments specify the changes without separate `reason` fields.\n"
            + f"- `tool_calls` contains zero to {len(executive_tool_order)} calls. Use each available tool at most once.\n"
            + "- Order calls as " + ", ".join(executive_tool_order) + ". Omit calls that are not needed.\n"
            + "- `update_predicates` must contain at least one operation.\n"
            + "- When no update, retrieval, or terminal check is needed, return `tool_calls: []`.\n"
            + retrieval_call_protocol
            + "- Include no other top-level fields.\n\n"
            + response_examples
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
            (
                _object_navigation_progress_rules(goal_kind=goal_kind)
                if allow_update_progress
                else ""
            ),
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
        observation_reference_lines.append(
            "The physical robot remains at the physical node stated in the backtrack context. "
            "Use this stored panorama to reassess the route and ground the next waypoint; execution starts from the physical robot pose. "
            "There is no need to switch back to the physical view or physically return to the anchor solely because the reference changed. "
            "A required physical revisit must be supported by the original route, actual path constraints, or necessary new observations; a generated recovery objective is not independent evidence for it. "
            "Selecting an anchor-grounded waypoint does not guarantee passage through the anchor. "
            "A planning-reference switch supplies no new execution evidence: arrival, entry, "
            "passage, approach, and stopping judgments must be grounded in actual observations "
            "and executed trajectories. The BEV reference marker identifies the planning anchor."
        )
    if landmark_text != "":
        observation_reference_lines.append(landmark_text)
    content.append(
        {"type": "text", "text": "\n".join(observation_reference_lines)}
    )
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
                        "Semantic Skill Selector requires current task progress"
                    )
                navigation_decision = normalize_vln_navigation_step(
                    parsed,
                    latest_task_progress=latest_task_progress,
                    allowed_backtrack_node_ids=allowed_backtrack_node_ids,
                    require_backtrack=require_backtrack,
                    system_owned_waypoint_objective=(
                        system_owned_waypoint_objective
                    ),
                    active_subgoal_ids=({item.subgoal_id for item in task_progress.items}),
                )
                return navigation_decision
            decision = normalize_vln_task_progress_step(
                parsed,
                retrieve_fields_by_ref=retrieve_fields_by_ref,
                retrieve_provided_fields_by_ref=retrieve_provided_fields_by_ref,
                allow_retrieve=allow_retrieve,
                allow_update_progress=allow_update_progress,
                require_retrieval_conclusion=require_retrieval_conclusion,
                require_terminal_check=terminal_check_required,
                    )
            projected_progress = deepcopy(task_progress)
            projected_progress.apply_agenda_updates(
                list(decision.progress_updates),
                list(decision.progress_condition_updates),
                current_node_id="" if planning_reference_panorama else visual_context.current_node_id,
            )
            if (
                isinstance(decision, TaskProgressDecision)
                and decision.terminal_check_decision == "done"
                and projected_progress.items
            ):
                raise ValueError("terminal_check done requires resolving or abandoning remaining agenda objectives")
            return decision
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
        raise ValueError("predicate_updates must be a list")
    normalized: list[dict[str, object]] = []
    for raw_update in raw_updates:
        if not isinstance(raw_update, dict):
            raise ValueError(f"progress condition update must be an object: {raw_update!r}")
        op = str(raw_update.get("op", "")).strip().lower()
        raw_id_field = "predicate_id"
        parent_field = "subgoal_id"
        expected_fields = {
            "add": {"op", parent_field, "content", "status"},
            "update": {"op", parent_field, raw_id_field, "status"},
            "rewrite": {
                "op",
                parent_field,
                raw_id_field,
                "content",
                "status",
            },
            "remove": {"op", parent_field, raw_id_field},
        }.get(op)
        if expected_fields is None or set(raw_update) != expected_fields:
            raise ValueError(f"invalid predicate operation; expected fields {expected_fields!r}: {raw_update!r}")
        parent_ref = raw_update.get(parent_field)
        if not isinstance(parent_ref, str) or not parent_ref.strip():
            raise ValueError("progress condition requires a non-empty subgoal_id")
        item: dict[str, object] = {
            "op": op,
            parent_field: parent_ref.strip(),
        }
        if raw_id_field in expected_fields:
            predicate_id = str(raw_update.get(raw_id_field, "")).strip()
            if predicate_id == "":
                raise ValueError(f"{op} progress condition update requires predicate_id")
            item["predicate_id"] = predicate_id
        if "content" in expected_fields:
            content = str(raw_update.get("content", "")).strip()
            if content == "":
                raise ValueError(f"{op} progress condition update requires content")
            item["content"] = content
        if "status" in expected_fields:
            status = str(raw_update.get("status", "")).strip().lower()
            if status not in {"unconfirmed", "confirmed"}:
                raise ValueError(f"invalid progress condition status: {raw_update!r}")
            item["status"] = status
        normalized.append(item)
    return normalized


def _normalize_vln_progress_updates(
    raw_updates: object,
) -> list[dict[str, object]]:
    if not isinstance(raw_updates, list):
        raise ValueError("update_task_state agenda_updates must be a list")
    normalized: list[dict[str, object]] = []
    for raw_update in raw_updates:
        if not isinstance(raw_update, dict):
            raise ValueError(f"progress update must be an object: {raw_update!r}")
        op = str(raw_update.get("op", "")).strip().lower()
        expected_fields = {
            "add": {"op", "content", "position"},
            "rewrite": {"op", "subgoal_id", "content"},
            "reorder": {"op", "subgoal_ids"},
            "complete": {"op", "subgoal_id", "result"},
            "abandon": {"op", "subgoal_id", "result"},
            "reopen": {"op", "subgoal_id", "position"},
        }.get(op)
        if expected_fields is None or set(raw_update) != expected_fields:
            raise ValueError(f"invalid agenda operation fields: {raw_update!r}")
        item = {"op": op}
        for name in expected_fields - {"op"}:
            value = raw_update[name]
            if name == "position":
                if type(value) is not int or value < 0:
                    raise ValueError("agenda position must be a nonnegative integer")
            elif name == "subgoal_ids":
                if not isinstance(value, list) or any(not isinstance(ref, str) or not ref.strip() for ref in value):
                    raise ValueError("reorder requires a list of subgoal IDs")
                value = [ref.strip() for ref in value]
                if len(value) != len(set(value)):
                    raise ValueError("reorder cannot duplicate a subgoal ID")
            else:
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"{op} requires non-empty {name}")
                value = value.strip()
            item[name] = value
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
        raise TypeError("Task Executive normalized a navigation decision")
    return decision


def normalize_vln_navigation_step(
    payload: dict[str, object],
    *,
    latest_task_progress: TaskProgressDecision,
    allowed_backtrack_node_ids: set[str] | None = None,
    require_backtrack: bool = False,
    system_owned_waypoint_objective: bool = False,
    active_subgoal_ids: set[str] | None = None,
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
        active_subgoal_ids=active_subgoal_ids,
    )
    if not isinstance(decision, NavigationModeDecision):
        raise TypeError("PCNP normalized a task-progress decision")
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
    active_subgoal_ids: set[str] | None = None,
) -> TaskProgressDecision | RetrieveRequest | NavigationModeDecision:
    if not isinstance(payload, dict):
        raise ValueError("NavProbe response must be an object")
    module_name = (
        "Semantic Skill Selector"
        if allow_navigation_actions
        else "Task Executive"
    )
    module_tools = (
        _VLN_NAVIGATION_TOOLS
        if allow_navigation_actions
        else _VLN_TASK_PROGRESS_TOOLS
    )
    progress_analysis = ""
    if not allow_navigation_actions:
        raw_analysis = payload.get("task_state_assessment")
        if not isinstance(raw_analysis, str) or raw_analysis.strip() == "":
            raise ValueError("Task Executive requires non-empty task_state_assessment")
        progress_analysis = raw_analysis.strip()
    if not allow_navigation_actions:
        payload = _parse_executive_tool_calls(
            payload,
            require_retrieval_conclusion=require_retrieval_conclusion,
        )
    selected_tools = [name for name in module_tools if name in payload]
    if allow_navigation_actions and len(selected_tools) != 1:
        raise ValueError(
            f"{module_name} must select exactly one available operation: "
            f"{payload!r}"
        )
    selected_tool = (
        selected_tools[0]
        if allow_navigation_actions
        else ("retrieve" if "retrieve" in payload else "update_task_state")
    )
    if require_backtrack and selected_tool != "backtrack":
        raise ValueError(
            f"only backtrack is available in this state: {payload!r}"
        )
    expected_top_fields = set(selected_tools)
    if not allow_navigation_actions:
        expected_top_fields.add("task_state_assessment")
    if require_retrieval_conclusion:
        expected_top_fields.add("retrieval_conclusion")
    if not allow_navigation_actions:
        expected_top_fields.add("predicate_updates")
    if set(payload) != expected_top_fields:
        raise ValueError(
            f"{module_name} returned unexpected top-level fields: "
            f"{payload!r}"
        )
    retrieval_conclusion = str(payload.get("retrieval_conclusion", "")).strip()
    if require_retrieval_conclusion and retrieval_conclusion == "":
        raise ValueError(
            "Task Executive must conclude the latest retrieval "
            f"before its next tool: {payload!r}"
        )

    raw_tool = (
        payload.get(selected_tool)
        if allow_navigation_actions
        else payload.get("update_task_state", {"agenda_updates": []})
    )
    if not isinstance(raw_tool, dict):
        raise ValueError(f"{selected_tool} tool payload must be an object: {payload!r}")
    if not allow_navigation_actions:
        if not allow_update_progress and (
            "update_task_state" in payload or selected_tool != "retrieve"
        ):
            raise ValueError(f"update_progress is not available in this state: {payload!r}")
        expected_fields = {"agenda_updates"}
        if require_terminal_check and selected_tool != "retrieve":
            expected_fields.add("terminal_check")
        if require_terminal_check:
            if selected_tool == "retrieve" and "terminal_check" in raw_tool:
                raise ValueError(
                    "Omit terminal_check when calling retrieve; assess the endpoint after the retrieved evidence returns."
                )
            if selected_tool != "retrieve" and "terminal_check" not in raw_tool:
                raise ValueError(
                    "Missing required field: update_task_state.arguments.terminal_check. "
                    "Without retrieve, call update_task_state with terminal_check even when agenda_updates is empty."
                )
        if set(raw_tool) != expected_fields:
            raise ValueError(
                "update_task_state fields must be exactly "
                f"{sorted(expected_fields)!r}: {payload!r}"
            )
        try:
            progress_updates = _normalize_vln_progress_updates(
                raw_tool.get("agenda_updates"),
            )
            progress_condition_updates = _normalize_vln_progress_condition_updates(
                payload.get("predicate_updates"),
            )
        except ValueError as exc:
            raise ValueError(f"{exc}: {payload!r}") from exc
        if selected_tool == "retrieve":
            if not allow_retrieve:
                raise ValueError(f"retrieve is not available in this state: {payload!r}")
            raw_retrieve = payload["retrieve"]
            if not isinstance(raw_retrieve, dict) or set(raw_retrieve) != {"query", "items"}:
                raise ValueError("retrieve arguments must contain exactly query and items")
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
                progress_updates=tuple(progress_updates),
                progress_analysis=progress_analysis,
            )
        terminal_check_decision = ""
        terminal_check_reasoning = ""
        terminal_check_missing_constraints: list[str] = []
        if require_terminal_check:
            raw_terminal_check = raw_tool.get("terminal_check")
            if not isinstance(raw_terminal_check, dict):
                raise ValueError(f"update_progress terminal_check must be an object: {payload!r}")
            expected_terminal_fields = {
                "decision",
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
            terminal_check_reasoning = progress_analysis
            raw_missing_constraints = raw_terminal_check.get("missing_constraints")
            if terminal_check_decision not in {"done", "continue"}:
                raise ValueError(f"terminal_check decision must be done or continue: {payload!r}")
            if not isinstance(raw_missing_constraints, list) or any(
                not isinstance(item, str) or not item.strip() for item in raw_missing_constraints
            ):
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
            progress_analysis=progress_analysis,
            progress_reasoning="",
            retrieval_conclusion=retrieval_conclusion,
            route_status="",
            status_reasoning="",
            progress_updates=progress_updates,
            progress_condition_updates=progress_condition_updates,
            terminal_check_decision=terminal_check_decision,
            terminal_check_reasoning=terminal_check_reasoning,
            terminal_check_missing_constraints=terminal_check_missing_constraints,
            update_progress_called="update_task_state" in payload,
        )

    if not allow_navigation_actions or latest_task_progress is None:
        raise ValueError(f"{selected_tool} is not available before update_progress: {payload!r}")
    progress_fields = {
        "progress_analysis": "",
        "progress_reasoning": "",
        "progress_updates": [],
    }
    if "subgoal_id" not in raw_tool:
        raise ValueError("navigation action requires subgoal_id")
    subgoal_id = raw_tool["subgoal_id"]
    if subgoal_id is None:
        if active_subgoal_ids:
            raise ValueError("subgoal_id=null is allowed only with an empty agenda")
    elif (
        not isinstance(subgoal_id, str) or not subgoal_id.strip()
        or (active_subgoal_ids is not None and subgoal_id not in active_subgoal_ids)
    ):
        raise ValueError("navigation action must reference an active subgoal_id")
    progress_fields["subgoal_id"] = subgoal_id
    raw_tool = {key: value for key, value in raw_tool.items() if key != "subgoal_id"}
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
        expected_fields = {"vertical_direction", "reason", "waypoint_target"}
        if set(raw_tool) != expected_fields:
            raise ValueError(f"vertical_transition fields must be exactly {sorted(expected_fields)!r}: {payload!r}")
        vertical_direction = str(raw_tool.get("vertical_direction", "")).strip()
        reason = str(raw_tool.get("reason", "")).strip()
        waypoint_target = str(raw_tool.get("waypoint_target", "")).strip()
        if vertical_direction not in {"up", "down"} or reason == "":
            raise ValueError(f"vertical_transition requires direction and reason: {payload!r}")
        if not waypoint_target:
            raise ValueError("vertical_transition requires waypoint_target with the full selected objective and endpoint")
        return NavigationModeDecision(
            action_mode="vertical_transition",
            stop_objective="",
            reasoning_action=reason,
            vertical_direction=vertical_direction,
            waypoint_target=waypoint_target,
            route_status="",
            action_objective=waypoint_target or f"Take the {vertical_direction} vertical transition.",
            action_reason=reason,
            **progress_fields,
        )
    raise ValueError(f"unsupported VLN progress-navigation tool: {payload!r}")


def _parse_executive_tool_calls(
    payload: dict[str, object],
    *,
    require_retrieval_conclusion: bool,
) -> dict[str, object]:
    expected_top_fields = {"task_state_assessment", "tool_calls"}
    if require_retrieval_conclusion:
        expected_top_fields.add("retrieval_conclusion")
    missing_fields = expected_top_fields - set(payload)
    unexpected_fields = set(payload) - expected_top_fields
    if missing_fields or unexpected_fields:
        details = []
        if missing_fields:
            details.append(f"Missing required top-level fields: {sorted(missing_fields)!r}.")
        if "retrieval_conclusion" in missing_fields:
            details.append("Answer the latest retrieval query in retrieval_conclusion before issuing further tools.")
        if unexpected_fields:
            details.append(f"Unexpected top-level fields: {sorted(unexpected_fields)!r}.")
        if "retrieval_conclusion" in unexpected_fields:
            details.append("Omit retrieval_conclusion because no retrieval round awaits a conclusion.")
        raise ValueError("Task Executive: " + " ".join(details))
    if require_retrieval_conclusion:
        if str(payload.get("retrieval_conclusion", "")).strip() == "":
            raise ValueError(
                "Task Executive must conclude the latest retrieval "
                f"before its next tool: {payload!r}"
            )
    raw_calls = payload.get("tool_calls")
    if not isinstance(raw_calls, list) or len(raw_calls) > 3:
        raise ValueError("tool_calls must contain zero to three calls")
    call_order = {"update_predicates": 0, "update_task_state": 1, "retrieve": 2}
    previous_position = -1
    normalized: dict[str, object] = {
        "task_state_assessment": payload["task_state_assessment"],
        "predicate_updates": [],
    }
    if require_retrieval_conclusion:
        normalized["retrieval_conclusion"] = str(payload["retrieval_conclusion"]).strip()
    for raw_call in raw_calls:
        if not isinstance(raw_call, dict) or set(raw_call) != {"name", "arguments"}:
            raise ValueError(f"invalid TPU tool call: {raw_call!r}")
        name = str(raw_call.get("name", "")).strip()
        arguments = raw_call.get("arguments")
        if not isinstance(arguments, dict):
            raise ValueError(f"TPU tool arguments must be an object: {raw_call!r}")
        if name not in call_order or call_order[name] <= previous_position:
            raise ValueError(
                "Task Executive tools may occur at most once, ordered as "
                "update_predicates, update_task_state, retrieve"
            )
        previous_position = call_order[name]
        if name == "update_predicates":
            if set(arguments) != {"predicate_updates"}:
                raise ValueError(
                    "update_predicates arguments must contain exactly predicate_updates"
                )
            condition_updates = arguments["predicate_updates"]
            if not isinstance(condition_updates, list) or not condition_updates:
                raise ValueError("update_predicates requires non-empty predicate_updates")
            normalized["predicate_updates"] = condition_updates
        else:
            normalized[name] = arguments
    return normalized
