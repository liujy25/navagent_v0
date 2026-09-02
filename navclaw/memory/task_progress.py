from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


PROGRESS_STATUS_VALUES = {"active", "done"}
LEGACY_PROGRESS_STATUS_MAP = {
    "pending": "active",
    "completed": "done",
}
PROGRESS_KIND_VALUES = {"task", "verify_candidate"}
PROGRESS_CONDITION_STATUS_VALUES = {"unconfirmed", "confirmed"}
LEGACY_PROGRESS_CONDITION_STATUS_MAP = {
    "missing": "unconfirmed",
    "satisfied": "confirmed",
}


@dataclass
class TaskProgressCondition:
    condition_id: str
    update_node_id: str
    content: str
    status: str = "unconfirmed"

    def __post_init__(self) -> None:
        condition_id = str(self.condition_id).strip()
        update_node_id = str(self.update_node_id).strip()
        content = str(self.content).strip()
        status = _normalize_progress_condition_status(self.status)
        if condition_id == "":
            raise ValueError("progress condition requires a condition_id")
        if content == "":
            raise ValueError("progress condition content must be non-empty")
        if status not in PROGRESS_CONDITION_STATUS_VALUES:
            raise ValueError(f"unsupported progress condition status: {status!r}")
        self.condition_id = condition_id
        self.update_node_id = update_node_id
        self.content = content
        self.status = status

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "TaskProgressCondition":
        return cls(
            condition_id=str(
                payload.get("condition_id", payload.get("knowledge_id", ""))
            ),
            update_node_id=str(payload.get("update_node_id", "")),
            content=str(payload.get("content", "")),
            status=str(payload.get("status", "unconfirmed")),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "condition_id": str(self.condition_id),
            "update_node_id": str(self.update_node_id),
            "content": str(self.content),
            "status": str(self.status),
        }


@dataclass
class TaskProgressItem:
    content: str
    status: str = "active"
    result: str = ""
    kind: str = "task"
    candidate: dict[str, object] = field(default_factory=dict)
    start_node_id: str = ""
    completion_node_id: str = ""
    conditions: list[TaskProgressCondition] = field(default_factory=list)

    def __post_init__(self) -> None:
        content = str(self.content).strip()
        status = _normalize_progress_status(self.status)
        result = str(self.result).strip()
        if status == "active":
            result = ""
        kind = str(self.kind).strip().lower() or "task"
        candidate = deepcopy(self.candidate) if isinstance(self.candidate, dict) else {}
        conditions = [
            TaskProgressCondition.from_dict(item.to_dict())
            if isinstance(item, TaskProgressCondition)
            else TaskProgressCondition.from_dict(item)
            for item in self.conditions
            if isinstance(item, (TaskProgressCondition, dict))
        ]
        start_node_id = str(self.start_node_id).strip()
        completion_node_id = (
            str(self.completion_node_id).strip() if status == "done" else ""
        )
        if content == "":
            raise ValueError("TaskProgressItem.content must be non-empty")
        if status not in PROGRESS_STATUS_VALUES:
            raise ValueError(f"unsupported TaskProgressItem.status: {status!r}")
        if kind not in PROGRESS_KIND_VALUES:
            raise ValueError(f"unsupported TaskProgressItem.kind: {kind!r}")
        if status == "done" and result == "":
            raise ValueError("done TaskProgressItem requires non-empty result")
        self.content = content
        self.status = status
        self.result = result
        self.kind = kind
        self.candidate = candidate
        self.start_node_id = start_node_id
        self.completion_node_id = completion_node_id
        self.conditions = conditions

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "TaskProgressItem":
        candidate = payload.get("candidate", {})
        raw_conditions = payload.get("conditions", payload.get("knowledge", []))
        return cls(
            content=str(payload.get("content", "")),
            status=str(payload.get("status", "active")),
            result=str(payload.get("result", "")),
            kind=str(payload.get("kind", "task")),
            candidate=deepcopy(candidate) if isinstance(candidate, dict) else {},
            start_node_id=str(payload.get("start_node_id", "")),
            completion_node_id=str(payload.get("completion_node_id", "")),
            conditions=[
                TaskProgressCondition.from_dict(item)
                for item in raw_conditions
                if isinstance(item, dict)
            ]
            if isinstance(raw_conditions, list)
            else [],
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "content": str(self.content),
            "status": str(self.status),
            "result": str(self.result),
            "kind": str(self.kind),
            "candidate": deepcopy(self.candidate),
            "start_node_id": str(self.start_node_id),
            "completion_node_id": str(self.completion_node_id),
            "conditions": [item.to_dict() for item in self.conditions],
        }


@dataclass
class TaskProgressUpdateResult:
    applied_updates: list[dict[str, object]] = field(default_factory=list)
    skipped_updates: list[dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "applied_updates": deepcopy(self.applied_updates),
            "skipped_updates": deepcopy(self.skipped_updates),
        }


@dataclass
class TaskProgressMemory:
    items: list[TaskProgressItem] = field(default_factory=list)
    next_candidate_index: int = 0

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "TaskProgressMemory":
        # Legacy input compatibility for older logs/state snapshots.
        raw_items = payload.get("progress_items", payload.get("items", payload.get("todos", [])))
        if not isinstance(raw_items, list):
            raise ValueError("TaskProgressMemory.progress_items must be a list")
        memory = cls(items=[TaskProgressItem.from_dict(item) for item in raw_items if isinstance(item, dict)])
        raw_next = payload.get("next_candidate_index")
        if isinstance(raw_next, int) and raw_next >= 0:
            memory.next_candidate_index = int(raw_next)
        else:
            memory.next_candidate_index = _next_candidate_index_from_items(memory.items)
        return memory

    @classmethod
    def fallback(cls, goal_text: str) -> "TaskProgressMemory":
        goal = str(goal_text).strip() or "target object"
        return cls(items=[TaskProgressItem(content=f"Find {goal}.", status="active", result="")])

    def ensure_initialized(self, goal_text: str) -> None:
        if self.items == []:
            self.items = TaskProgressMemory.fallback(goal_text).items

    def to_dict(self) -> dict[str, object]:
        return {
            "progress_items": [item.to_dict() for item in self.items],
            "next_candidate_index": int(self.next_candidate_index),
        }

    def format_for_prompt(
        self,
        *,
        include_node_bindings: bool = True,
        show_empty_progress_conditions: bool = False,
    ) -> str:
        if self.items == []:
            return "empty"
        lines = []
        for index, item in enumerate(self.items):
            result = f"; result={item.result}" if item.result != "" else ""
            node_binding = ""
            if include_node_bindings and item.start_node_id != "":
                completion = item.completion_node_id or "pending"
                node_binding = f"; node span: {item.start_node_id} -> {completion}"
            lines.append(
                f"{index}. [{item.status}] ({item.kind}) "
                f"{item.content}{result}{node_binding}"
            )
            for condition in item.conditions:
                update_node = (
                    f"; updated at {condition.update_node_id}"
                    if include_node_bindings and condition.update_node_id != ""
                    else ""
                )
                lines.append(
                    f"   - {condition.condition_id} [{condition.status}] "
                    f"{condition.content}{update_node}"
                )
            if (
                show_empty_progress_conditions
                and item.kind == "task"
                and item.conditions == []
            ):
                lines.append("   conditions: empty")
            if item.kind == "verify_candidate":
                candidate_text = _format_candidate_for_prompt(item.candidate)
                if candidate_text != "":
                    lines.append(candidate_text)
        return "\n".join(lines)

    def apply_updates(
        self,
        raw_updates: object,
        *,
        candidate_source_view: dict[str, object] | None = None,
        current_node_id: str = "",
    ) -> TaskProgressUpdateResult:
        result = TaskProgressUpdateResult()
        node_id = str(current_node_id).strip()
        self.ensure_current_task_started(node_id)
        if raw_updates is None:
            return result
        if not isinstance(raw_updates, list):
            result.skipped_updates.append({"reason": "updates_not_list", "update": deepcopy(raw_updates)})
            return result
        for raw_update in raw_updates:
            if not isinstance(raw_update, dict):
                result.skipped_updates.append({"reason": "update_not_object", "update": deepcopy(raw_update)})
                continue
            self._apply_one_update(
                raw_update,
                result,
                candidate_source_view=candidate_source_view,
                current_node_id=node_id,
            )
            self.ensure_current_task_started(node_id)
        return result

    def apply_condition_updates(
        self,
        raw_updates: object,
        *,
        current_node_id: str = "",
    ) -> TaskProgressUpdateResult:
        result = TaskProgressUpdateResult()
        node_id = str(current_node_id).strip()
        self.ensure_current_task_started(node_id)
        if raw_updates is None:
            return result
        if not isinstance(raw_updates, list):
            result.skipped_updates.append(
                {
                    "reason": "progress_condition_updates_not_list",
                    "update": deepcopy(raw_updates),
                }
            )
            return result
        for raw_update in raw_updates:
            if not isinstance(raw_update, dict):
                result.skipped_updates.append(
                    {
                        "reason": "progress_condition_update_not_object",
                        "update": deepcopy(raw_update),
                    }
                )
                continue
            self._apply_one_condition_update(
                raw_update,
                result,
                current_node_id=node_id,
            )
        return result

    def ensure_current_task_started(self, current_node_id: str) -> None:
        node_id = str(current_node_id).strip()
        if node_id == "":
            return
        for item in self.items:
            if item.status != "active":
                continue
            if item.start_node_id == "":
                item.start_node_id = node_id
            return

    def _apply_one_update(
        self,
        update: dict[str, Any],
        result: TaskProgressUpdateResult,
        *,
        candidate_source_view: dict[str, object] | None,
        current_node_id: str,
    ) -> None:
        op = str(update.get("op", "")).strip().lower()
        if op not in {"update", "rewrite", "add", "insert", "remove"}:
            result.skipped_updates.append({"reason": "unsupported_op", "update": deepcopy(update)})
            return
        index = update.get("index")
        if op in {"update", "rewrite", "insert", "remove"}:
            if not isinstance(index, int):
                result.skipped_updates.append({"reason": "missing_integer_index", "update": deepcopy(update)})
                return
        if op in {"update", "rewrite", "remove"} and not (0 <= int(index) < len(self.items)):
            result.skipped_updates.append({"reason": "index_out_of_range", "update": deepcopy(update)})
            return

        if op == "remove":
            removed = self.items.pop(int(index))
            applied = {"op": op, "index": int(index), "removed": removed.to_dict()}
            result.applied_updates.append(applied)
            return

        if "satisfied_constraints" in update or "missing_constraints" in update:
            done_skip_reason = _legacy_done_constraint_skip_reason(update)
            if done_skip_reason is not None:
                result.skipped_updates.append(
                    {
                        "reason": done_skip_reason,
                        "update": deepcopy(update),
                    }
                )
                return

        if op == "add":
            item = _progress_item_from_update(update)
            if item is None:
                result.skipped_updates.append({"reason": "invalid_progress_item", "update": deepcopy(update)})
                return
            self._ensure_verify_candidate_metadata(item, candidate_source_view=candidate_source_view)
            _bind_new_done_item(item, current_node_id=current_node_id)
            self.items.append(item)
            result.applied_updates.append({"op": op, "index": len(self.items) - 1, "progress_item": item.to_dict()})
            return

        if op == "insert":
            item = _progress_item_from_update(update)
            if item is None:
                result.skipped_updates.append({"reason": "invalid_progress_item", "update": deepcopy(update)})
                return
            self._ensure_verify_candidate_metadata(item, candidate_source_view=candidate_source_view)
            _bind_new_done_item(item, current_node_id=current_node_id)
            insert_at = max(0, min(int(index), len(self.items)))
            self.items.insert(insert_at, item)
            result.applied_updates.append({"op": op, "index": insert_at, "progress_item": item.to_dict()})
            return

        current = self.items[int(index)]
        # Node-span text is rendered beside content and owned by the system.
        # An ordinary update must not copy that annotation into semantic task text.
        content = (
            str(update.get("content", "")).strip()
            if op == "rewrite"
            else current.content
        )
        status = _normalize_progress_status(update.get("status", current.status))
        result_text = str(update.get("result", current.result)).strip()
        kind = str(update.get("kind", current.kind)).strip().lower() or current.kind
        candidate = deepcopy(current.candidate)
        raw_candidate = update.get("candidate")
        if isinstance(raw_candidate, dict):
            candidate = _merge_candidate(candidate, raw_candidate)
        try:
            item = TaskProgressItem(
                content=content,
                status=status,
                result=result_text,
                kind=kind,
                candidate=candidate,
                start_node_id=current.start_node_id,
                completion_node_id=current.completion_node_id,
                conditions=[
                    TaskProgressCondition.from_dict(entry.to_dict())
                    for entry in current.conditions
                ],
            )
        except ValueError as exc:
            result.skipped_updates.append(
                {"reason": "invalid_progress_item", "error": str(exc), "update": deepcopy(update)}
            )
            return
        _preserve_and_update_node_binding(
            item,
            existing=current,
            current_node_id=current_node_id,
        )
        self._ensure_verify_candidate_metadata(item, candidate_source_view=candidate_source_view)
        self.items[int(index)] = item
        result.applied_updates.append({"op": op, "index": int(index), "progress_item": item.to_dict()})

    def _apply_one_condition_update(
        self,
        update: dict[str, Any],
        result: TaskProgressUpdateResult,
        *,
        current_node_id: str,
    ) -> None:
        op = str(update.get("op", "")).strip().lower()
        item_index = update.get("item_index")
        if current_node_id == "":
            result.skipped_updates.append(
                {"reason": "condition_update_missing_current_node", "update": deepcopy(update)}
            )
            return
        if not isinstance(item_index, int) or isinstance(item_index, bool):
            result.skipped_updates.append(
                {"reason": "missing_integer_item_index", "update": deepcopy(update)}
            )
            return
        if not (0 <= int(item_index) < len(self.items)):
            result.skipped_updates.append(
                {"reason": "item_index_out_of_range", "update": deepcopy(update)}
            )
            return

        current = self.items[int(item_index)]
        conditions = [
            TaskProgressCondition.from_dict(item.to_dict())
            for item in current.conditions
        ]
        error = _apply_progress_condition_operations(
            conditions,
            [update],
            current_node_id=current_node_id,
        )
        if error is not None:
            result.skipped_updates.append(
                {"reason": "invalid_progress_condition", "error": error, "update": deepcopy(update)}
            )
            return
        if [item.to_dict() for item in conditions] == [
            item.to_dict() for item in current.conditions
        ]:
            result.skipped_updates.append(
                {"reason": "progress_condition_unchanged", "update": deepcopy(update)}
            )
            return
        current.conditions = conditions
        result.applied_updates.append(
            {
                "op": op,
                "item_index": int(item_index),
                "conditions": [item.to_dict() for item in conditions],
            }
        )

    def _ensure_verify_candidate_metadata(
        self,
        item: TaskProgressItem,
        *,
        candidate_source_view: dict[str, object] | None,
    ) -> None:
        if item.kind != "verify_candidate":
            return
        candidate = deepcopy(item.candidate)
        if str(candidate.get("candidate_id", "")).strip() == "":
            candidate["candidate_id"] = self._next_candidate_id()
        if not isinstance(candidate.get("source_view"), dict) and candidate_source_view is not None:
            candidate["source_view"] = deepcopy(candidate_source_view)
        item.candidate = candidate

    def _next_candidate_id(self) -> str:
        candidate_id = f"c{int(self.next_candidate_index)}"
        self.next_candidate_index += 1
        return candidate_id


def _apply_progress_condition_operations(
    conditions: list[TaskProgressCondition],
    raw_updates: list[object],
    *,
    current_node_id: str,
) -> str | None:
    for raw_update in raw_updates:
        if not isinstance(raw_update, dict):
            return "progress condition update must be an object"
        op = str(raw_update.get("op", "")).strip().lower()
        if op == "add":
            try:
                content = str(raw_update.get("content", "")).strip()
                if any(item.content.casefold() == content.casefold() for item in conditions):
                    return "progress condition content must be distinct"
                conditions.append(
                    TaskProgressCondition(
                        condition_id=_next_progress_condition_id(conditions),
                        update_node_id=current_node_id,
                        content=content,
                        status=str(raw_update.get("status", "unconfirmed")),
                    )
                )
            except ValueError as exc:
                return str(exc)
            continue
        condition_id = str(
            raw_update.get("condition_id", raw_update.get("knowledge_id", ""))
        ).strip()
        item = next(
            (
                candidate
                for candidate in conditions
                if candidate.condition_id == condition_id
            ),
            None,
        )
        if item is None:
            return f"unknown progress condition id: {condition_id!r}"
        if op == "update":
            status = _normalize_progress_condition_status(raw_update.get("status", ""))
            if status not in PROGRESS_CONDITION_STATUS_VALUES:
                return f"unsupported progress condition status: {status!r}"
            if item.status != status:
                item.status = status
                item.update_node_id = current_node_id
        elif op == "rewrite":
            content = str(raw_update.get("content", "")).strip()
            if content == "":
                return "progress condition content must be non-empty"
            status = _normalize_progress_condition_status(raw_update.get("status", ""))
            if status not in PROGRESS_CONDITION_STATUS_VALUES:
                return f"unsupported progress condition status: {status!r}"
            if any(
                candidate.condition_id != condition_id
                and candidate.content.casefold() == content.casefold()
                for candidate in conditions
            ):
                return "progress condition content must be distinct"
            if item.content != content or item.status != status:
                item.content = content
                item.status = status
                item.update_node_id = current_node_id
        elif op == "remove":
            conditions.remove(item)
        else:
            return f"unsupported progress condition op: {op!r}"
    return None


def _progress_item_from_update(update: dict[str, Any]) -> TaskProgressItem | None:
    candidate = update.get("candidate", {})
    try:
        return TaskProgressItem(
            content=str(update.get("content", "")),
            status=str(update.get("status", "active")),
            result=str(update.get("result", "")),
            kind=str(update.get("kind", "task")),
            candidate=deepcopy(candidate) if isinstance(candidate, dict) else {},
        )
    except ValueError:
        return None


def _bind_new_done_item(item: TaskProgressItem, *, current_node_id: str) -> None:
    node_id = str(current_node_id).strip()
    if node_id == "" or item.status != "done":
        return
    if item.start_node_id == "":
        item.start_node_id = node_id
    if item.completion_node_id == "":
        item.completion_node_id = node_id


def _preserve_and_update_node_binding(
    item: TaskProgressItem,
    *,
    existing: TaskProgressItem,
    current_node_id: str,
) -> None:
    existing_start = str(existing.start_node_id).strip()
    item.start_node_id = existing_start
    if item.status != "done":
        item.completion_node_id = ""
        return
    if item.start_node_id == "":
        item.start_node_id = str(current_node_id).strip()
    existing_completion = str(existing.completion_node_id).strip()
    item.completion_node_id = existing_completion or str(current_node_id).strip()


def _normalize_progress_status(value: object) -> str:
    status = str(value).strip().lower()
    return LEGACY_PROGRESS_STATUS_MAP.get(status, status)


def _normalize_progress_condition_status(value: object) -> str:
    status = str(value).strip().lower()
    return LEGACY_PROGRESS_CONDITION_STATUS_MAP.get(status, status)


def _next_progress_condition_id(
    conditions: list[TaskProgressCondition],
) -> str:
    used = {str(item.condition_id) for item in conditions}
    index = 0
    while f"pc{index}" in used:
        index += 1
    return f"pc{index}"


def _legacy_done_constraint_skip_reason(update: dict[str, Any]) -> str | None:
    if _normalize_progress_status(update.get("status", "")) != "done":
        return None
    missing_constraints = _constraint_text_list(update.get("missing_constraints"))
    if missing_constraints:
        return "done_update_has_missing_constraints"
    if not _constraint_text_list(update.get("satisfied_constraints")):
        return "done_update_missing_satisfied_constraints"
    return None


def _constraint_text_list(value: object) -> list[str]:
    if isinstance(value, str):
        text = value.strip()
        return [] if text == "" else [text]
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip() != ""]


def _merge_candidate(base: dict[str, object], patch: dict[str, object]) -> dict[str, object]:
    merged = deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            nested = deepcopy(merged[key])
            nested.update(deepcopy(value))
            merged[key] = nested
        else:
            merged[key] = deepcopy(value)
    return merged


def _format_candidate_for_prompt(candidate: dict[str, object]) -> str:
    if candidate == {}:
        return ""
    lines = []
    for key in (
        "candidate_id",
        "category",
        "hypothesized_object",
        "actual_object",
    ):
        if key not in candidate:
            continue
        value = candidate[key]
        if value in ("", None, {}, []):
            continue
        lines.append(f"   candidate.{key}: {value}")
    return "\n".join(lines)


def _next_candidate_index_from_items(items: list[TaskProgressItem]) -> int:
    max_index = -1
    for item in items:
        candidate_id = str(item.candidate.get("candidate_id", ""))
        if candidate_id.startswith("c") and candidate_id[1:].isdigit():
            max_index = max(max_index, int(candidate_id[1:]))
    return max_index + 1
