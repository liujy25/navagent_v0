"""Field-selective retrieval from episode graph memory."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
import json
import math
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from navclaw.agent.visual_action_context import ordered_visual_views_left_to_right
from navclaw.agent.visual_action_context import panorama_angle_order_text
from navclaw.agent.visual_action_context import VisualViewContext
from navclaw.agent.vln_landmark_context import _landmark_labels_by_id
from navclaw.agent.vln_landmark_context import _global_landmark_index
from navclaw.agent.visual_policy_prompt_images import _compose_labeled_image_strip
from navclaw.agent.visual_policy_prompt_images import image_content_for_array
from navclaw.agent.visual_policy_prompt_images import image_content_for_current_panorama_strip
from navclaw.agent.visual_policy_prompt_images import image_content_for_movement_history_sheet
from navclaw.mapping.exploration.bev_visuals import place_node_marker_style
from navclaw.mapping.exploration.manager import _densify_path_xy
from navclaw.mapping.exploration.manager import _project_points
from navclaw.mapping.exploration.overlay_drawing import RGB_TRAJECTORY_WIDTH
from navclaw.mapping.exploration.overlay_drawing import _draw_place_node_circle
from navclaw.visualization.action_mode_overlays import BevOverlayTransform
from navclaw.visualization.action_mode_overlays import _edge_display_path_xy
from navclaw.visualization.action_mode_overlays import render_task_progress_bev_overlay_result
from navclaw.visualization.action_mode_overlays import _node_labels_by_id

if TYPE_CHECKING:
    from navclaw.agent.state import NavClawAgentState


@dataclass(frozen=True)
class MemoryIndexEntry:
    ref: str
    kind: str
    floor_id: str
    available_fields: dict[str, int]
    knowledge: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    def to_index_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "ref": str(self.ref),
            "floor_id": str(self.floor_id),
            "knowledge": [str(item) for item in self.knowledge],
            "available_fields": {
                str(name): int(count)
                for name, count in self.available_fields.items()
                if int(count) > 0
            },
            "metadata": deepcopy(self.metadata),
        }
        return payload


@dataclass(frozen=True)
class RetrieveItem:
    ref: str
    fields: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {"ref": str(self.ref), "fields": [str(name) for name in self.fields]}


@dataclass(frozen=True)
class RetrieveRequest:
    query: str
    items: tuple["RetrieveItem", ...]
    retrieval_conclusion: str = ""
    progress_condition_updates: tuple[dict[str, object], ...] = ()
    already_provided_items: tuple["RetrieveItem", ...] = ()

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "query": str(self.query),
            "items": [item.to_dict() for item in self.items],
            "progress_condition_updates": [
                deepcopy(item) for item in self.progress_condition_updates
            ],
        }
        if str(self.retrieval_conclusion).strip() != "":
            payload["retrieval_conclusion"] = str(self.retrieval_conclusion)
        if self.already_provided_items:
            payload["already_provided_items"] = [
                item.to_dict() for item in self.already_provided_items
            ]
        return payload


@dataclass(frozen=True)
class RetrieveRound:
    round_index: int
    request: RetrieveRequest
    source_obs_ids: tuple[str, ...]
    conclusion: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "round_index": int(self.round_index),
            "query": str(self.request.query),
            "items": [item.to_dict() for item in self.request.items],
            "already_provided_items": [
                item.to_dict() for item in self.request.already_provided_items
            ],
            "source_obs_ids": [str(obs_id) for obs_id in self.source_obs_ids],
            "conclusion": str(self.conclusion),
            "progress_condition_updates": [
                deepcopy(item)
                for item in self.request.progress_condition_updates
            ],
        }


@dataclass(frozen=True)
class _EvidencePanel:
    key: str
    label: str
    image_content: dict[str, object]
    source_obs_ids: tuple[str, ...]


@dataclass(frozen=True)
class MaterializedMemoryContext:
    loaded_fields_by_ref: dict[str, list[str]]
    skipped_fields_by_ref: dict[str, list[str]]
    missing_fields_by_ref: dict[str, list[str]]
    source_obs_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "status": "materialized",
            "loaded_fields_by_ref": deepcopy(self.loaded_fields_by_ref),
            "skipped_fields_by_ref": deepcopy(self.skipped_fields_by_ref),
            "missing_fields_by_ref": deepcopy(self.missing_fields_by_ref),
            "source_obs_ids": [str(obs_id) for obs_id in self.source_obs_ids],
        }


@dataclass
class RetrievalWorkspace:
    entries: list[MemoryIndexEntry]
    max_retrieve_rounds: int
    rounds: list[RetrieveRound] = field(default_factory=list)
    text_evidence: dict[str, str] = field(default_factory=dict)
    image_panels: dict[str, _EvidencePanel] = field(default_factory=dict)
    selected_node_ids_by_floor: dict[str, set[str]] = field(default_factory=dict)
    selected_edge_ids_by_floor: dict[str, set[str]] = field(default_factory=dict)
    selected_landmark_ids_by_floor: dict[str, set[str]] = field(default_factory=dict)
    node_display_labels_by_floor: dict[str, dict[str, int]] = field(default_factory=dict)
    landmark_display_labels_by_floor: dict[str, dict[str, int]] = field(default_factory=dict)
    shared_bev_by_floor: dict[str, np.ndarray] = field(default_factory=dict)
    shared_bev_transform_by_floor: dict[str, BevOverlayTransform] = field(default_factory=dict)

    @property
    def retrieve_count(self) -> int:
        return len(self.rounds)

    @property
    def can_retrieve(self) -> bool:
        return self.retrieve_count < int(self.max_retrieve_rounds)

    @property
    def has_pending_evidence(self) -> bool:
        return self.rounds != [] and str(self.rounds[-1].conclusion).strip() == ""

    def conclude_latest_retrieval(self, conclusion: str) -> None:
        text = str(conclusion).strip()
        if text == "":
            raise ValueError("latest retrieval requires a non-empty conclusion")
        if not self.has_pending_evidence:
            raise ValueError("no unconcluded retrieval evidence is available")
        self.rounds[-1] = replace(self.rounds[-1], conclusion=text)

    def discard_concluded_raw_evidence(self) -> None:
        if self.rounds != [] and self.has_pending_evidence:
            raise ValueError("cannot discard raw evidence before concluding it")
        self.text_evidence.clear()
        self.image_panels.clear()

    def clear_materialized_evidence(self) -> None:
        self.text_evidence.clear()
        self.image_panels.clear()
        self.selected_node_ids_by_floor.clear()
        self.selected_edge_ids_by_floor.clear()
        self.selected_landmark_ids_by_floor.clear()
        self.node_display_labels_by_floor.clear()
        self.landmark_display_labels_by_floor.clear()
        self.shared_bev_by_floor.clear()
        self.shared_bev_transform_by_floor.clear()

    def fields_by_ref(
        self,
        *,
        provided_fields_by_ref: dict[str, list[str]] | None = None,
    ) -> dict[str, list[str]]:
        provided = provided_fields_by_ref or {}
        return {
            str(entry.ref): [
                str(name)
                for name, count in entry.available_fields.items()
                if int(count) > 0
                and str(name) not in set(provided.get(str(entry.ref), []))
            ]
            for entry in self.entries
        }

    def retrieval_log_text(self) -> str:
        if self.rounds == []:
            return "none"
        lines: list[str] = []
        for round_record in self.rounds:
            refs = ", ".join(
                f"{item.ref}[{', '.join(item.fields)}]"
                for item in round_record.request.items
            )
            provided_refs = ", ".join(
                f"{item.ref}[{', '.join(item.fields)}]"
                for item in round_record.request.already_provided_items
            )
            provided_text = (
                f"already provided={provided_refs}; "
                if provided_refs != ""
                else ""
            )
            lines.append(
                f"round {round_record.round_index}: "
                f"query={round_record.request.query}; retrieved={refs}; "
                f"{provided_text}"
                f"conclusion={round_record.conclusion or 'pending; raw evidence attached below'}"
            )
        return "\n".join(lines)

    def prompt_content(self) -> list[dict[str, object]]:
        content = self.conclusion_context_content()
        content.extend(self._materialized_field_content())
        return content

    def materialized_context_content(self) -> list[dict[str, object]]:
        content = self._bev_context_content()
        content.extend(self._materialized_field_content())
        return content

    def _materialized_field_content(self) -> list[dict[str, object]]:
        content: list[dict[str, object]] = []
        for key in sorted(self.text_evidence):
            content.append({"type": "text", "text": self.text_evidence[key]})
        for key in sorted(self.image_panels):
            panel = self.image_panels[key]
            content.append({"type": "text", "text": str(panel.label)})
            content.append(deepcopy(panel.image_content))
        return content

    def conclusion_context_content(self) -> list[dict[str, object]]:
        content: list[dict[str, object]] = [
            {
                "type": "text",
                "text": "Retrieval rounds:\n" + self.retrieval_log_text(),
            }
        ]
        content.extend(self._bev_context_content())
        return content

    def _bev_context_content(self) -> list[dict[str, object]]:
        content: list[dict[str, object]] = []
        for floor_id in sorted(self.shared_bev_by_floor):
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"Retrieved BEV for {floor_id}:\n"
                        f"- nodes: {_inline_labeled_refs(self.selected_node_ids_by_floor.get(floor_id, set()), self.node_display_labels_by_floor.get(floor_id, {}))}\n"
                        f"- edges (highlighted executed trajectories): {_inline_edge_refs(self, floor_id)}\n"
                        f"- landmarks: {_inline_labeled_refs(self.selected_landmark_ids_by_floor.get(floor_id, set()), {})}\n"
                        "- circles are node display numbers; landmark squares show the numeric suffix of each landmark ref."
                    ),
                }
            )
            content.append(image_content_for_array(self.shared_bev_by_floor[floor_id]))
        return content

    def to_dict(self) -> dict[str, object]:
        return {
            "enabled": True,
            "max_retrieve_rounds": int(self.max_retrieve_rounds),
            "retrieve_count": int(self.retrieve_count),
            "memory_index": memory_index_dict(self.entries),
            "rounds": [round_record.to_dict() for round_record in self.rounds],
            "workspace": {
                "text_evidence_keys": sorted(self.text_evidence),
                "image_panels": [
                    {
                        "key": panel.key,
                        "label": panel.label,
                        "source_obs_ids": [str(obs_id) for obs_id in panel.source_obs_ids],
                    }
                    for panel in sorted(self.image_panels.values(), key=lambda item: item.key)
                ],
                "shared_bev_floors": sorted(self.shared_bev_by_floor),
                "shared_bev_transform_by_floor": {
                    str(floor_id): transform.to_dict()
                    for floor_id, transform in sorted(
                        self.shared_bev_transform_by_floor.items()
                    )
                },
                "selected_node_ids_by_floor": _sorted_set_mapping(
                    self.selected_node_ids_by_floor
                ),
                "selected_edge_ids_by_floor": _sorted_set_mapping(
                    self.selected_edge_ids_by_floor
                ),
                "selected_landmark_ids_by_floor": _sorted_set_mapping(
                    self.selected_landmark_ids_by_floor
                ),
                "node_display_labels_by_floor": _sorted_label_mapping(
                    self.node_display_labels_by_floor
                ),
                "landmark_display_labels_by_floor": _sorted_label_mapping(
                    self.landmark_display_labels_by_floor
                ),
            },
        }


def build_memory_index(
    state: "NavClawAgentState",
) -> list[MemoryIndexEntry]:
    candidates = _landmark_candidates(state)
    landmark_ids_by_node = _landmark_ids_by_node(state=state, candidates=candidates)
    node_ids_by_landmark = _node_ids_by_landmark(state=state, candidates=candidates)
    entries: list[MemoryIndexEntry] = []
    for node in sorted(state.graph.iter_nodes(), key=lambda item: _reference_sort_key(str(item.id))):
        landmark_ids = sorted(
            landmark_ids_by_node.get(str(node.id), set()),
            key=_reference_sort_key,
        )
        entries.append(
            MemoryIndexEntry(
                ref=str(node.id),
                kind="node",
                floor_id=str(node.floor_id),
                available_fields={
                    "rgb": len(node.obs_ids),
                    "landmarks": len(landmark_ids),
                },
                knowledge=tuple(str(item.content) for item in node.knowledge),
                metadata={"associated_landmark_refs": landmark_ids},
            )
        )
    for edge in sorted(state.graph.iter_edges(), key=lambda item: _reference_sort_key(str(item.id))):
        try:
            src_node = state.graph.get_node(str(edge.src_id))
            floor_id = str(src_node.floor_id)
        except KeyError:
            src_node = None
            floor_id = ""
        entries.append(
            MemoryIndexEntry(
                ref=str(edge.id),
                kind="edge",
                floor_id=floor_id,
                available_fields={
                    "rgb": int(
                        src_node is not None
                        and len(src_node.obs_ids) > 0
                        and len(edge.path_xy) >= 2
                    ),
                    "trajectory": int(len(edge.path_xy) >= 2),
                    "movement_rgb": len(edge.rgb_history_obs_ids),
                },
                knowledge=(
                    f"Actual {str(edge.relation or 'move')} edge from "
                    f"{edge.src_id} to {edge.dst_id}.",
                    *(str(item.content) for item in edge.knowledge),
                ),
                metadata={
                    "src_node_id": str(edge.src_id),
                    "dst_node_id": str(edge.dst_id),
                    "traversal_count": int(edge.traversal_count),
                },
            )
        )
    for candidate in candidates:
        landmark_id = str(candidate.get("landmark_id", ""))
        associated_nodes = sorted(
            {
                str(node_id)
                for node_id in node_ids_by_landmark.get(landmark_id, set())
            },
            key=_reference_sort_key,
        )
        detections = _candidate_detections(candidate)
        if detections == []:
            continue
        floor_id = _landmark_floor_id(
            state=state,
            associated_node_ids=associated_nodes,
        )
        entries.append(
            MemoryIndexEntry(
                ref=landmark_id,
                kind="landmark",
                floor_id=floor_id,
                available_fields={"rgb": len(detections)},
                knowledge=(
                    str(candidate.get("class_name", "unknown landmark")),
                    *(
                        str(item.get("content", "")).strip()
                        for item in list(candidate.get("knowledge", []))
                        if isinstance(item, dict)
                        and str(item.get("content", "")).strip() != ""
                    ),
                ),
                metadata={"associated_node_refs": associated_nodes},
            )
        )
    return sorted(entries, key=lambda entry: (_kind_rank(entry.kind), _reference_sort_key(entry.ref)))


def memory_index_dict(
    entries: list[MemoryIndexEntry],
    *,
    provided_fields_by_ref: dict[str, list[str]] | None = None,
) -> dict[str, object]:
    groups: dict[str, dict[str, object]] = {
        "nodes": {},
        "edges": {},
        "landmarks": {},
    }
    group_by_kind = {"node": "nodes", "edge": "edges", "landmark": "landmarks"}
    provided = provided_fields_by_ref or {}
    for entry in entries:
        payload = entry.to_index_dict()
        available_fields = dict(payload["available_fields"])
        provided_fields: dict[str, int] = {}
        for field_name in provided.get(str(entry.ref), []):
            count = available_fields.pop(str(field_name), None)
            if count is not None:
                provided_fields[str(field_name)] = int(count)
        payload["available_fields"] = available_fields
        if provided_fields:
            payload["provided_fields"] = provided_fields
        groups[group_by_kind[str(entry.kind)]][str(entry.ref)] = payload
    return groups


def memory_index_text(
    entries: list[MemoryIndexEntry],
    *,
    provided_fields_by_ref: dict[str, list[str]] | None = None,
) -> str:
    prompt_index = memory_index_dict(
        entries,
        provided_fields_by_ref=provided_fields_by_ref,
    )
    floor_ids = {str(entry.floor_id) for entry in entries}
    single_floor_id = next(iter(floor_ids)) if len(floor_ids) == 1 else ""
    for group_name, group in prompt_index.items():
        for payload in group.values():
            if single_floor_id != "":
                payload.pop("floor_id", None)
            if payload.get("knowledge") == []:
                payload.pop("knowledge", None)
            metadata = payload.get("metadata")
            if isinstance(metadata, dict):
                if group_name == "edges":
                    metadata.pop("src_node_id", None)
                    metadata.pop("dst_node_id", None)
                elif group_name == "landmarks":
                    metadata.pop("associated_node_refs", None)
                if metadata == {}:
                    payload.pop("metadata", None)
    if single_floor_id != "":
        prompt_index = {"floor_id": single_floor_id, **prompt_index}
    return json.dumps(
        {"memory_index": prompt_index},
        ensure_ascii=False,
        indent=2,
    )


def normalize_retrieve_request(
    payload: dict[str, object],
    *,
    fields_by_ref: dict[str, list[str]],
    provided_fields_by_ref: dict[str, list[str]] | None = None,
    require_retrieval_conclusion: bool = False,
) -> RetrieveRequest:
    retrieval_conclusion = str(payload.get("retrieval_conclusion", "")).strip()
    if require_retrieval_conclusion and retrieval_conclusion == "":
        raise ValueError(
            "task progress updater must conclude the latest retrieval before "
            f"requesting another one: {payload!r}"
        )
    raw_retrieve = payload.get("retrieve")
    if not isinstance(raw_retrieve, dict):
        raise ValueError(f"retrieve decision requires a retrieve object: {payload!r}")
    query = str(raw_retrieve.get("query", "")).strip()
    if query == "":
        raise ValueError(f"retrieve query must be non-empty: {payload!r}")
    raw_items = raw_retrieve.get("items")
    if not isinstance(raw_items, list) or raw_items == []:
        raise ValueError(f"retrieve items must be a non-empty list: {payload!r}")
    provided = provided_fields_by_ref or {}
    items: list[RetrieveItem] = []
    already_provided_items: list[RetrieveItem] = []
    for raw_item in raw_items:
        if not isinstance(raw_item, dict):
            raise ValueError(f"retrieve item must be an object: {raw_item!r}")
        ref = _canonical_retrieve_ref(
            raw_item.get("ref", ""),
            fields_by_ref=fields_by_ref,
        )
        allowed_fields = fields_by_ref.get(ref)
        if allowed_fields is None:
            raise ValueError(f"unknown retrieve reference: {ref!r}")
        raw_fields = raw_item.get("fields")
        if not isinstance(raw_fields, list) or raw_fields == []:
            raise ValueError(f"retrieve fields must be a non-empty list: {raw_item!r}")
        fields: list[str] = []
        already_provided_fields: list[str] = []
        for raw_field in raw_fields:
            name = str(raw_field).strip()
            if name in allowed_fields:
                if name not in fields:
                    fields.append(name)
                continue
            if name in provided.get(ref, []):
                if name not in already_provided_fields:
                    already_provided_fields.append(name)
                continue
            raise ValueError(
                f"field {name!r} is unavailable for {ref!r}; "
                f"available={allowed_fields!r}; "
                f"provided={provided.get(ref, [])!r}"
            )
        if fields:
            items.append(RetrieveItem(ref=ref, fields=tuple(fields)))
        if already_provided_fields:
            already_provided_items.append(
                RetrieveItem(ref=ref, fields=tuple(already_provided_fields))
            )
    if items == []:
        raise ValueError(
            "all requested fields are already attached in the current planning "
            "observation; use that evidence directly or request an available "
            "historical field"
        )
    return RetrieveRequest(
        query=query,
        items=tuple(items),
        already_provided_items=tuple(already_provided_items),
        retrieval_conclusion=retrieval_conclusion,
    )


def _canonical_retrieve_ref(
    raw_ref: object,
    *,
    fields_by_ref: dict[str, list[str]],
) -> str:
    ref = str(raw_ref).strip()
    if ref in fields_by_ref:
        return ref
    for group_prefix, entity_prefix in (
        ("nodes.", "n"),
        ("edges.", "e"),
        ("landmarks.", "landmark_"),
    ):
        if not ref.startswith(group_prefix):
            continue
        candidate = ref[len(group_prefix) :]
        if candidate.startswith(entity_prefix) and candidate in fields_by_ref:
            return candidate
    return ref


def execute_retrieve_request(
    *,
    state: "NavClawAgentState",
    workspace: RetrievalWorkspace,
    request: RetrieveRequest,
) -> RetrieveRound:
    if not workspace.can_retrieve:
        raise ValueError(
            "retrieve budget exhausted: "
            f"{workspace.retrieve_count}/{workspace.max_retrieve_rounds}"
        )
    source_obs_ids = _materialize_request(
        state=state,
        workspace=workspace,
        request=request,
    )
    _render_shared_bev(state=state, workspace=workspace)
    record = RetrieveRound(
        round_index=int(workspace.retrieve_count),
        request=RetrieveRequest(
            query=request.query,
            items=request.items,
            already_provided_items=request.already_provided_items,
            progress_condition_updates=request.progress_condition_updates,
        ),
        source_obs_ids=tuple(source_obs_ids),
    )
    workspace.rounds.append(record)
    return record


def materialize_memory_context(
    *,
    state: "NavClawAgentState",
    workspace: RetrievalWorkspace,
    entries: list[MemoryIndexEntry],
    provided_fields_by_ref: dict[str, list[str]] | None = None,
) -> MaterializedMemoryContext:
    entry_refs = {str(entry.ref) for entry in workspace.entries}
    unknown_refs = [
        str(entry.ref)
        for entry in entries
        if str(entry.ref) not in entry_refs
    ]
    if unknown_refs:
        raise ValueError(
            f"memory context entries are absent from workspace: {unknown_refs}"
        )
    provided = provided_fields_by_ref or {}
    loaded_items: list[RetrieveItem] = []
    skipped_items: list[RetrieveItem] = []
    for entry in entries:
        available_fields = [
            str(name)
            for name, count in entry.available_fields.items()
            if int(count) > 0
        ]
        provided_fields = set(provided.get(str(entry.ref), []))
        skipped_fields = [
            name
            for name in available_fields
            if name in provided_fields
        ]
        loaded_fields = [name for name in available_fields if name not in skipped_fields]
        loaded_items.append(
            RetrieveItem(ref=str(entry.ref), fields=tuple(loaded_fields))
        )
        if skipped_fields:
            skipped_items.append(
                RetrieveItem(ref=str(entry.ref), fields=tuple(skipped_fields))
            )

    workspace.clear_materialized_evidence()
    source_obs_ids = _materialize_request(
        state=state,
        workspace=workspace,
        request=RetrieveRequest(
            query="System-preloaded memory context.",
            items=tuple(loaded_items),
        ),
    )
    _render_shared_bev(state=state, workspace=workspace)
    actually_loaded: list[RetrieveItem] = []
    missing_items: list[RetrieveItem] = []
    for item in loaded_items:
        present_fields = [
            field_name
            for field_name in item.fields
            if f"{item.ref}.{field_name}" in workspace.text_evidence
            or f"{item.ref}.{field_name}" in workspace.image_panels
        ]
        missing_fields = [
            field_name for field_name in item.fields if field_name not in present_fields
        ]
        if present_fields:
            actually_loaded.append(
                RetrieveItem(ref=item.ref, fields=tuple(present_fields))
            )
        if missing_fields:
            missing_items.append(
                RetrieveItem(ref=item.ref, fields=tuple(missing_fields))
            )
    return MaterializedMemoryContext(
        loaded_fields_by_ref=_fields_by_ref_from_items(actually_loaded),
        skipped_fields_by_ref=_fields_by_ref_from_items(skipped_items),
        missing_fields_by_ref=_fields_by_ref_from_items(missing_items),
        source_obs_ids=tuple(source_obs_ids),
    )


def _fields_by_ref_from_items(
    items: list[RetrieveItem],
) -> dict[str, list[str]]:
    return {
        str(item.ref): [str(field_name) for field_name in item.fields]
        for item in items
        if item.fields
    }


def _materialize_request(
    *,
    state: "NavClawAgentState",
    workspace: RetrievalWorkspace,
    request: RetrieveRequest,
) -> list[str]:
    entry_by_ref = {str(entry.ref): entry for entry in workspace.entries}
    candidates_by_id = {
        str(candidate.get("landmark_id", "")): candidate
        for candidate in _landmark_candidates(state)
    }
    source_obs_ids: list[str] = []
    for item in request.items:
        entry = entry_by_ref[str(item.ref)]
        if entry.kind == "node":
            node = state.graph.get_node(str(item.ref))
            _select_node(workspace, node_id=str(node.id), floor_id=str(node.floor_id))
            for field_name in item.fields:
                key = f"{item.ref}.{field_name}"
                if field_name == "rgb":
                    obs_ids = [str(obs_id) for obs_id in node.obs_ids]
                    _add_node_rgb_panel(
                        state=state,
                        workspace=workspace,
                        key=key,
                        node_id=str(node.id),
                        obs_ids=obs_ids,
                    )
                    _extend_unique(source_obs_ids, obs_ids)
                elif field_name == "landmarks":
                    obs_ids = _add_node_landmark_panel(
                        state=state,
                        workspace=workspace,
                        key=key,
                        node_id=str(node.id),
                        floor_id=str(node.floor_id),
                        candidates_by_id=candidates_by_id,
                    )
                    _extend_unique(source_obs_ids, obs_ids)
        elif entry.kind == "edge":
            edge = _edge_by_id(state, str(item.ref))
            src_node = state.graph.get_node(str(edge.src_id))
            dst_node = state.graph.get_node(str(edge.dst_id))
            floor_id = str(src_node.floor_id)
            _select_node(workspace, node_id=str(src_node.id), floor_id=floor_id)
            _select_node(workspace, node_id=str(dst_node.id), floor_id=str(dst_node.floor_id))
            workspace.selected_edge_ids_by_floor.setdefault(floor_id, set()).add(str(edge.id))
            for field_name in item.fields:
                key = f"{item.ref}.{field_name}"
                if field_name == "trajectory":
                    workspace.text_evidence.setdefault(
                        key,
                        (
                            f"Edge {edge.id} actual trajectory is highlighted on the shared BEV; "
                            f"path_xy_points={len(edge.path_xy)}."
                        ),
                    )
                elif field_name == "rgb":
                    obs_ids = _add_edge_rgb_panel(
                        state=state,
                        workspace=workspace,
                        key=key,
                        edge=edge,
                        src_node=src_node,
                        dst_node=dst_node,
                    )
                    _extend_unique(source_obs_ids, obs_ids)
                elif field_name == "movement_rgb":
                    obs_ids = _movement_keyframes(
                        state=state,
                        obs_ids=[str(obs_id) for obs_id in edge.rgb_history_obs_ids],
                    )
                    _add_movement_panel(
                        state=state,
                        workspace=workspace,
                        key=key,
                        edge_id=str(edge.id),
                        obs_ids=obs_ids,
                    )
                    _extend_unique(source_obs_ids, obs_ids)
        elif entry.kind == "landmark":
            candidate = candidates_by_id.get(str(item.ref))
            if candidate is None:
                raise ValueError(f"missing landmark candidate for memory index ref {item.ref!r}")
            associated_nodes = list(entry.metadata.get("associated_node_refs", []))
            floor_id = str(entry.floor_id or state.system.current_floor_id)
            workspace.selected_landmark_ids_by_floor.setdefault(floor_id, set()).add(str(item.ref))
            for node_id in associated_nodes:
                node = state.graph.get_node(str(node_id))
                _select_node(workspace, node_id=str(node.id), floor_id=str(node.floor_id))
            for field_name in item.fields:
                key = f"{item.ref}.{field_name}"
                if field_name == "rgb":
                    obs_ids = _add_landmark_rgb_panel(
                        state=state,
                        workspace=workspace,
                        key=key,
                        candidate=candidate,
                    )
                    _extend_unique(source_obs_ids, obs_ids)
    return source_obs_ids


def _render_shared_bev(
    *,
    state: "NavClawAgentState",
    workspace: RetrievalWorkspace,
) -> None:
    candidates_by_id = {
        str(candidate.get("landmark_id", "")): candidate
        for candidate in _landmark_candidates(state)
    }
    floor_ids = set(workspace.selected_node_ids_by_floor)
    floor_ids.update(workspace.selected_edge_ids_by_floor)
    floor_ids.update(workspace.selected_landmark_ids_by_floor)
    for floor_id in sorted(floor_ids):
        floor_nodes = list(state.graph.iter_nodes(floor_id=str(floor_id)))
        all_node_labels = _node_labels_by_id(floor_nodes)
        workspace.node_display_labels_by_floor[floor_id] = {
            node_id: int(all_node_labels[node_id])
            for node_id in workspace.selected_node_ids_by_floor.get(floor_id, set())
            if node_id in all_node_labels
        }
        all_landmark_labels = _landmark_labels_by_id(
            landmark_controller=state.landmark_controller,
            graph=state.graph,
            floor_id=str(floor_id),
        )
        workspace.landmark_display_labels_by_floor[floor_id] = {
            landmark_id: int(all_landmark_labels[landmark_id])
            for landmark_id in workspace.selected_landmark_ids_by_floor.get(floor_id, set())
            if landmark_id in all_landmark_labels
        }
        landmark_markers: list[dict[str, object]] = []
        for landmark_id in sorted(
            workspace.selected_landmark_ids_by_floor.get(floor_id, set()),
            key=_reference_sort_key,
        ):
            candidate = candidates_by_id.get(landmark_id)
            if candidate is None:
                continue
            position = candidate.get("position")
            if not isinstance(position, dict) or position.get("x") is None or position.get("y") is None:
                continue
            landmark_markers.append(
                {
                    "label": int(all_landmark_labels[landmark_id]),
                    "xy": [float(position["x"]), float(position["y"])],
                }
            )
        render_result = render_task_progress_bev_overlay_result(
            graph=state.graph,
            global_exploration=state.global_exploration_for_floor(floor_id),
            floor_id=floor_id,
            landmark_markers=landmark_markers,
            show_graph=True,
            node_ids=set(workspace.selected_node_ids_by_floor.get(floor_id, set())),
            edge_ids=set(workspace.selected_edge_ids_by_floor.get(floor_id, set())),
        )
        workspace.shared_bev_by_floor[floor_id] = render_result.image
        workspace.shared_bev_transform_by_floor[floor_id] = render_result.transform


def _add_node_rgb_panel(
    *,
    state: "NavClawAgentState",
    workspace: RetrievalWorkspace,
    key: str,
    node_id: str,
    obs_ids: list[str],
) -> None:
    if key in workspace.image_panels or obs_ids == []:
        return
    views = _views_for_obs_ids(obs_ids)
    angles = [int(view.angle_deg) for view in views]
    workspace.image_panels[key] = _EvidencePanel(
        key=key,
        label=(
            f"Retrieved node {node_id} panorama RGB.\n"
            f"- Views are ordered left-to-right as {panorama_angle_order_text(angles)}.\n"
            f"- Its angle labels use the mapping above relative to the stored heading of node {node_id}."
        ),
        image_content=image_content_for_current_panorama_strip(
            cache=state.cache,
            views=views,
            include_visited_nodes=False,
        ),
        source_obs_ids=tuple(obs_ids),
    )


def _add_node_landmark_panel(
    *,
    state: "NavClawAgentState",
    workspace: RetrievalWorkspace,
    key: str,
    node_id: str,
    floor_id: str,
    candidates_by_id: dict[str, dict[str, object]],
) -> list[str]:
    node = state.graph.get_node(node_id)
    node_obs_ids = {str(obs_id) for obs_id in node.obs_ids}
    selected_candidates = [
        candidate
        for candidate in candidates_by_id.values()
        if any(
            str(detection.get("obs_id", "")) in node_obs_ids
            for detection in _candidate_detections(candidate)
        )
    ]
    for candidate in selected_candidates:
        workspace.selected_landmark_ids_by_floor.setdefault(floor_id, set()).add(
            str(candidate.get("landmark_id", ""))
        )
    if key in workspace.image_panels or selected_candidates == []:
        return []
    obs_ids = [str(obs_id) for obs_id in node.obs_ids]
    ordered_views = ordered_visual_views_left_to_right(_views_for_obs_ids(obs_ids))
    ordered_obs_ids = [str(view.obs_id) for view in ordered_views]
    angles = [int(view.angle_deg) for view in ordered_views]
    image = _annotated_landmark_sheet(
        state=state,
        obs_ids=ordered_obs_ids,
        candidates=selected_candidates,
        image_labels=[f"angle_{angle}" for angle in angles],
    )
    workspace.image_panels[key] = _EvidencePanel(
        key=key,
        label=(
            f"Retrieved landmark detections on node {node_id} panorama RGB.\n"
            f"- Views are ordered left-to-right as {panorama_angle_order_text(angles)}.\n"
            f"- Its angle labels use the mapping above relative to the stored heading of node {node_id}."
        ),
        image_content=image_content_for_array(image),
        source_obs_ids=tuple(obs_ids),
    )
    return obs_ids


def _add_landmark_rgb_panel(
    *,
    state: "NavClawAgentState",
    workspace: RetrievalWorkspace,
    key: str,
    candidate: dict[str, object],
) -> list[str]:
    detections = _candidate_detections(candidate)
    obs_ids: list[str] = []
    for detection in detections:
        obs_id = str(detection.get("obs_id", ""))
        if obs_id != "" and obs_id not in obs_ids:
            obs_ids.append(obs_id)
    if key in workspace.image_panels or obs_ids == []:
        return obs_ids
    image = _annotated_landmark_sheet(
        state=state,
        obs_ids=obs_ids,
        candidates=[candidate],
    )
    workspace.image_panels[key] = _EvidencePanel(
        key=key,
        label=f"Retrieved RGB evidence for {candidate.get('landmark_id', '')}.",
        image_content=image_content_for_array(image),
        source_obs_ids=tuple(obs_ids),
    )
    return obs_ids


def _add_movement_panel(
    *,
    state: "NavClawAgentState",
    workspace: RetrievalWorkspace,
    key: str,
    edge_id: str,
    obs_ids: list[str],
) -> None:
    if key in workspace.image_panels:
        return
    content = image_content_for_movement_history_sheet(cache=state.cache, obs_ids=obs_ids)
    if content is None:
        return
    workspace.image_panels[key] = _EvidencePanel(
        key=key,
        label=f"Retrieved chronological movement RGB for edge {edge_id}.",
        image_content=content,
        source_obs_ids=tuple(obs_ids),
    )


def _add_edge_rgb_panel(
    *,
    state: "NavClawAgentState",
    workspace: RetrievalWorkspace,
    key: str,
    edge,
    src_node,
    dst_node,
) -> list[str]:
    if key in workspace.image_panels:
        return list(workspace.image_panels[key].source_obs_ids)
    rendered = _render_edge_rgb_overlay(
        state=state,
        edge=edge,
        src_node=src_node,
        dst_node=dst_node,
    )
    if rendered is None:
        return []
    obs_id, image = rendered
    workspace.image_panels[key] = _EvidencePanel(
        key=key,
        label=(
            f"Retrieved edge {edge.id} start-view RGB from {src_node.id} toward "
            f"{dst_node.id}; the blue line is the executed path and the numbered "
            "node marker is the endpoint."
        ),
        image_content=image_content_for_array(image),
        source_obs_ids=(str(obs_id),),
    )
    return [str(obs_id)]


def _render_edge_rgb_overlay(
    *,
    state: "NavClawAgentState",
    edge,
    src_node,
    dst_node,
) -> tuple[str, np.ndarray] | None:
    display_path_xy = _edge_display_path_xy(
        edge=edge,
        src_xy=src_node.position[:2],
        dst_xy=dst_node.position[:2],
    )
    path_xy = _densify_path_xy(display_path_xy)
    if len(path_xy) < 2:
        return None
    path_z = _interpolated_edge_path_z(
        path_xy=path_xy,
        src_z=float(src_node.position[2]),
        dst_z=float(dst_node.position[2]),
    )
    path_xyz = np.column_stack([path_xy, path_z]).astype(np.float32)
    selected: tuple[tuple[float, int], str, np.ndarray, np.ndarray] | None = None
    for obs_id in src_node.obs_ids:
        observation = state.cache.get_observation(str(obs_id)).observation
        if (
            observation.rgb is None
            or observation.intrinsics is None
            or observation.T_cam_odom is None
        ):
            continue
        rgb = np.asarray(observation.rgb, dtype=np.uint8)
        image_height, image_width = rgb.shape[:2]
        projected, _depths, valid = _project_points(
            points_odom=path_xyz,
            T_cam_odom=np.asarray(observation.T_cam_odom, dtype=np.float32),
            intrinsics=np.asarray(observation.intrinsics, dtype=np.float32),
            image_width=image_width,
            image_height=image_height,
        )
        if not bool(valid[-1]):
            continue
        endpoint = projected[-1]
        center_distance_sq = float(
            (float(endpoint[0]) - (float(image_width) - 1.0) / 2.0) ** 2
            + (float(endpoint[1]) - (float(image_height) - 1.0) / 2.0) ** 2
        )
        score = (-center_distance_sq, int(np.count_nonzero(valid)))
        if selected is None or score > selected[0]:
            selected = (score, str(obs_id), rgb, np.column_stack([projected, valid]))
    if selected is None:
        return None

    _score, obs_id, rgb, projection_with_valid = selected
    projected = projection_with_valid[:, :2]
    valid = projection_with_valid[:, 2].astype(bool)
    image = Image.fromarray(rgb).convert("RGBA")
    trajectory_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    trajectory_draw = ImageDraw.Draw(trajectory_layer, "RGBA")
    node_style = place_node_marker_style(
        image_width=int(image.size[0]),
        image_height=int(image.size[1]),
        mode="rgb",
    )
    for run in _contiguous_projected_runs(projected=projected, valid=valid):
        trajectory_draw.line(
            run,
            fill=node_style.fill,
            width=RGB_TRAJECTORY_WIDTH,
            joint="curve",
        )
    image.alpha_composite(trajectory_layer)
    _draw_place_node_circle(
        draw=ImageDraw.Draw(image, "RGBA"),
        center_xy=(float(projected[-1, 0]), float(projected[-1, 1])),
        label=str(dst_node.id),
        font=ImageFont.load_default(),
        image_size=(int(image.size[0]), int(image.size[1])),
        image=image,
    )
    return obs_id, np.asarray(image.convert("RGB"), dtype=np.uint8)


def _interpolated_edge_path_z(
    *,
    path_xy: np.ndarray,
    src_z: float,
    dst_z: float,
) -> np.ndarray:
    segment_lengths = np.linalg.norm(np.diff(path_xy, axis=0), axis=1)
    cumulative = np.concatenate(
        [np.zeros((1,), dtype=np.float32), np.cumsum(segment_lengths, dtype=np.float32)]
    )
    total = float(cumulative[-1])
    if total <= 1e-6:
        return np.full((len(path_xy),), float(src_z), dtype=np.float32)
    ratios = cumulative / total
    return np.asarray(float(src_z) + ratios * (float(dst_z) - float(src_z)), dtype=np.float32)


def _contiguous_projected_runs(
    *,
    projected: np.ndarray,
    valid: np.ndarray,
) -> list[list[tuple[float, float]]]:
    runs: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    for point, is_valid in zip(projected, valid):
        if bool(is_valid):
            current.append((float(point[0]), float(point[1])))
            continue
        if len(current) >= 2:
            runs.append(current)
        current = []
    if len(current) >= 2:
        runs.append(current)
    return runs


def _annotated_landmark_sheet(
    *,
    state: "NavClawAgentState",
    obs_ids: list[str],
    candidates: list[dict[str, object]],
    image_labels: list[str] | None = None,
) -> np.ndarray:
    images: list[np.ndarray] = []
    labels: list[str] = []
    for index, obs_id in enumerate(obs_ids):
        raw = np.asarray(state.cache.get_observation(str(obs_id)).observation.rgb, dtype=np.uint8)
        canvas = Image.fromarray(raw).convert("RGB")
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default()
        for candidate in candidates:
            landmark_id = str(candidate.get("landmark_id", ""))
            display_label = str(_global_landmark_index(landmark_id))
            for detection in _candidate_detections(candidate):
                if str(detection.get("obs_id", "")) != str(obs_id):
                    continue
                bbox = list(detection.get("bbox", []))
                if len(bbox) != 4:
                    continue
                box = tuple(float(value) for value in bbox)
                draw.rectangle(box, outline=(235, 67, 53), width=4)
                draw.text(
                    (box[0] + 2, max(0.0, box[1] - 12)),
                    display_label,
                    fill=(235, 67, 53),
                    font=font,
                )
        images.append(np.asarray(canvas, dtype=np.uint8))
        labels.append(
            str(image_labels[index])
            if image_labels is not None and index < len(image_labels)
            else str(obs_id)
        )
    return _compose_labeled_image_strip(images=images, labels=labels)


def _views_for_obs_ids(obs_ids: list[str]) -> list[VisualViewContext]:
    count = len(obs_ids)
    return [
        VisualViewContext(
            angle_deg=int(round(float(index) * 360.0 / float(count))) % 360,
            obs_id=str(obs_id),
            rgb_id=f"{obs_id}:rgb",
            depth_id=f"{obs_id}:depth",
        )
        for index, obs_id in enumerate(obs_ids)
    ]


def _movement_keyframes(
    *,
    state: "NavClawAgentState",
    obs_ids: list[str],
) -> list[str]:
    if len(obs_ids) <= 2:
        return list(obs_ids)
    observations = [state.cache.get_observation(obs_id).observation for obs_id in obs_ids]
    selected = [0]
    last_kept = observations[0].pose
    previous = observations[0].pose
    distance = 0.0
    for index in range(1, len(observations) - 1):
        pose = observations[index].pose
        distance += math.sqrt(
            (float(pose.x) - float(previous.x)) ** 2
            + (float(pose.y) - float(previous.y)) ** 2
            + (float(pose.z) - float(previous.z)) ** 2
        )
        yaw_difference = abs((float(pose.yaw) - float(last_kept.yaw) + 180.0) % 360.0 - 180.0)
        if distance >= 0.8 or yaw_difference >= 45.0:
            selected.append(index)
            last_kept = pose
            distance = 0.0
        previous = pose
    selected.append(len(obs_ids) - 1)
    return [str(obs_ids[index]) for index in selected]


def _select_node(workspace: RetrievalWorkspace, *, node_id: str, floor_id: str) -> None:
    workspace.selected_node_ids_by_floor.setdefault(str(floor_id), set()).add(str(node_id))


def _edge_by_id(state: "NavClawAgentState", edge_id: str):
    for edge in state.graph.iter_edges():
        if str(edge.id) == str(edge_id):
            return edge
    raise ValueError(f"unknown graph edge: {edge_id!r}")


def _landmark_candidates(state: "NavClawAgentState") -> list[dict[str, object]]:
    context = state.landmark_controller.buffer.to_context()
    candidates: list[dict[str, object]] = []
    for key in ("confirmed_landmarks", "needs_verification_candidates", "pending_candidates"):
        candidates.extend(
            dict(item)
            for item in list(context.get(key, []))
            if isinstance(item, dict)
        )
    return sorted(
        candidates,
        key=lambda item: _reference_sort_key(str(item.get("landmark_id", ""))),
    )


def _candidate_detections(candidate: dict[str, object]) -> list[dict[str, object]]:
    return [
        dict(item)
        for item in list(candidate.get("detections", []))
        if isinstance(item, dict)
    ]


def _landmark_ids_by_node(
    *,
    state: "NavClawAgentState",
    candidates: list[dict[str, object]],
) -> dict[str, set[str]]:
    result = {str(node.id): set() for node in state.graph.iter_nodes()}
    for candidate in candidates:
        landmark_id = str(candidate.get("landmark_id", ""))
        detection_obs_ids = {
            str(detection.get("obs_id", ""))
            for detection in _candidate_detections(candidate)
        }
        for node in state.graph.iter_nodes():
            if detection_obs_ids.intersection(str(obs_id) for obs_id in node.obs_ids):
                result[str(node.id)].add(landmark_id)
    return result


def _node_ids_by_landmark(
    *,
    state: "NavClawAgentState",
    candidates: list[dict[str, object]],
) -> dict[str, set[str]]:
    by_node = _landmark_ids_by_node(state=state, candidates=candidates)
    result: dict[str, set[str]] = {}
    for node_id, landmark_ids in by_node.items():
        for landmark_id in landmark_ids:
            result.setdefault(landmark_id, set()).add(node_id)
    return result


def _landmark_floor_id(
    *,
    state: "NavClawAgentState",
    associated_node_ids: list[str],
) -> str:
    if associated_node_ids != []:
        return str(state.graph.get_node(str(associated_node_ids[0])).floor_id)
    return str(state.system.current_floor_id)


def _kind_rank(kind: str) -> int:
    return {"node": 0, "edge": 1, "landmark": 2}.get(str(kind), 3)


def _reference_sort_key(value: str) -> tuple[str, int, str]:
    text = str(value)
    prefix, separator, suffix = text.rpartition("_")
    if separator == "":
        prefix = "".join(character for character in text if not character.isdigit())
        suffix = text[len(prefix) :]
    try:
        index = int(suffix)
    except ValueError:
        index = 10**9
    return prefix, index, text


def _sorted_set_mapping(value: dict[str, set[str]]) -> dict[str, list[str]]:
    return {
        str(key): sorted(items, key=_reference_sort_key)
        for key, items in sorted(value.items())
    }


def _sorted_label_mapping(
    value: dict[str, dict[str, int]],
) -> dict[str, dict[str, int]]:
    return {
        str(floor_id): {
            str(ref): int(label)
            for ref, label in sorted(
                labels.items(),
                key=lambda item: _reference_sort_key(str(item[0])),
            )
        }
        for floor_id, labels in sorted(value.items())
    }


def _inline_labeled_refs(values: set[str], labels_by_ref: dict[str, int]) -> str:
    ordered = sorted((str(value) for value in values), key=_reference_sort_key)
    if ordered == []:
        return "none"
    return ", ".join(
        (
            f"{labels_by_ref[ref]}={ref}"
            if ref in labels_by_ref
            else ref
        )
        for ref in ordered
    )


def _inline_edge_refs(workspace: RetrievalWorkspace, floor_id: str) -> str:
    edge_refs = sorted(
        workspace.selected_edge_ids_by_floor.get(str(floor_id), set()),
        key=_reference_sort_key,
    )
    if edge_refs == []:
        return "none"
    entries_by_ref = {entry.ref: entry for entry in workspace.entries}
    values: list[str] = []
    for ref in edge_refs:
        entry = entries_by_ref.get(str(ref))
        metadata = {} if entry is None else entry.metadata
        src = str(metadata.get("src_node_id", ""))
        dst = str(metadata.get("dst_node_id", ""))
        values.append(f"{ref} ({src} -> {dst})" if src != "" and dst != "" else str(ref))
    return ", ".join(values)


def _extend_unique(target: list[str], values: list[str]) -> None:
    for value in values:
        text = str(value)
        if text != "" and text not in target:
            target.append(text)
