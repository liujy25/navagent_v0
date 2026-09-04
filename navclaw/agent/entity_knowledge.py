from __future__ import annotations

from dataclasses import dataclass
import json
from typing import TYPE_CHECKING

from navclaw.memory.entity_knowledge import EntityKnowledge
from navclaw.memory.entity_knowledge import next_entity_knowledge_id

if TYPE_CHECKING:
    from navclaw.agent.episodic_retrieval import RetrievalWorkspace
    from navclaw.agent.state import NavClawAgentState
    from navclaw.llm.client import LLMClient


@dataclass(frozen=True)
class EntityKnowledgeUpdate:
    op: str
    ref: str
    knowledge_id: str = ""
    content: str = ""

    def to_dict(self) -> dict[str, str]:
        payload = {"op": str(self.op), "ref": str(self.ref)}
        if self.knowledge_id != "":
            payload["knowledge_id"] = str(self.knowledge_id)
        if self.content != "":
            payload["content"] = str(self.content)
        return payload


def manage_retrieved_knowledge(
    *,
    client: "LLMClient",
    state: "NavClawAgentState",
    workspace: "RetrievalWorkspace",
    current_node_id: str,
    progress_updates: list[dict[str, object]],
) -> dict[str, object]:
    refs = _retrieved_refs(workspace)
    if refs == []:
        raise ValueError("knowledge management requires retrieved entities")
    existing = {
        ref: [item.to_dict() for item in _entity_knowledge(state, ref)]
        for ref in refs
    }
    retrieval_rounds = [
        {
            "round": int(round_record.round_index) + 1,
            "query": str(round_record.request.query),
            "request": [item.to_dict() for item in round_record.request.items],
            "conclusion": str(round_record.conclusion),
        }
        for round_record in workspace.rounds
    ]
    system_prompt = """
You are the Entity Knowledge Manager (EKM) in NavClaw.
Convert evidence-grounded retrieval conclusions into compact reusable knowledge for the entities retrieved in the current decision step.
Do not change task progress, reinterpret raw visual evidence, or alter the selected high-level action.
Return only valid JSON matching the provided output contract.
""".strip()
    user_prompt = f"""
Retrieved entity refs:
{json.dumps(refs, ensure_ascii=False)}

Retrieval trace:
{json.dumps(retrieval_rounds, ensure_ascii=False, indent=2)}

Committed task-progress updates:
{json.dumps(progress_updates, ensure_ascii=False, indent=2)}

Existing entity knowledge:
{json.dumps(existing, ensure_ascii=False, indent=2)}

Decision objective:
Update only the retrieved entities with compact facts that remain useful beyond the current decision step.

Retrieval-trace semantics:
- Each round contains its verification query, the entity-field request that was materialized, and TPU's conclusion from that evidence.
- A fact written to an entity must be supported by a conclusion derived from evidence requested from that entity.
- Use only details stated in the supplied conclusions; raw retrieved images are not provided to this module.

Entity-knowledge semantics:
- Entity knowledge is a concise retrieval-derived fact associated with one node, edge, or landmark.
- It is not raw observation data, a task-progress condition, a temporary query, or a navigation command.

Update rules:
- Use only refs listed in `Retrieved entity refs`.
- Add a fact when it is distinct, supported, entity-local, and applicable beyond the current decision step.
- Update an existing fact when the retrieval trace refines or corrects the same claim.
- Remove an existing fact only when the retrieval trace establishes that it is invalid; absence of new support is insufficient.
- Do not create duplicates or paraphrased copies of existing knowledge.
- Do not store the current action choice, waypoint candidate, temporary uncertainty, retrieval query wording, or policy advice.
- Task-progress updates may establish why a fact matters but are not independent evidence about an entity.
- Return an empty list when no reusable entity knowledge changes.

Output contract:
{{
  "entity_knowledge_updates": [
    {{"op":"add","ref":"<retrieved ref>","content":"<compact entity-local fact>"}},
    {{"op":"update","ref":"<retrieved ref>","knowledge_id":"<existing id>","content":"<corrected or refined fact>"}},
    {{"op":"remove","ref":"<retrieved ref>","knowledge_id":"<existing id>"}}
  ]
}}
""".strip()
    parsed = client.manage_knowledge(system_prompt, user_prompt)
    updates = normalize_entity_knowledge_updates(parsed, allowed_refs=set(refs))
    result = apply_entity_knowledge_updates(
        state=state,
        updates=updates,
        current_node_id=current_node_id,
    )
    return {
        "retrieved_refs": refs,
        "proposed_updates": [update.to_dict() for update in updates],
        **result,
        "knowledge_after": {
            ref: [item.to_dict() for item in _entity_knowledge(state, ref)]
            for ref in refs
        },
    }


def normalize_entity_knowledge_updates(
    payload: object,
    *,
    allowed_refs: set[str],
) -> list[EntityKnowledgeUpdate]:
    if not isinstance(payload, dict) or set(payload) != {"entity_knowledge_updates"}:
        raise ValueError(f"invalid entity knowledge output: {payload!r}")
    raw_updates = payload.get("entity_knowledge_updates")
    if not isinstance(raw_updates, list):
        raise ValueError("entity_knowledge_updates must be a list")
    updates: list[EntityKnowledgeUpdate] = []
    for raw_update in raw_updates:
        if not isinstance(raw_update, dict):
            raise ValueError(f"entity knowledge update must be an object: {raw_update!r}")
        op = str(raw_update.get("op", "")).strip()
        ref = str(raw_update.get("ref", "")).strip()
        if ref not in allowed_refs:
            raise ValueError(f"entity knowledge update uses an unretrieved ref: {ref!r}")
        expected_fields = {
            "add": {"op", "ref", "content"},
            "update": {"op", "ref", "knowledge_id", "content"},
            "remove": {"op", "ref", "knowledge_id"},
        }.get(op)
        if expected_fields is None or set(raw_update) != expected_fields:
            raise ValueError(f"invalid entity knowledge update fields: {raw_update!r}")
        knowledge_id = str(raw_update.get("knowledge_id", "")).strip()
        content = str(raw_update.get("content", "")).strip()
        if op in {"update", "remove"} and knowledge_id == "":
            raise ValueError(f"{op} requires a knowledge_id")
        if op in {"add", "update"} and content == "":
            raise ValueError(f"{op} requires non-empty content")
        updates.append(
            EntityKnowledgeUpdate(
                op=op,
                ref=ref,
                knowledge_id=knowledge_id,
                content=content,
            )
        )
    return updates


def apply_entity_knowledge_updates(
    *,
    state: "NavClawAgentState",
    updates: list[EntityKnowledgeUpdate],
    current_node_id: str,
) -> dict[str, object]:
    node_id = str(current_node_id).strip()
    if node_id == "" or not state.graph.has_node(node_id):
        raise ValueError(f"knowledge update node must exist in graph: {node_id!r}")
    applied: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    for update in updates:
        knowledge = _entity_knowledge(state, update.ref)
        if update.op == "add":
            if any(item.content.casefold() == update.content.casefold() for item in knowledge):
                skipped.append({**update.to_dict(), "reason": "duplicate_content"})
                continue
            item = EntityKnowledge(
                knowledge_id=next_entity_knowledge_id(knowledge),
                update_node_id=node_id,
                content=update.content,
            )
            knowledge.append(item)
            applied.append({"op": "add", "ref": update.ref, **item.to_dict()})
            continue
        item_index = next(
            (
                index
                for index, item in enumerate(knowledge)
                if str(item.knowledge_id) == str(update.knowledge_id)
            ),
            None,
        )
        if item_index is None:
            raise ValueError(
                f"unknown knowledge_id {update.knowledge_id!r} for {update.ref!r}"
            )
        if update.op == "update":
            knowledge[item_index].content = str(update.content)
            knowledge[item_index].update_node_id = node_id
            applied.append({"op": "update", "ref": update.ref, **knowledge[item_index].to_dict()})
        else:
            removed = knowledge.pop(item_index)
            applied.append({"op": "remove", "ref": update.ref, **removed.to_dict()})
    return {"applied_updates": applied, "skipped_updates": skipped}


def _retrieved_refs(workspace: "RetrievalWorkspace") -> list[str]:
    refs: list[str] = []
    for round_record in workspace.rounds:
        for item in round_record.request.items:
            ref = str(item.ref)
            if ref not in refs:
                refs.append(ref)
    return refs


def _entity_knowledge(state: "NavClawAgentState", ref: str) -> list[EntityKnowledge]:
    if state.graph.has_node(str(ref)):
        return state.graph.get_node(str(ref)).knowledge
    for edge in state.graph.iter_edges():
        if str(edge.id) == str(ref):
            return edge.knowledge
    candidate = state.landmark_controller.buffer.find_candidate(str(ref))
    if candidate is not None:
        return candidate.knowledge
    raise ValueError(f"unknown knowledge entity ref: {ref!r}")
