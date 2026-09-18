"""Render matched robot prompt fixtures before/after synchronization, without a model.

Run in the robot Python environment with optional tiktoken available. The before
revision is required so future runs do not silently use a different baseline.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import tiktoken
from navprobe.agent import visual_policy
from navprobe.agent.visual_action_context import VisualActionContext, VisualViewContext
from navprobe.agent.visual_policy_decisions import TaskProgressDecision
from navprobe.memory.task_progress import TaskProgressMemory


class Captured(BaseException):
    pass


def render(module, name, *, navigation=False, retrieved=False, anchor=False,
           terminal=False, eligible=True, completed=0, anchors=True):
    result = {}

    def capture(system, user):
        result["system"] = system
        result["user_blocks"] = [user] if isinstance(user, str) else [
            block["text"] for block in user if block.get("type") == "text"
        ]
        raise Captured()

    client = SimpleNamespace(generate_task_progress_memory=capture,
                             decide_vln_task_progress_step=capture,
                             decide_vln_navigation_step=capture)
    if name == "Initializer":
        try:
            module.ensure_task_progress_memory(
                client=client, task_progress=TaskProgressMemory(),
                goal_text="{original_instruction}", task_type="vln_instruction",
            )
        except Captured:
            return result
    views = [VisualViewContext(angle_deg=angle, obs_id=f"obs_{angle}",
                              rgb_id=f"rgb_{angle}", depth_id=f"depth_{angle}",
                              visible_visited_nodes=["n1"])
             for angle in (0, 180, 90, 270)]
    memory = SimpleNamespace(agenda_initialized=True, items=[],
                             format_for_prompt=lambda **_: "{original_goal_and_task_state}")

    def images(**kwargs):
        prefix = "Planning reference" if kwargs.get("planning_reference") else "Current"
        return [{"type": "text", "text": f"{prefix} {direction} view:"}
                for direction in ("front", "back", "left", "right")]

    with ExitStack() as stack:
        stack.enter_context(patch.object(module, "image_content_for_current_panorama_views", side_effect=images))
        stack.enter_context(patch.object(module, "image_content_for_movement_history_sheet", return_value=None))
        try:
            module._run_vln_task_module_prompt(
                client=client, cache=SimpleNamespace(), goal_kind="vln_instruction",
                visual_context=VisualActionContext(current_node_id="n0" if anchor else "n2", views=views),
                task_progress=memory, memory_index_text="{entity_memory_index}",
                retrieval_workspace_content=([{"type": "text", "text": "{retrieval_evidence_and_conclusions}"}] if retrieved or anchor else []),
                retrieve_max_rounds=0 if navigation else 6,
                retrieve_completed_rounds=0 if navigation else completed,
                retrieve_fields_by_ref={"n1": ["rgb"]},
                allow_retrieve=eligible and not navigation, allow_update_progress=not navigation,
                allow_navigation_actions=navigation, require_retrieval_conclusion=retrieved,
                detected_landmarks_text="{detected_landmarks_by_view}",
                latest_task_progress=TaskProgressDecision(progress_analysis="{executive_assessment}", progress_reasoning="") if navigation else None,
                allowed_backtrack_node_ids=({"n1", "n2"} if anchor else {"n0", "n1"}) if anchors else set(),
                system_owned_waypoint_objective=True, planning_reference_panorama=anchor,
                backtrack_context_text="Backtrack context:\n- Planning reference node: n0\n- Physical robot node: n2\n- Objective: {backtrack_objective}\n- Reason: {backtrack_reason}" if anchor else "",
                terminal_check_context={"movement": "stay", "stop_objective": "{stop_objective}"} if terminal else None,
            )
        except Captured:
            return result
    raise RuntimeError(f"No prompt captured for {name}")



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-ref", required=True)
    args = parser.parse_args()
    source = "navprobe/agent/visual_policy.py"
    assert Path(visual_policy.__file__).resolve() == ROOT / source
    before_source = subprocess.check_output(["git", "show", f"{args.before_ref}:{source}"], cwd=ROOT)
    after_source = (ROOT / source).read_bytes()
    before = ModuleType("navprobe.agent._robot_prompt_baseline")
    exec(compile(before_source, f"{args.before_ref}:{source}", "exec"), before.__dict__)
    encoding = tiktoken.get_encoding("o200k_base")

    def measure(prompt):
        system, blocks = prompt["system"], prompt["user_blocks"]
        result = dict(system_chars=len(system), user_chars=sum(map(len, blocks)),
                      system_tokens=len(encoding.encode(system)),
                      user_tokens=sum(len(encoding.encode(block)) for block in blocks))
        result.update(chars=result["system_chars"] + result["user_chars"],
                      tokens=result["system_tokens"] + result["user_tokens"])
        return result

    variants = [
        ("Initializer", {}),
        ("Executive: regular", {}),
        ("Executive: first step, budget 6 but retrieve disabled", {"eligible": False}),
        ("Executive: after retrieval, 1 of 6 rounds", {"retrieved": True, "completed": 1}),
        ("Executive: anchor, budget exhausted", {"anchor": True, "eligible": False, "completed": 6}),
        ("Executive: terminal, retrieve available", {"terminal": True}),
        ("Executive: terminal after final retrieval", {"terminal": True, "retrieved": True, "eligible": False, "completed": 6}),
        ("Skill: physical reference", {"navigation": True}),
        ("Skill: first step, no anchors", {"navigation": True, "anchors": False}),
        ("Skill: historical reference", {"navigation": True, "anchor": True}),
    ]
    rows = []
    for name, flags in variants:
        old, new = render(before, name, **flags), render(visual_policy, name, **flags)
        a, b = measure(old), measure(new)
        assert old["system"] == new["system"], name
        rows.append(dict(variant=name, flags=flags, before=a, after=b,
                         delta_chars=b["chars"] - a["chars"],
                         delta_tokens=b["tokens"] - a["tokens"],
                         before_text=old, after_text=new))
    data = dict(before_ref=args.before_ref, source=source,
                upstream_ref="bfa2805b219bfe7e0e0f721ed9bb0a47f9049f61",
                before_sha256=hashlib.sha256(before_source).hexdigest(),
                after_sha256=hashlib.sha256(after_source).hexdigest(),
                tokenizer=f"tiktoken {tiktoken.__version__}", encoding=encoding.name,
                rows=rows)
    output = Path(__file__).resolve().parent
    (output / "prompt_usage.json").write_text(json.dumps(data, indent=2) + "\n")
    lines = [
        "# Robot prompt text usage after NavProbe synchronization", "",
        f"Robot baseline: `{args.before_ref}`. Upstream: `{data['upstream_ref']}`.", "",
        "## Method", "",
        "Both versions are rendered with their actual prompt constructors and identical placeholder inputs: default VLN, dynamic agenda, FSS, four view captions, and the conditional modes below. The script asserts that the imported runtime belongs to this repository. Full text fixtures, flags, source hashes, and component counts are in [prompt_usage.json](prompt_usage.json).",
        "",
        f"Tokens use **{data['tokenizer']}, `{encoding.name}`**. Characters are Unicode code points, including whitespace. System and individual user text blocks are counted separately and summed; all measured system strings are unchanged.",
        "",
        "Counts include the fixed placeholder task/memory payloads. Excluded: image tokens, API envelopes, generated/reasoning tokens, retries, and variable production history. These are local text counts, not API billing or measured episode usage. The variants are alternatives and must not be summed as an episode total. No robot or live model is called.",
        "", "## Results", "",
        "| Variant | Characters before | After | Delta | Tokens before | After | Delta | Token change |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        a, b = row["before"], row["after"]
        lines.append(f"| {row['variant']} | {a['chars']:,} | {b['chars']:,} | {row['delta_chars']:+,} | {a['tokens']:,} | {b['tokens']:,} | {row['delta_tokens']:+,} | {100 * row['delta_tokens'] / a['tokens']:+.1f}% |")
    lines += [
        "", "## Interpretation", "",
        "This comparison includes P0 contract/semantic repairs and P1.1/P1.2 retrieval guidance/compression together. Most variants shrink; initialization and historical-reference skill prompts grow because they now preserve instruction semantics and explain physical execution and historical `stay` accurately. Compression does not imply improved navigation or stronger-model compatibility without new rollouts.", "",
        "Other runtime modules are unchanged, so their fixed-text deltas are zero under matched inputs. The P1.3 handoff audit introduced no additional context packet. Actual dynamic inputs and episode token totals may change as decisions change.", "",
        "## Reproduction", "",
        "Use the matching runtime source identified by `after_sha256`, the declared robot dependencies, and optional `tiktoken==0.14.0` with the `o200k_base` vocabulary available. No tokenizer dependency was added to the robot runtime.", "",
        "```bash",
        f"python docs/prompt-sync-20260918/measure_prompt_usage.py --before-ref {args.before_ref}",
        "```", "",
    ]
    (output / "PROMPT_USAGE_REPORT.md").write_text("\n".join(lines))
    for row in rows:
        print(f"{row['variant']}: {row['delta_chars']:+} characters, {row['delta_tokens']:+} tokens")


if __name__ == "__main__":
    main()
