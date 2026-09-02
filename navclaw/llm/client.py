from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass

from openai import OpenAI

from navclaw.llm.request_config import _model_uses_temperature_parameter


def _normalize_prompt_text(text: str) -> str:
    return re.sub(r"\n[ \t]*\n(?:[ \t]*\n)+", "\n\n", str(text).strip())


def _normalize_prompt_content(
    content: str | list[dict[str, object]],
) -> str | list[dict[str, object]]:
    if isinstance(content, str):
        return _normalize_prompt_text(content)
    normalized: list[dict[str, object]] = []
    for part in content:
        normalized_part = dict(part)
        if str(normalized_part.get("type", "")) == "text":
            normalized_part["text"] = _normalize_prompt_text(
                str(normalized_part.get("text", ""))
            )
        normalized.append(normalized_part)
    return normalized


@dataclass(frozen=True)
class LLMUsageEntry:
    request_index: int
    call_name: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    def to_dict(self) -> dict[str, object]:
        return {
            "request_index": int(self.request_index),
            "call_name": str(self.call_name),
            "model": str(self.model),
            "prompt_tokens": int(self.prompt_tokens),
            "completion_tokens": int(self.completion_tokens),
            "total_tokens": int(self.total_tokens),
        }


class LLMClient:
    """One OpenAI-compatible multimodal client shared by every VLN module."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.model = str(model).strip()
        if not self.model:
            raise ValueError("model must be non-empty")
        self.base_url = None if base_url is None else str(base_url)
        self.client = OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            timeout=2000,
        )
        self._usage_entries: list[LLMUsageEntry] = []
        self._call_lock = threading.Lock()

    @staticmethod
    def _base_url_is_openai_official(base_url: str | None) -> bool:
        normalized = str(base_url or "").strip().lower().rstrip("/")
        return normalized in {
            "",
            "https://api.openai.com",
            "https://api.openai.com/v1",
        }

    def reset_usage(self) -> None:
        self._usage_entries = []

    def get_usage_summary(self) -> dict[str, object]:
        return {
            "model": self.model,
            "request_count": len(self._usage_entries),
            "prompt_tokens": sum(item.prompt_tokens for item in self._usage_entries),
            "completion_tokens": sum(
                item.completion_tokens for item in self._usage_entries
            ),
            "total_tokens": sum(item.total_tokens for item in self._usage_entries),
            "requests": [item.to_dict() for item in self._usage_entries],
        }

    def _record_usage(self, call_name: str, completion: object) -> None:
        usage = getattr(completion, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(
            getattr(usage, "total_tokens", prompt_tokens + completion_tokens)
            or prompt_tokens + completion_tokens
        )
        self._usage_entries.append(
            LLMUsageEntry(
                request_index=len(self._usage_entries) + 1,
                call_name=str(call_name),
                model=self.model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )
        )

    def _create_visual_json_completion(
        self,
        *,
        call_name: str,
        system_prompt: str,
        user_prompt: str | list[dict[str, object]],
        max_new_tokens: int,
        token_field: str,
        retry_count: int,
        client_kind: str = "la",
        response_schema: dict[str, object] | None = None,
        temperature: float | None = None,
    ) -> dict[str, object]:
        del client_kind
        if token_field not in {"max_tokens", "max_completion_tokens"}:
            raise ValueError(f"unsupported token field: {token_field!r}")
        official_openai = self._base_url_is_openai_official(self.base_url)
        effective_token_field = (
            "max_completion_tokens"
            if official_openai and token_field == "max_tokens"
            else token_field
        )
        messages = [
            {"role": "system", "content": _normalize_prompt_text(system_prompt)},
            {"role": "user", "content": _normalize_prompt_content(user_prompt)},
        ]
        last_error: Exception | None = None
        for attempt_index in range(max(1, int(retry_count))):
            try:
                request: dict[str, object] = {
                    "model": self.model,
                    "messages": messages,
                    effective_token_field: int(max_new_tokens),
                    "timeout": 120,
                }
                if response_schema is not None and official_openai:
                    request["response_format"] = {
                        "type": "json_schema",
                        "json_schema": response_schema,
                    }
                if temperature is not None and _model_uses_temperature_parameter(
                    self.model
                ):
                    request["temperature"] = float(temperature)
                if not official_openai and not self.model.lower().startswith(
                    ("gpt-", "chatgpt-")
                ):
                    request["extra_body"] = {"enable_thinking": False}
                with self._call_lock:
                    completion = self.client.chat.completions.create(**request)
                    self._record_usage(call_name, completion)
                if not completion.choices:
                    raise ValueError(f"{call_name} returned no choices")
                return self._parse_json(
                    call_name,
                    completion.choices[0].message.content,
                )
            except Exception as exc:
                last_error = exc
                if attempt_index + 1 < max(1, int(retry_count)):
                    time.sleep(min(2.0, 0.5 * float(attempt_index + 1)))
        if last_error is not None:
            raise last_error
        raise RuntimeError(f"{call_name} failed without a captured error")

    @staticmethod
    def _parse_json(call_name: str, content: object) -> dict[str, object]:
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"{call_name} returned empty non-text content")
        stripped = content.strip()
        if stripped.startswith("```"):
            stripped = stripped.strip("`").strip()
            if stripped.lower().startswith("json"):
                stripped = stripped[4:].strip()
        try:
            parsed, _ = json.JSONDecoder().raw_decode(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{call_name} returned invalid JSON") from exc
        if not isinstance(parsed, dict):
            raise ValueError(f"{call_name} must return a JSON object")
        return parsed

    def _call(
        self,
        call_name: str,
        system_prompt: str,
        user_prompt: str | list[dict[str, object]],
        *,
        max_new_tokens: int,
        retry_count: int,
    ) -> dict[str, object]:
        return self._create_visual_json_completion(
            call_name=call_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_new_tokens=max_new_tokens,
            token_field="max_completion_tokens",
            retry_count=retry_count,
        )

    def generate_task_progress_memory(self, system_prompt, user_prompt):
        return self._call(
            "visual_task_progress_generator",
            system_prompt,
            user_prompt,
            max_new_tokens=4096,
            retry_count=2,
        )

    def update_task_progress_memory(
        self,
        system_prompt,
        user_prompt,
        *,
        retrieve_fields_by_ref=None,
    ):
        suffix = "_with_retrieval" if retrieve_fields_by_ref is not None else ""
        return self._call(
            f"visual_task_progress_updater{suffix}",
            system_prompt,
            user_prompt,
            max_new_tokens=4096,
            retry_count=5,
        )

    def decide_vln_task_progress_step(self, system_prompt, user_prompt):
        return self._call(
            "vln_task_progress_updater",
            system_prompt,
            user_prompt,
            max_new_tokens=8192,
            retry_count=5,
        )

    def decide_vln_navigation_step(self, system_prompt, user_prompt):
        return self._call(
            "vln_progress_conditioned_navigation_planner",
            system_prompt,
            user_prompt,
            max_new_tokens=8192,
            retry_count=5,
        )

    def manage_knowledge(self, system_prompt, user_prompt):
        return self._call(
            "knowledge_manager",
            system_prompt,
            user_prompt,
            max_new_tokens=2048,
            retry_count=3,
        )

    def summarize_node(self, system_prompt, user_prompt):
        return self._call(
            "visual_node_summary",
            system_prompt,
            user_prompt,
            max_new_tokens=2048,
            retry_count=2,
        )

    def decide_visual_action(self, system_prompt, user_prompt):
        return self._create_visual_json_completion(
            call_name="visual_action_point",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_new_tokens=4096,
            token_field="max_tokens",
            retry_count=3,
            temperature=0.0,
        )

    def verify_vertical_transition_waypoint(self, system_prompt, user_prompt):
        return self._call(
            "vertical_transition_waypoint_verifier",
            system_prompt,
            user_prompt,
            max_new_tokens=2048,
            retry_count=2,
        )

    def decide_stop_confirmation(self, system_prompt, user_prompt):
        return self._call(
            "visual_stop_confirmation",
            system_prompt,
            user_prompt,
            max_new_tokens=2048,
            retry_count=2,
        )

    def decide_vertical_transition_step(self, system_prompt, user_prompt):
        return self._call(
            "vertical_transition_step_planner",
            system_prompt,
            user_prompt,
            max_new_tokens=4096,
            retry_count=2,
        )
