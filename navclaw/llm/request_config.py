from __future__ import annotations


def _model_uses_temperature_parameter(model: str) -> bool:
    normalized = str(model).strip().lower()
    return not (
        normalized == "4o"
        or normalized.startswith("4o-")
        or normalized.startswith("gpt-")
        or normalized.startswith("chatgpt-")
        or normalized.startswith("o1")
        or normalized.startswith("o3")
        or normalized.startswith("o4")
    )
