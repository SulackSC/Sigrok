from __future__ import annotations

import re

from sigrok.genai import SIGROK_PERSONALITY_SYSTEM_PROMPT
from sigrok.streaming.mentions import strip_bot_mention


def normalize_bot_response(text: str, *, bot_handle: str = "") -> str:
    normalized = text.strip()
    if (
        len(normalized) >= 2
        and normalized[0] == normalized[-1]
        and normalized[0] in {'"', "'"}
    ):
        normalized = normalized[1:-1].strip()
    normalized = re.sub(r"\s+\n", "\n", normalized)
    normalized = re.sub(r"^sigrok:\s*", "", normalized, flags=re.IGNORECASE).strip()
    if bot_handle:
        normalized = strip_bot_mention(normalized, bot_handle)
    return normalized


def should_skip_response(question: str, response: str) -> bool:
    r = response.strip()
    return (
        not r
        or r.lower() == question.lower().strip()
        or (len(r) < 200 and r in SIGROK_PERSONALITY_SYSTEM_PROMPT)
        or r in {"not worth my time", "I couldn't answer that right now."}
    )


def truncate_for_platform(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3].rstrip(" ,;:\n\t") + "..."
