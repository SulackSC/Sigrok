from __future__ import annotations

import re
from pathlib import Path

from loguru import logger

from sigrok.genai import SIGROK_PERSONALITY_SYSTEM_PROMPT
from sigrok.streaming.mentions import strip_bot_mention

NSFW_FILTER_REPLACEMENT = "🍆"

_RESOURCE_ROOT = Path(__file__).resolve().parents[3] / "resources"
_NSFW_BLOCKLIST_PATH = _RESOURCE_ROOT / "nsfw_blocklist.txt"

_LEET_MAP = str.maketrans(
    {
        "@": "a",
        "0": "o",
        "1": "i",
        "!": "i",
        "3": "e",
        "4": "a",
        "5": "s",
        "$": "s",
        "7": "t",
    }
)


def _load_nsfw_blocklist(path: Path | None = None) -> tuple[str, ...]:
    blocklist_path = path or _NSFW_BLOCKLIST_PATH
    terms: list[str] = []
    try:
        raw = blocklist_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.warning(
            f"NSFW blocklist missing at {blocklist_path}; output filter disabled."
        )
        return tuple()
    except OSError as exc:
        logger.warning(
            f"Failed to read NSFW blocklist {blocklist_path}: {exc}; output filter disabled."
        )
        return tuple()

    for line in raw.splitlines():
        stripped = line.split("#", 1)[0].strip().lower()
        if stripped:
            terms.append(stripped)
    return tuple(terms)


_NSFW_BLOCKLIST = _load_nsfw_blocklist()


def _normalize_for_nsfw_scan(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _normalize_leet_for_nsfw_scan(text: str) -> str:
    normalized = text.lower().translate(_LEET_MAP)
    return re.sub(r"\s+", " ", normalized).strip()


def _term_matches(text: str, term: str) -> bool:
    if " " in term:
        return term in text
    return re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text) is not None


def _text_matches_blocklist(text: str) -> bool:
    variants = {_normalize_for_nsfw_scan(text), _normalize_leet_for_nsfw_scan(text)}
    for normalized in variants:
        for term in _NSFW_BLOCKLIST:
            if _term_matches(normalized, term):
                return True
    return False


def apply_nsfw_filter(text: str) -> str:
    if not _NSFW_BLOCKLIST:
        return text
    if _text_matches_blocklist(text):
        logger.info("NSFW output filter triggered; replacing response.")
        return NSFW_FILTER_REPLACEMENT
    return text


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
