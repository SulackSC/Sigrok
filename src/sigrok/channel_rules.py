from __future__ import annotations

import re
import time
from typing import Any, Optional

from loguru import logger

_CACHE_TTL_SECONDS = 60.0
# channel_id -> (expires_at, rules_text)
_rules_cache: dict[int, tuple[float, str]] = {}


def _bot_mention_patterns(bot_user: Any) -> list[str]:
    """Return regex patterns that match a leading/trailing @Sigrok mention token."""
    patterns: list[str] = []
    bot_id = getattr(bot_user, "id", None)
    if bot_id is not None:
        patterns.append(rf"<@!?{re.escape(str(bot_id))}>")
    for attr in ("name", "display_name", "global_name"):
        name = getattr(bot_user, attr, None)
        if name:
            patterns.append(rf"@{re.escape(str(name))}")
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for p in patterns:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def _mention_token_regex(bot_user: Any) -> Optional[re.Pattern[str]]:
    patterns = _bot_mention_patterns(bot_user)
    if not patterns:
        return None
    return re.compile("|".join(patterns), re.IGNORECASE)


def is_channel_rule_pin(content: str, bot_user: Any) -> bool:
    """True if content starts and ends with a bot mention and has a non-empty body."""
    return bool(extract_channel_rule_body(content, bot_user))


def extract_channel_rule_body(content: str, bot_user: Any) -> str:
    """
    If content starts and ends with an @Sigrok mention, return the trimmed body between them.
    Otherwise return "".
    """
    text = (content or "").strip()
    if not text or bot_user is None:
        return ""
    token_re = _mention_token_regex(bot_user)
    if token_re is None:
        return ""

    start_match = token_re.match(text)
    if not start_match:
        return ""

    # Find the last mention token that ends at the end of the string
    end_match = None
    for match in token_re.finditer(text):
        if match.end() == len(text):
            end_match = match
    if end_match is None:
        return ""
    # Need two distinct tokens (start and end)
    if end_match.start() <= start_match.start():
        return ""

    body = text[start_match.end() : end_match.start()].strip()
    return body


def clear_channel_rules_cache(channel_id: Optional[int] = None) -> None:
    """Clear cached rules for one channel, or all channels if channel_id is None."""
    if channel_id is None:
        _rules_cache.clear()
    else:
        _rules_cache.pop(channel_id, None)


async def fetch_channel_rules(channel: Any, bot_user: Any) -> str:
    """
    Load pinned messages that start and end with @Sigrok, return concatenated rule bodies.
    Results are cached per channel for ~60s.
    """
    channel_id = getattr(channel, "id", None)
    if channel_id is None or bot_user is None:
        return ""

    now = time.monotonic()
    cached = _rules_cache.get(channel_id)
    if cached is not None:
        expires_at, rules = cached
        if now < expires_at:
            return rules

    pins_fn = getattr(channel, "pins", None)
    if pins_fn is None:
        return ""

    try:
        pins = await pins_fn()
    except Exception as exc:
        logger.warning(f"Failed to fetch pins for channel {channel_id}: {exc}")
        return ""

    # Discord returns pins newest-first; plan wants oldest first
    bodies: list[str] = []
    for pin in reversed(list(pins)):
        body = extract_channel_rule_body(getattr(pin, "content", "") or "", bot_user)
        if body:
            bodies.append(body)

    rules = "\n\n".join(bodies)
    _rules_cache[channel_id] = (now + _CACHE_TTL_SECONDS, rules)
    return rules
