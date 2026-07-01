from __future__ import annotations

import re


def normalize_handle(handle: str) -> str:
    return handle.lstrip("@").strip().lower()


def message_mentions_bot(text: str, bot_handle: str) -> bool:
    handle = normalize_handle(bot_handle)
    if not handle:
        return False
    pattern = rf"(?<!\w)@{re.escape(handle)}\b"
    return bool(re.search(pattern, text, flags=re.IGNORECASE))


def strip_bot_mention(text: str, bot_handle: str) -> str:
    handle = normalize_handle(bot_handle)
    if not handle:
        return " ".join(text.split()).strip()
    pattern = rf"(?<!\w)@{re.escape(handle)}\b"
    stripped = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    return " ".join(stripped.split()).strip(" ,:\n\t")
