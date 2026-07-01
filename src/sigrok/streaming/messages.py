from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional


@dataclass
class StreamingChatMessage:
    platform: str
    channel_key: str
    message_id: str
    author_id: str
    author_name: str
    author_display_name: str
    content: str
    created_at: str
    reply_to_message_id: Optional[str] = None
    author_is_bot: bool = False

    def to_genai_message(self) -> dict[str, Any]:
        return {
            "id": self.message_id,
            "author_id": self.author_id,
            "author_name": self.author_name,
            "author_display_name": self.author_display_name,
            "author_handle": self.author_name,
            "author_is_bot": self.author_is_bot,
            "created_at": self.created_at,
            "reply_to_message_id": self.reply_to_message_id,
            "content": self.content.strip() or "[no text]",
            "attachments": [],
        }

    @staticmethod
    def now_iso() -> str:
        return (
            datetime.now(timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        )
