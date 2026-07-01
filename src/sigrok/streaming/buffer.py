from __future__ import annotations

from collections import deque

from sigrok.streaming.messages import StreamingChatMessage


class RecentMessageBuffer:
    def __init__(self, max_size: int) -> None:
        self._max_size = max(1, max_size)
        self._buffers: dict[str, deque[StreamingChatMessage]] = {}

    def _key(self, platform: str, channel_key: str) -> str:
        return f"{platform}:{channel_key}"

    def append(self, message: StreamingChatMessage) -> None:
        key = self._key(message.platform, message.channel_key)
        buf = self._buffers.setdefault(key, deque(maxlen=self._max_size))
        buf.append(message)

    def history(self, platform: str, channel_key: str) -> list[StreamingChatMessage]:
        key = self._key(platform, channel_key)
        buf = self._buffers.get(key)
        if not buf:
            return []
        return list(buf)
