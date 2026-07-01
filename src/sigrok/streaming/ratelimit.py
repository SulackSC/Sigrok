from __future__ import annotations

import asyncio
import time
from typing import Optional


class StreamingRateLimiter:
    def __init__(
        self,
        *,
        global_cooldown_seconds: float,
        per_user_cooldown_seconds: float,
    ) -> None:
        self._global_cooldown = max(0.0, global_cooldown_seconds)
        self._per_user_cooldown = max(0.0, per_user_cooldown_seconds)
        self._last_global_reply = 0.0
        self._last_user_reply: dict[str, float] = {}
        self._reply_lock = asyncio.Lock()

    def _user_key(self, platform: str, channel_key: str, author_id: str) -> str:
        return f"{platform}:{channel_key}:{author_id}"

    def allows_reply(self, platform: str, channel_key: str, author_id: str) -> bool:
        now = time.monotonic()
        if self._global_cooldown > 0:
            if now - self._last_global_reply < self._global_cooldown:
                return False
        user_key = self._user_key(platform, channel_key, author_id)
        last_user = self._last_user_reply.get(user_key, 0.0)
        if self._per_user_cooldown > 0 and now - last_user < self._per_user_cooldown:
            return False
        return True

    def record_reply(self, platform: str, channel_key: str, author_id: str) -> None:
        now = time.monotonic()
        self._last_global_reply = now
        self._last_user_reply[self._user_key(platform, channel_key, author_id)] = now

    async def acquire_reply_lock(self) -> bool:
        if self._reply_lock.locked():
            return False
        await self._reply_lock.acquire()
        return True

    def release_reply_lock(self) -> None:
        if self._reply_lock.locked():
            self._reply_lock.release()

    async def reply_guard(self) -> Optional["ReplyGuard"]:
        acquired = await self.acquire_reply_lock()
        if not acquired:
            return None
        return ReplyGuard(self)


class ReplyGuard:
    def __init__(self, limiter: StreamingRateLimiter) -> None:
        self._limiter = limiter

    async def __aenter__(self) -> "ReplyGuard":
        return self

    async def __aexit__(self, *_: object) -> None:
        self._limiter.release_reply_lock()
