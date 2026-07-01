from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from loguru import logger

from sigrok.config import StreamingSettings, Tokens, YouTubeStreamingSettings
from sigrok.streaming.messages import StreamingChatMessage

OnYouTubeMessage = Callable[[StreamingChatMessage, dict[str, Any]], Awaitable[None]]

_YOUTUBE_SCOPES = ["https://www.googleapis.com/auth/youtube.force-ssl"]


class YouTubeChatAdapter:
    def __init__(
        self,
        *,
        streaming: StreamingSettings,
        tokens: Tokens,
        on_message: OnYouTubeMessage,
    ) -> None:
        self._settings = streaming.youtube
        self._tokens = tokens.youtube
        self._on_message = on_message
        self._tasks: list[asyncio.Task[None]] = []
        self._stop = asyncio.Event()
        self._channel_live_chat: dict[str, str] = {}
        self._seen_message_ids: dict[str, set[str]] = {}
        self._credentials: Optional[Credentials] = None
        self._youtube_service: Any = None

    @property
    def bot_display_name(self) -> str:
        return self._settings.bot_display_name

    @property
    def max_chars(self) -> int:
        return self._settings.max_chars

    def _build_credentials(self) -> Credentials:
        if not self._tokens.client_id or not self._tokens.client_secret:
            raise RuntimeError("YouTube client_id/client_secret missing in tokens.youtube config.")
        if not self._tokens.refresh_token:
            raise RuntimeError("YouTube refresh_token missing in tokens.youtube config.")
        return Credentials(
            token=None,
            refresh_token=self._tokens.refresh_token,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=self._tokens.client_id,
            client_secret=self._tokens.client_secret,
            scopes=_YOUTUBE_SCOPES,
        )

    def _ensure_youtube_service(self) -> Any:
        if self._credentials is None:
            self._credentials = self._build_credentials()
        if self._credentials.expired and self._credentials.refresh_token:
            self._credentials.refresh(Request())
        if self._youtube_service is None:
            self._youtube_service = build(
                "youtube",
                "v3",
                credentials=self._credentials,
                cache_discovery=False,
            )
        return self._youtube_service

    async def start(self) -> None:
        if not self._settings.enabled:
            return
        for channel_id in self._settings.channels:
            channel_id = channel_id.strip()
            if not channel_id:
                continue
            self._seen_message_ids.setdefault(channel_id, set())
            task = asyncio.create_task(self._poll_channel(channel_id))
            self._tasks.append(task)
        logger.info("YouTube chat adapter started for {} channel(s).", len(self._tasks))

    async def stop(self) -> None:
        self._stop.set()
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        self._stop.clear()
        self._credentials = None
        self._youtube_service = None

    async def _resolve_live_chat_id(self, channel_id: str) -> Optional[str]:
        def _fetch() -> Optional[str]:
            yt = self._ensure_youtube_service()
            search = (
                yt.search()
                .list(
                    part="id",
                    channelId=channel_id,
                    type="video",
                    eventType="live",
                    maxResults=1,
                )
                .execute()
            )
            items = search.get("items") or []
            if not items:
                return None
            video_id = items[0]["id"]["videoId"]
            video = (
                yt.videos()
                .list(part="liveStreamingDetails", id=video_id)
                .execute()
            )
            video_items = video.get("items") or []
            if not video_items:
                return None
            details = video_items[0].get("liveStreamingDetails") or {}
            return details.get("activeLiveChatId")

        return await asyncio.to_thread(_fetch)

    async def _poll_channel(self, channel_id: str) -> None:
        page_token: Optional[str] = None
        while not self._stop.is_set():
            try:
                live_chat_id = await self._resolve_live_chat_id(channel_id)
                if not live_chat_id:
                    await asyncio.sleep(30)
                    continue
                self._channel_live_chat[channel_id] = live_chat_id

                def _list_messages() -> tuple[list[dict[str, Any]], Optional[str], int]:
                    yt = self._ensure_youtube_service()
                    kwargs: dict[str, Any] = {
                        "liveChatId": live_chat_id,
                        "part": "id,snippet,authorDetails",
                        "maxResults": 200,
                    }
                    if page_token:
                        kwargs["pageToken"] = page_token
                    response = yt.liveChatMessages().list(**kwargs).execute()
                    return (
                        response.get("items") or [],
                        response.get("nextPageToken"),
                        int(response.get("pollingIntervalMillis") or 5000),
                    )

                items, page_token, interval_ms = await asyncio.to_thread(_list_messages)
                seen = self._seen_message_ids[channel_id]
                for item in items:
                    message_id = str(item.get("id") or "")
                    if not message_id or message_id in seen:
                        continue
                    seen.add(message_id)
                    if len(seen) > 2000:
                        seen.clear()
                    snippet = item.get("snippet") or {}
                    author = item.get("authorDetails") or {}
                    if author.get("isChatModerator") and author.get("channelId") == channel_id:
                        pass
                    display = str(snippet.get("displayMessage") or "")
                    author_name = str(author.get("displayName") or "unknown")
                    author_channel_id = str(author.get("channelId") or author_name)
                    message = StreamingChatMessage(
                        platform="youtube",
                        channel_key=channel_id,
                        message_id=message_id,
                        author_id=author_channel_id,
                        author_name=author_name,
                        author_display_name=author_name,
                        content=display,
                        created_at=str(snippet.get("publishedAt") or StreamingChatMessage.now_iso()),
                    )
                    await self._on_message(message, item)

                await asyncio.sleep(max(interval_ms / 1000.0, 2.0))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception(f"YouTube chat poll error for {channel_id}: {exc}")
                await asyncio.sleep(15)

    async def send_message(self, channel_id: str, text: str) -> None:
        live_chat_id = self._channel_live_chat.get(channel_id)
        if not live_chat_id:
            live_chat_id = await self._resolve_live_chat_id(channel_id)
        if not live_chat_id:
            raise RuntimeError(f"No active live chat for YouTube channel {channel_id}")

        def _insert() -> None:
            yt = self._ensure_youtube_service()
            body = {
                "snippet": {
                    "liveChatId": live_chat_id,
                    "type": "textMessageEvent",
                    "textMessageDetails": {"messageText": text},
                }
            }
            yt.liveChatMessages().insert(part="snippet", body=body).execute()

        await asyncio.to_thread(_insert)
