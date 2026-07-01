from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Optional

from loguru import logger
from twitchio import eventsub
from twitchio.ext import commands
from twitchio.models.eventsub_ import ChatMessage

from sigrok.config import StreamingSettings, Tokens, TwitchStreamingSettings
from sigrok.streaming.messages import StreamingChatMessage

if TYPE_CHECKING:
    from twitchio.user import PartialUser

OnTwitchMessage = Callable[[StreamingChatMessage, ChatMessage], Awaitable[None]]


class SigrokTwitchBot(commands.Bot):
    def __init__(
        self,
        *,
        settings: TwitchStreamingSettings,
        tokens: Tokens,
        on_message: OnTwitchMessage,
    ) -> None:
        if not tokens.twitch_client_id or not tokens.twitch_client_secret:
            raise RuntimeError("Twitch client_id/client_secret missing in tokens config.")
        if not settings.bot_user_id:
            raise RuntimeError("streaming.twitch.bot_user_id is required.")
        if not tokens.twitch_bot_access_token or not tokens.twitch_bot_refresh_token:
            raise RuntimeError("Twitch bot access/refresh tokens missing in tokens config.")

        self._stream_settings = settings
        self._tokens = tokens
        self._on_message_cb = on_message
        self._broadcaster_logins: dict[str, str] = {}

        super().__init__(
            client_id=tokens.twitch_client_id,
            client_secret=tokens.twitch_client_secret,
            bot_id=settings.bot_user_id,
            owner_id=settings.owner_user_id or None,
            prefix="!",
        )

    async def setup_hook(self) -> None:
        await self.add_token(
            self._tokens.twitch_bot_access_token,
            self._tokens.twitch_bot_refresh_token,
        )
        for login in self._stream_settings.channels:
            normalized = login.strip().lower()
            if not normalized:
                continue
            users = await self.fetch_users(logins=[normalized])
            if not users:
                logger.warning(f"Twitch channel login not found: {normalized}")
                continue
            broadcaster = users[0]
            self._broadcaster_logins[broadcaster.id] = broadcaster.name
            payload = eventsub.ChatMessageSubscription(
                broadcaster_user_id=broadcaster.id,
                user_id=self.bot_id,
            )
            await self.subscribe_websocket(payload, as_bot=True)
            logger.info(
                "Subscribed to Twitch chat for channel {} ({})",
                broadcaster.name,
                broadcaster.id,
            )

    async def event_message(self, payload: ChatMessage) -> None:
        if payload.chatter.id == self.bot_id:
            return
        if payload.source_broadcaster is not None:
            return

        channel_key = payload.broadcaster.name.lower()
        reply_to = payload.reply.parent_message_id if payload.reply else None
        message = StreamingChatMessage(
            platform="twitch",
            channel_key=channel_key,
            message_id=payload.id,
            author_id=payload.chatter.id,
            author_name=payload.chatter.name,
            author_display_name=payload.chatter.display_name or payload.chatter.name,
            content=payload.text,
            created_at=StreamingChatMessage.now_iso(),
            reply_to_message_id=reply_to,
        )
        await self._on_message_cb(message, payload)

    async def send_chat(
        self,
        payload: ChatMessage,
        text: str,
        *,
        reply_to_message_id: Optional[str] = None,
    ) -> None:
        broadcaster: PartialUser = payload.broadcaster
        await broadcaster.send_message(
            text,
            sender=self.bot_id,
            token_for=self.bot_id,
            reply_to_message_id=reply_to_message_id,
        )


class TwitchChatAdapter:
    def __init__(
        self,
        *,
        streaming: StreamingSettings,
        tokens: Tokens,
        on_message: OnTwitchMessage,
    ) -> None:
        self._settings = streaming.twitch
        self._tokens = tokens
        self._on_message = on_message
        self._bot: Optional[SigrokTwitchBot] = None
        self._run_task: Optional[asyncio.Task[None]] = None
        self._started = False

    @property
    def bot_username(self) -> str:
        return self._settings.bot_username

    @property
    def max_chars(self) -> int:
        return self._settings.max_chars

    async def start(self) -> None:
        if self._started or not self._settings.enabled:
            return
        self._bot = SigrokTwitchBot(
            settings=self._settings,
            tokens=self._tokens,
            on_message=self._on_message,
        )
        self._run_task = asyncio.create_task(
            self._bot.start(load_tokens=False, save_tokens=False),
            name="twitch-chat-adapter",
        )
        self._started = True
        logger.info("Twitch chat adapter started.")

    async def stop(self) -> None:
        if self._bot is not None:
            await self._bot.close(save_tokens=False)
            self._bot = None
        if self._run_task is not None:
            self._run_task.cancel()
            try:
                await self._run_task
            except asyncio.CancelledError:
                pass
            self._run_task = None
        self._started = False

    async def send_reply(
        self,
        source: ChatMessage,
        text: str,
        *,
        reply_to_message_id: Optional[str] = None,
    ) -> None:
        if self._bot is None:
            raise RuntimeError("Twitch adapter is not started.")
        await self._bot.send_chat(
            source,
            text,
            reply_to_message_id=reply_to_message_id,
        )
