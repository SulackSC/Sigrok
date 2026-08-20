from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from discord.ext import commands
from loguru import logger

from sigrok import genai
from sigrok.config import settings
from sigrok.streaming.buffer import RecentMessageBuffer
from sigrok.streaming.mentions import message_mentions_bot, strip_bot_mention
from sigrok.streaming.messages import StreamingChatMessage
from sigrok.streaming.ratelimit import StreamingRateLimiter
from sigrok.streaming.response import (
    apply_nsfw_filter,
    normalize_bot_response,
    should_skip_response,
    truncate_for_platform,
)
from sigrok.streaming.kick_client import KickChatAdapter
from sigrok.streaming.twitch_client import TwitchChatAdapter
from sigrok.streaming.youtube_client import YouTubeChatAdapter

_PLATFORM_HANDLES = {
    "twitch": lambda: settings.streaming.twitch.bot_username,
    "youtube": lambda: settings.streaming.youtube.bot_display_name,
    "kick": lambda: settings.streaming.kick.bot_username,
}

_PLATFORM_MAX_CHARS = {
    "twitch": lambda: settings.streaming.twitch.max_chars,
    "youtube": lambda: settings.streaming.youtube.max_chars,
    "kick": lambda: settings.streaming.kick.max_chars,
}


class StreamingChatCog(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        self._state_path = Path(settings.streaming.state_file)
        self._buffer = RecentMessageBuffer(settings.streaming.recent_messages_buffer)
        self._rate_limiter = StreamingRateLimiter(
            global_cooldown_seconds=settings.streaming.global_reply_cooldown_seconds,
            per_user_cooldown_seconds=settings.streaming.per_user_cooldown_seconds,
        )
        self._twitch: Optional[TwitchChatAdapter] = None
        self._youtube: Optional[YouTubeChatAdapter] = None
        self._kick: Optional[KickChatAdapter] = None
        self._twitch_sources: dict[str, Any] = {}
        self._youtube_sources: dict[str, dict[str, Any]] = {}
        self._kick_sources: dict[str, dict[str, Any]] = {}
        self._load_state()

    def cog_unload(self) -> None:
        try:
            loop = self.bot.loop
            if loop.is_running():
                loop.create_task(self._stop_adapters())
        except Exception as exc:
            logger.warning(f"Streaming chat cog unload cleanup failed: {exc}")

    def _load_state(self) -> None:
        if not self._state_path.exists():
            return
        try:
            json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(f"Failed to read streaming state file {self._state_path}: {exc}")

    def _save_state(self) -> None:
        payload = {"platforms": ["twitch", "youtube", "kick"]}
        self._state_path.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )

    def _bot_handle(self, platform: str) -> str:
        getter = _PLATFORM_HANDLES.get(platform)
        return getter() if getter else "sigrok"

    def _max_chars(self, platform: str) -> int:
        getter = _PLATFORM_MAX_CHARS.get(platform)
        return int(getter()) if getter else 500

    async def _handle_incoming(
        self,
        message: StreamingChatMessage,
        *,
        platform: str,
        source: Any,
    ) -> None:
        self._buffer.append(message)
        if message.author_is_bot:
            return

        bot_handle = self._bot_handle(platform)
        if not message_mentions_bot(message.content, bot_handle):
            return

        if not self._rate_limiter.allows_reply(
            platform, message.channel_key, message.author_id
        ):
            return

        guard = await self._rate_limiter.reply_guard()
        if guard is None:
            return

        async with guard:
            question = strip_bot_mention(message.content, bot_handle)
            max_chars = self._max_chars(platform)
            if not question:
                response = "say something after the @"
            else:
                history = self._buffer.history(platform, message.channel_key)
                prior = [m.to_genai_message() for m in history[:-1]]
                response = await genai.client.answer_social_question(
                    platform=platform,
                    account_handle=bot_handle,
                    question=question,
                    messages=prior,
                    current_message=message.to_genai_message(),
                    max_chars=max_chars,
                )
                response = apply_nsfw_filter(
                    normalize_bot_response(response, bot_handle=bot_handle)
                )
                if should_skip_response(question, response):
                    logger.info(
                        "Skipping {} reply for {} due to empty/invalid model output.",
                        platform,
                        message.message_id,
                    )
                    return

            response = truncate_for_platform(response, max_chars)
            try:
                await self._send_reply(platform, message, source, response)
            except Exception as exc:
                logger.exception(f"Failed to send {platform} reply: {exc}")
                return

            self._rate_limiter.record_reply(
                platform, message.channel_key, message.author_id
            )
            bot_message = StreamingChatMessage(
                platform=platform,
                channel_key=message.channel_key,
                message_id=f"bot:{message.message_id}",
                author_id="sigrok",
                author_name=bot_handle,
                author_display_name=bot_handle,
                content=response,
                created_at=StreamingChatMessage.now_iso(),
                author_is_bot=True,
            )
            self._buffer.append(bot_message)
            self._save_state()

    async def _send_reply(
        self,
        platform: str,
        message: StreamingChatMessage,
        source: Any,
        text: str,
    ) -> None:
        if platform == "twitch" and self._twitch is not None:
            await self._twitch.send_reply(
                source,
                text,
                reply_to_message_id=message.message_id,
            )
            return
        if platform == "youtube" and self._youtube is not None:
            await self._youtube.send_message(message.channel_key, text)
            return
        if platform == "kick" and self._kick is not None:
            await self._kick.send_message(message.channel_key, text)
            return
        raise RuntimeError(f"No adapter available for platform {platform}")

    async def _on_twitch_message(self, message: StreamingChatMessage, source: Any) -> None:
        self._twitch_sources[message.message_id] = source
        await self._handle_incoming(message, platform="twitch", source=source)

    async def _on_youtube_message(
        self, message: StreamingChatMessage, source: dict[str, Any]
    ) -> None:
        self._youtube_sources[message.message_id] = source
        await self._handle_incoming(message, platform="youtube", source=source)

    async def _on_kick_message(
        self, message: StreamingChatMessage, source: dict[str, Any]
    ) -> None:
        self._kick_sources[message.message_id] = source
        await self._handle_incoming(message, platform="kick", source=source)

    async def _start_adapters(self) -> None:
        if not settings.streaming.enabled:
            logger.info("Streaming chat integration disabled.")
            return

        if settings.streaming.twitch.enabled:
            self._twitch = TwitchChatAdapter(
                streaming=settings.streaming,
                tokens=settings.tokens,
                on_message=self._on_twitch_message,
            )
            try:
                await self._twitch.start()
            except Exception as exc:
                logger.exception(f"Failed to start Twitch adapter: {exc}")
                self._twitch = None

        if settings.streaming.youtube.enabled:
            self._youtube = YouTubeChatAdapter(
                streaming=settings.streaming,
                tokens=settings.tokens,
                on_message=self._on_youtube_message,
            )
            try:
                await self._youtube.start()
            except Exception as exc:
                logger.exception(f"Failed to start YouTube adapter: {exc}")
                self._youtube = None

        if settings.streaming.kick.enabled:
            self._kick = KickChatAdapter(
                streaming=settings.streaming,
                on_message=self._on_kick_message,
            )
            try:
                await self._kick.start(tokens=settings.tokens.kick)
            except Exception as exc:
                logger.exception(f"Failed to start Kick adapter: {exc}")
                self._kick = None

    async def _stop_adapters(self) -> None:
        if self._twitch is not None:
            await self._twitch.stop()
            self._twitch = None
        if self._youtube is not None:
            await self._youtube.stop()
            self._youtube = None
        if self._kick is not None:
            await self._kick.stop()
            self._kick = None

    @commands.Cog.listener()
    async def on_ready(self) -> None:
        if not settings.streaming.enabled:
            return
        await self._start_adapters()


def setup(bot: commands.Bot) -> None:
    bot.add_cog(StreamingChatCog(bot))
