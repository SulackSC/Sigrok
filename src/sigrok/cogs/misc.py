import asyncio
import re
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncIterator, Optional

from discord import Message
from discord.errors import Forbidden, HTTPException
from discord.ext import commands
from loguru import logger

from sigrok import db, genai
from sigrok.channel_rules import fetch_channel_rules
from sigrok.config import settings
from sigrok.genai import GenAILlamaCpp, SIGROK_PERSONALITY_SYSTEM_PROMPT
from sigrok.relationships import validate_reflection_response
from sigrok.streaming.response import apply_nsfw_filter
from sigrok.supersigrok import (
    RATE_LIMIT_FALLBACK_REPLY,
    badge_reply,
    humanize_duration,
    is_supersigrok,
)


async def _try_enter_typing(channel) -> Optional[object]:
    """Start Discord typing; return context manager instance for __aexit__, or None if rate-limited."""
    ctx = channel.typing()
    try:
        await ctx.__aenter__()
        return ctx
    except HTTPException as exc:
        if exc.status == 429:
            logger.warning(
                f"Discord typing rate limited for channel={getattr(channel, 'id', None)}; "
                "continuing without typing"
            )
            return None
        raise


@asynccontextmanager
async def _safe_typing(channel) -> AsyncIterator[None]:
    ctx = await _try_enter_typing(channel)
    try:
        yield
    finally:
        if ctx is not None:
            try:
                await ctx.__aexit__(None, None, None)
            except Exception:
                pass


class Misc(commands.Cog):
    bot: commands.Bot
    # Scheduled @schedule jobs: cap how many human turns feed the SLM (plus bot lines in window).
    _DEFERRED_SCHEDULE_RECENT_HUMAN_TURNS = 5

    def __init__(self, bot):
        self.bot = bot
        self._relationship_tasks: set[asyncio.Task[None]] = set()
        self._relationship_reflection_semaphore = asyncio.Semaphore(1)

    def cog_unload(self) -> None:
        for task in self._relationship_tasks:
            task.cancel()
        self._relationship_tasks.clear()

    def _track_relationship_task(self, task: asyncio.Task[None]) -> None:
        self._relationship_tasks.add(task)
        task.add_done_callback(self._relationship_tasks.discard)

    async def _maybe_schedule_relationship_reflection(self, message: Message) -> None:
        guild = message.guild
        if guild is None:
            return
        channel_id = getattr(message.channel, "id", None)
        if not isinstance(channel_id, int):
            return
        if not isinstance(genai.client, genai.GenAIOpenCodeGo):
            logger.warning(
                "Skipping relationship reflection: the active backend is not "
                "OpenCode Go, so Max Thinking cannot be guaranteed."
            )
            return
        try:
            lease_token = await db.try_claim_relationship_run(guild.id, channel_id)
        except Exception:
            logger.exception(
                "Failed to claim relationship reflection "
                f"guild={guild.id} channel={channel_id}"
            )
            return
        if lease_token is None:
            return
        task = asyncio.create_task(
            self._run_relationship_reflection(message, lease_token),
            name=f"sigrok-relationship-reflection-{channel_id}",
        )
        self._track_relationship_task(task)

    async def _run_relationship_reflection(
        self, message: Message, lease_token: str
    ) -> None:
        guild = message.guild
        channel_id = getattr(message.channel, "id", None)
        if guild is None or not isinstance(channel_id, int):
            return
        guild_id = guild.id
        try:
            async with self._relationship_reflection_semaphore:
                provider = genai.client
                if not isinstance(provider, genai.GenAIOpenCodeGo):
                    raise RuntimeError(
                        "OpenCode Go is required for relationship reflection"
                    )
                response, allowed_user_ids, message_author_by_id = (
                    await provider.reflect_relationships_from_channel(message)
                )
                updates = validate_reflection_response(
                    response,
                    allowed_user_ids=allowed_user_ids,
                    message_author_by_id=message_author_by_id,
                )
                if updates is None:
                    raise ValueError("Relationship reflection returned invalid JSON")
                changed_rows = await db.apply_relationship_updates(
                    guild_id, updates
                )
                completed = await db.complete_relationship_run(
                    channel_id, lease_token
                )
                if not completed:
                    raise RuntimeError(
                        "Relationship reflection lease was no longer owned"
                    )
                logger.info(
                    "relationship_reflection_complete "
                    f"guild={guild_id} channel={channel_id} "
                    f"candidates={len(updates)} changed={len(changed_rows)}"
                )
        except asyncio.CancelledError:
            try:
                await db.release_relationship_run(channel_id, lease_token)
            except Exception:
                logger.exception(
                    "Failed to release cancelled relationship lease "
                    f"channel={channel_id}"
                )
            raise
        except Exception:
            logger.exception(
                "Relationship reflection failed "
                f"guild={guild_id} channel={channel_id}"
            )
            try:
                await db.release_relationship_run(channel_id, lease_token)
            except Exception:
                logger.exception(
                    f"Failed to release relationship lease channel={channel_id}"
                )

    async def _format_reply_chain_snippet(
        self,
        message: Message,
        max_depth: int = 4,
        max_total_chars: int = 500,
    ) -> str:
        """
        Build a short "reply chain" snippet to include in the bot's reply text.
        """
        if not message.reference or message.reference.message_id is None:
            return ""

        lines: list[str] = []
        current: Message = message
        depth = 0

        while (
            current.reference
            and current.reference.message_id is not None
            and depth < max_depth
        ):
            try:
                if (
                    current.reference.resolved
                    and isinstance(current.reference.resolved, Message)
                ):
                    parent = current.reference.resolved
                else:
                    parent = await current.channel.fetch_message(
                        current.reference.message_id
                    )
            except Exception:
                break

            content = (parent.content or "").strip()
            content = content.replace("\n", " ")
            if not content and parent.attachments:
                content = "attachments"

            excerpt = content[:120] + ("..." if len(content) > 120 else "")
            lines.append(
                f"[reply:{parent.author.name} id={parent.id}] {excerpt}".lower()
            )

            current = parent
            depth += 1

        if not lines:
            return ""

        # Show oldest first in the snippet.
        lines.reverse()
        snippet = "\n".join(lines)
        return snippet[:max_total_chars].rstrip()

    def _is_whitelisted_channel(self, message: Message) -> bool:
        if message.guild is None:
            return False
        channel_id = getattr(message.channel, "id", None)
        return any(
            entry.guild == message.guild.id
            and (entry.channel == 0 or entry.channel == channel_id)
            for entry in settings.bot.whitelist
        )

    def _is_whitelisted_guild(self, message: Message) -> bool:
        """True if this server appears in bot whitelist (any entry). Used for @Sigrok pings."""
        if message.guild is None:
            return False
        return any(entry.guild == message.guild.id for entry in settings.bot.whitelist)

    async def _is_authorized_guild(self, message: Message) -> bool:
        """TOML whitelist or active SuperSigrok subscriber guild."""
        if message.guild is None:
            return False
        if self._is_whitelisted_guild(message):
            return True
        from sigrok.supersigrok import guild_is_authorized

        return await guild_is_authorized(message.guild.id)

    def _strip_transcript_format(self, content: str) -> str:
        """Remove leaked internal transcript format like '[ID: 123 | user]: ' or 'ID: 123 | user]: '."""
        return re.sub(
            r"^(?:\[)?ID:\s*\d+\s*\|\s*[^\]:]+(?:\s+replying to \d+)?\]:\s*",
            "",
            content,
            flags=re.IGNORECASE,
        ).strip()

    def _strip_bot_mention(self, content: str) -> str:
        if self.bot.user is None:
            return content.strip()
        mention_patterns = [
            f"<@{self.bot.user.id}>",
            f"<@!{self.bot.user.id}>",
            f"@{self.bot.user.name}",
        ]
        question = content
        for mention in mention_patterns:
            question = question.replace(mention, " ")
        return " ".join(question.split()).strip()

    def _strip_bot_mention_raw(self, content: str) -> str:
        """Strip bot mention tokens from raw message content (preserves @schedule, etc.)."""
        if self.bot.user is None:
            return content.strip()
        mention_patterns = [
            f"<@{self.bot.user.id}>",
            f"<@!{self.bot.user.id}>",
            f"@{self.bot.user.name}",
        ]
        out = content
        for mention in mention_patterns:
            out = out.replace(mention, " ")
        # Preserve newlines — do not " ".join(all.split()) or multi-line @schedule bodies collapse.
        lines = [" ".join(line.split()) for line in out.splitlines()]
        return "\n".join(lines).strip()

    def _message_mentions_self(self, message: Message) -> bool:
        if self.bot.user is None:
            return False

        if self.bot.user in message.mentions:
            return True

        raw_content = message.content or ""
        mention_tokens = (
            f"<@{self.bot.user.id}>",
            f"<@!{self.bot.user.id}>",
        )
        if any(token in raw_content for token in mention_tokens):
            return True

        clean_content = (message.clean_content or "").lower()
        return f"@{self.bot.user.name}".lower() in clean_content

    async def _send_response_to_ping(self, message: Message, text: str) -> None:
        """
        Prefer a real Discord reply (message reference). If Discord returns 403 (usually
        missing **Read Message History**), fall back to a normal channel message so the
        user still gets an answer — see BOT_PERMISSIONS.md to restore threaded replies.
        """
        text = text[:1999]
        ref = message.to_reference(fail_if_not_exists=False)
        try:
            await message.channel.send(text, reference=ref, mention_author=False)
        except Forbidden:
            try:
                await message.channel.send(text, reference=message, mention_author=False)
            except Forbidden:
                logger.warning(
                    "Reply blocked (likely no Read Message History); sending plain message. "
                    f"channel_id={message.channel.id} guild_id={getattr(message.guild, 'id', None)}"
                )
                await message.channel.send(text)

    def _post_process_mention_reply(self, text: str) -> str:
        text = self._strip_transcript_format(text)
        text = re.sub(r"^sigrok:\s*", "", text, flags=re.IGNORECASE).strip()
        text = self._strip_bot_mention(text)
        return self._normalize_bot_response(text)

    def _mention_reply_is_failure(self, question: str, r: str) -> bool:
        r = r.strip()
        prompt = SIGROK_PERSONALITY_SYSTEM_PROMPT.strip()
        return (
            not r
            or r.lower() == question.lower().strip()
            or r == prompt
            or r in {"not worth my time", "I couldn't answer that right now."}
        )

    @staticmethod
    def _deferred_self_post_llm_prompt(scheduled: str) -> str:
        """
        Frame @schedule / cron prompts so the model writes a channel post as Sigrok,
        not a reply to a user ping.
        """
        text = scheduled.strip()
        return (
            "You are not answering anyone or reacting to a ping. You are Sigrok posting in this "
            "channel of your own accord — write in your voice as a normal message.\n\n"
            f"You decide to: {text}\n\n"
            "Output only the text you send in Discord (no preamble, no addressing the scheduler, "
            "no 'In response to' or similar)."
        )

    async def _react_to_failed_llm_response(self, message: Message) -> None:
        for emoji in ("🫃", "❌"):
            try:
                await message.add_reaction(emoji)
            except (Forbidden, HTTPException) as exc:
                logger.warning(
                    f"Failed to add fallback reaction {emoji} to message {message.id}: {exc}"
                )
                return

    def _normalize_bot_response(self, text: str) -> str:
        normalized = text.strip()
        if len(normalized) >= 2 and normalized[0] == normalized[-1] and normalized[0] in {'"', "'"}:
            normalized = normalized[1:-1].strip()
        normalized = re.sub(r"\s+\n", "\n", normalized)
        return normalized

    async def _handle_bot_mention(self, message: Message) -> None:
        raw_stripped = self._strip_bot_mention_raw(message.content or "")
        if re.search(r"(?i)@schedule\b", raw_stripped):
            sched = self.bot.get_cog("ConditionalPosts")
            if sched is not None and await sched.handle_schedule_mention(message):
                return

        question = self._strip_bot_mention(message.clean_content)
        has_images = genai.client._message_has_images(message)
        if (
            not has_images
            and message.reference
            and message.reference.message_id is not None
        ):
            try:
                ref = await message.channel.fetch_message(message.reference.message_id)
                has_images = genai.client._message_has_images(ref)
            except Exception:
                pass
        preview = question if len(question) <= 80 else f"{question[:77]}..."
        logger.info(
            f"Mention received in guild={getattr(message.guild, 'id', None)} "
            f"channel={getattr(message.channel, 'id', None)} message={message.id}: {preview}"
        )
        logger.debug(f"Full mention question for message={message.id}: {question}")
        if not question:
            if has_images:
                question = "Describe this image."
            else:
                async with _safe_typing(message.channel):
                    await self._send_response_to_ping(message, "Ask me something after the ping.")
                return

        use_max_thinking = is_supersigrok(message.author)
        if not use_max_thinking:
            cfg = settings.bot.supersigrok
            decision = await db.check_and_consume_free_usage(
                message.author.id,
                replies_per_window=cfg.free_replies_per_window,
                window_minutes=cfg.free_window_minutes,
            )
            if decision.silent:
                logger.info(
                    f"Free-user rate limit silent drop user={message.author.id} "
                    f"message={message.id}"
                )
                return
            if decision.send_lockout_reply:
                remaining = humanize_duration(decision.remaining_seconds)
                async with _safe_typing(message.channel):
                    try:
                        response = await genai.client.answer_rate_limit_cooldown(
                            remaining_human=remaining
                        )
                    except Exception:
                        logger.exception(
                            f"Cheap rate-limit reply failed user={message.author.id}"
                        )
                        response = RATE_LIMIT_FALLBACK_REPLY
                    response = self._post_process_mention_reply(response or "")
                    if not response.strip():
                        response = RATE_LIMIT_FALLBACK_REPLY
                    response = apply_nsfw_filter(response)
                    await self._send_response_to_ping(message, response)
                return

        # The daily Max Thinking pass is independent and never blocks or posts to Discord.
        await self._maybe_schedule_relationship_reflection(message)

        # Do not filter transcript context by user ids. Conversation-level prompts
        # need the full recent exchange, not just the asker or mentioned users.
        context_user_ids: Optional[set[int]] = None
        channel_rules = await fetch_channel_rules(message.channel, self.bot.user)
        if use_max_thinking:
            logger.info(
                f"SuperSigrok Max Thinking for user={message.author.id} "
                f"message={message.id}"
            )

        async with _safe_typing(message.channel):
            response = await genai.client.answer_message_question(
                message,
                question,
                context_user_ids,
                channel_rules=channel_rules or None,
                max_thinking=use_max_thinking,
            )
        response = self._post_process_mention_reply(response)
        if not use_max_thinking:
            response = apply_nsfw_filter(response)
        r = response.strip()
        if self._mention_reply_is_failure(question, r):
            await self._react_to_failed_llm_response(message)
            return
        if use_max_thinking:
            response = badge_reply(response)
        await self._send_response_to_ping(message, response)

    async def run_deferred_mention_reply(
        self,
        message: Message,
        question: str,
        *,
        history_before: Optional[datetime] = None,
    ) -> None:
        """
        Run the normal @Sigrok genai reply using a stored prompt (for ConditionalPosts jobs).
        The bot reply is still threaded to the original schedule message (Discord reference).

        Transcript cutoff: if history_before is omitted, uses current time UTC so the SLM sees
        recent channel activity at fire time. Passing history_before overrides that.
        Context is capped to a few recent human turns; reply-chain ancestors are not merged in.
        """
        has_images = genai.client._message_has_images(message)
        if (
            not has_images
            and message.reference
            and message.reference.message_id is not None
        ):
            try:
                ref = await message.channel.fetch_message(message.reference.message_id)
                has_images = genai.client._message_has_images(ref)
            except Exception:
                pass

        if not question.strip():
            async with _safe_typing(message.channel):
                await self._send_response_to_ping(message, "Ask me something after the ping.")
            return

        llm_question = self._deferred_self_post_llm_prompt(question)
        effective_history_before = (
            history_before if history_before is not None else datetime.now(timezone.utc)
        )

        context_user_ids: Optional[set[int]] = None
        channel_rules = await fetch_channel_rules(message.channel, self.bot.user)
        use_max_thinking = is_supersigrok(message.author)

        async with _safe_typing(message.channel):
            response = await genai.client.answer_message_question(
                message,
                llm_question,
                context_user_ids,
                history_before=effective_history_before,
                recent_context_human_turns=self._DEFERRED_SCHEDULE_RECENT_HUMAN_TURNS,
                merge_reply_chain=False,
                channel_rules=channel_rules or None,
                max_thinking=use_max_thinking,
            )
        response = self._post_process_mention_reply(response)
        if not use_max_thinking:
            response = apply_nsfw_filter(response)
        r = response.strip()
        if self._mention_reply_is_failure(llm_question, r):
            await self._react_to_failed_llm_response(message)
            return
        if use_max_thinking:
            response = badge_reply(response)
        await self._send_response_to_ping(message, response)

    @commands.Cog.listener()
    async def on_message(self, message: Message) -> None:
        try:
            if message.author.bot:
                return

            # SuperSigrok DMs (and upsell for everyone else). Prefix commands skip this.
            if message.guild is None:
                content = (message.content or "").strip()
                prefix = settings.bot.prefix
                if content.startswith(prefix):
                    return
                if is_supersigrok(message.author):
                    await self._handle_bot_mention(message)
                else:
                    await message.channel.send(
                        "DMs are SuperSigrok-only. "
                        f"Run `{prefix}supersigrok` to subscribe."
                    )
                return

            channel_whitelisted = self._is_whitelisted_channel(message)
            guild_whitelisted = self._is_whitelisted_guild(message)
            guild_authorized = await self._is_authorized_guild(message)
            is_mention = bool(self.bot.user and self.bot.user in message.mentions)

            if is_mention or channel_whitelisted:
                logger.info(
                    f"on_message guild={getattr(message.guild, 'id', None)} "
                    f"channel={getattr(message.channel, 'id', None)} "
                    f"mention={is_mention} channel_whitelisted={channel_whitelisted} "
                    f"guild_whitelisted={guild_whitelisted} "
                    f"guild_authorized={guild_authorized} message={message.id}"
                )

            # @Sigrok: respond in any channel of an authorized server
            # (TOML whitelist or SuperSigrok subscriber guild).
            if is_mention:
                if guild_authorized:
                    voice_cog = self.bot.get_cog("VoiceRecCog")
                    if voice_cog is not None:
                        handled = await voice_cog.handle_mention_voice_phrase(message)
                        if handled:
                            return
                    await self._handle_bot_mention(message)
                return

            if not channel_whitelisted:
                return

            return
        except Exception as exc:
            logger.exception(f"Unhandled error in on_message: {exc}")


def setup(bot):
    bot.add_cog(Misc(bot))
