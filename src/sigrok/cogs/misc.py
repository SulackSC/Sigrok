import asyncio
import re
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncIterator, Optional

from discord import Message
from discord.errors import Forbidden, HTTPException
from discord.ext import commands
from loguru import logger

from sigrok import genai
from sigrok.config import settings
from sigrok.genai import GenAILlamaCpp, SIGROK_PERSONALITY_SYSTEM_PROMPT


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

        # Do not filter transcript context by user ids. Conversation-level prompts
        # need the full recent exchange, not just the asker or mentioned users.
        context_user_ids: Optional[set[int]] = None

        async with _safe_typing(message.channel):
            response = await genai.client.answer_message_question(
                message, question, context_user_ids
            )
        response = self._post_process_mention_reply(response)
        r = response.strip()
        if self._mention_reply_is_failure(question, r):
            await self._react_to_failed_llm_response(message)
            return
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

        async with _safe_typing(message.channel):
            response = await genai.client.answer_message_question(
                message,
                llm_question,
                context_user_ids,
                history_before=effective_history_before,
                recent_context_human_turns=self._DEFERRED_SCHEDULE_RECENT_HUMAN_TURNS,
                merge_reply_chain=False,
            )
        response = self._post_process_mention_reply(response)
        r = response.strip()
        if self._mention_reply_is_failure(llm_question, r):
            await self._react_to_failed_llm_response(message)
            return
        await self._send_response_to_ping(message, response)

    @commands.Cog.listener()
    async def on_message(self, message: Message) -> None:
        try:
            if message.author.bot:
                return

            channel_whitelisted = self._is_whitelisted_channel(message)
            guild_whitelisted = self._is_whitelisted_guild(message)
            is_mention = bool(self.bot.user and self.bot.user in message.mentions)

            if is_mention or channel_whitelisted:
                logger.info(
                    f"on_message guild={getattr(message.guild, 'id', None)} "
                    f"channel={getattr(message.channel, 'id', None)} "
                    f"mention={is_mention} channel_whitelisted={channel_whitelisted} "
                    f"guild_whitelisted={guild_whitelisted} message={message.id}"
                )

            # @Sigrok: respond in any channel of a whitelisted server (not only whitelist rows that match this channel).
            if is_mention:
                if guild_whitelisted:
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
