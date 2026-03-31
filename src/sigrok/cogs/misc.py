import asyncio
import re
from datetime import datetime

from discord import Message
from discord.errors import Forbidden, HTTPException
from discord.ext import commands
from loguru import logger

from sigrok import db, genai
from sigrok.config import settings
from sigrok.genai import SIGROK_PERSONALITY_SYSTEM_PROMPT


class Misc(commands.Cog):
    bot: commands.Bot
    IQ_BOT_APP_ID = 1361882951935197244
    _RESPONSE_STOP_WORDS = {
        "a",
        "about",
        "an",
        "and",
        "are",
        "for",
        "from",
        "have",
        "how",
        "if",
        "into",
        "its",
        "just",
        "like",
        "more",
        "not",
        "that",
        "the",
        "their",
        "them",
        "they",
        "this",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
        "you",
        "your",
    }

    def __init__(self, bot):
        self.bot = bot
        self.respect_cooldowns: dict[tuple[int, int], datetime] = {}

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

    def _is_iq_bot_bet_challenge(self, message: Message) -> bool:
        if self.bot.user is None or message.guild is None:
            return False
        if not self._is_whitelisted_guild(message):
            return False
        if not self._message_mentions_self(message):
            return False

        author_id = getattr(message.author, "id", None)
        application_id = getattr(message, "application_id", None)
        if author_id != self.IQ_BOT_APP_ID and application_id != self.IQ_BOT_APP_ID:
            return False

        content = message.content.lower()
        required_fragments = (
            "you have been challenged by",
            "to bet iq",
            "do you accept",
        )
        return all(fragment in content for fragment in required_fragments)

    async def _auto_accept_iq_bot_bet(self, message: Message) -> bool:
        if not self._is_iq_bot_bet_challenge(message):
            return False

        for reaction in message.reactions:
            if str(reaction.emoji) == "✅" and reaction.me:
                return True

        try:
            await asyncio.sleep(2)

            for reaction in message.reactions:
                if str(reaction.emoji) == "✅" and reaction.me:
                    return True

            await message.add_reaction("✅")
            logger.info(
                f"Auto-accepted IQ Bot challenge in guild={getattr(message.guild, 'id', None)} "
                f"channel={getattr(message.channel, 'id', None)} message={message.id}"
            )
        except (Forbidden, HTTPException) as exc:
            logger.warning(
                f"Failed to auto-accept IQ Bot challenge message {message.id}: {exc}"
            )
        return True

    def _extract_target_user_ids(self, message: Message, question: str) -> set[int]:
        target_user_ids = {
            member.id
            for member in message.mentions
            if self.bot.user is None or member.id != self.bot.user.id
        }
        if message.guild is None:
            return target_user_ids

        lowered_question = question.lower()
        for member in getattr(message.guild, "members", []):
            if member.bot or member.id in target_user_ids:
                continue

            candidates = {member.name.lower()}
            if member.display_name:
                candidates.add(member.display_name.lower())

            for candidate in candidates:
                if len(candidate) < 3:
                    continue
                if re.search(rf"(?<!\w){re.escape(candidate)}(?!\w)", lowered_question):
                    target_user_ids.add(member.id)
                    break

        return target_user_ids

    def _should_score_message(self, message: Message) -> bool:
        if not settings.genai.respect.enabled:
            return False
        if message.guild is None or not self._is_whitelisted_channel(message):
            return False
        if message.author.bot:
            return False
        if self.bot.user and self.bot.user in message.mentions:
            return False

        content = message.content.strip()
        if len(content) < settings.genai.respect.min_chars:
            return False
        if len(content.split()) < settings.genai.respect.min_words:
            return False

        cooldown_key = (message.guild.id, message.author.id)
        last_scored_at = self.respect_cooldowns.get(cooldown_key)
        if last_scored_at is None:
            return True

        elapsed_seconds = (message.created_at - last_scored_at).total_seconds()
        return elapsed_seconds >= settings.genai.respect.cooldown_seconds

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

    async def _react_to_failed_llm_response(self, message: Message) -> None:
        for emoji in ("🫃", "❌"):
            try:
                await message.add_reaction(emoji)
            except (Forbidden, HTTPException) as exc:
                logger.warning(
                    f"Failed to add fallback reaction {emoji} to message {message.id}: {exc}"
                )
                return

    @classmethod
    def _response_keywords(cls, text: str) -> set[str]:
        return {
            word
            for word in re.findall(r"[a-z0-9']+", text.lower())
            if len(word) >= 3 and word not in cls._RESPONSE_STOP_WORDS
        }

    def _normalize_bot_response(self, text: str) -> str:
        normalized = text.strip()
        if len(normalized) >= 2 and normalized[0] == normalized[-1] and normalized[0] in {'"', "'"}:
            normalized = normalized[1:-1].strip()
        normalized = re.sub(r"\s+\n", "\n", normalized)
        return normalized

    def _response_needs_retry(
        self, question: str, response: str, *, is_reply_chain: bool = False
    ) -> bool:
        return False

    async def _handle_bot_mention(self, message: Message) -> None:
        question = self._strip_bot_mention(message.clean_content)
        logger.info(
            f"Mention received in guild={getattr(message.guild, 'id', None)} "
            f"channel={getattr(message.channel, 'id', None)} message={message.id}: {question}"
        )
        if not question:
            async with message.channel.typing():
                await self._send_response_to_ping(message, "Ask me something after the ping.")
            return

        target_user_ids = self._extract_target_user_ids(message, question)
        if target_user_ids:
            # Keep asker in transcript when filtering by @user / name (not role-based; was excluding pinger)
            target_user_ids = target_user_ids | {message.author.id}
        is_reply_chain = bool(message.reference and message.reference.message_id)
        has_images = genai.client._message_has_images(message)
        if not has_images and is_reply_chain:
            try:
                ref = await message.channel.fetch_message(message.reference.message_id)
                has_images = genai.client._message_has_images(ref)
            except Exception:
                pass
        async with message.channel.typing():
            response = await genai.client.answer_message_question(
                message, question, target_user_ids or None
            )
            response = self._strip_transcript_format(response)
            response = re.sub(r"^sigrok:\s*", "", response, flags=re.IGNORECASE).strip()
            response = self._strip_bot_mention(response)
            response = self._normalize_bot_response(response)
            if not has_images and self._response_needs_retry(question, response, is_reply_chain=is_reply_chain):
                logger.info(
                    f"Retrying mention reply for message={message.id} due to likely stale or assistant-style response."
                )
                response = await genai.client.answer_message_question(
                    message,
                    question,
                    target_user_ids or None,
                    retry_hint=(
                        "Answer only the latest message. Keep it brief and in character. "
                        "If ambiguous, ask one short clarifying question."
                    ),
                )
                response = self._strip_transcript_format(response)
                response = re.sub(r"^sigrok:\s*", "", response, flags=re.IGNORECASE).strip()
                response = self._strip_bot_mention(response)
                response = self._normalize_bot_response(response)
            r = response.strip()
            if (
                not r
                or r.lower() == question.lower().strip()
                or (len(r) < 200 and r in SIGROK_PERSONALITY_SYSTEM_PROMPT)
                or r in {"not worth my time", "I couldn't answer that right now."}
            ):
                await self._react_to_failed_llm_response(message)
                return
            await self._send_response_to_ping(message, response)

    async def _handle_respect_scoring(self, message: Message) -> None:
        if not self._should_score_message(message):
            return

        assert message.guild is not None
        cooldown_key = (message.guild.id, message.author.id)
        delta, reason = await genai.client.score_message_respect(message)
        self.respect_cooldowns[cooldown_key] = message.created_at

        if delta == 0:
            logger.info(
                f"No IQ change for {message.author.name} ({message.author.id}): {reason}"
            )
            return

        updated_user = await db.adjust_user_iq(message.guild.id, message.author.id, delta)
        logger.info(
            f"Adjusted IQ for {message.author.name} ({message.author.id}) by {delta}. "
            f"New IQ: {updated_user.iq}. Reason: {reason}"
        )

    @commands.Cog.listener()
    async def on_message(self, message: Message) -> None:
        try:
            if await self._auto_accept_iq_bot_bet(message):
                return

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

            # IQ changes now come from hidden modifiers emitted during model-driven
            # reply/judgement flows, rather than a separate score call on every message.
            return
        except Exception as exc:
            logger.exception(f"Unhandled error in on_message: {exc}")


def setup(bot):
    bot.add_cog(Misc(bot))
