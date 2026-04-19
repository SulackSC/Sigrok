import asyncio
import re
import time
from datetime import datetime, timezone
from typing import Optional

from discord import Message
from discord.errors import Forbidden, HTTPException
from discord.ext import commands
from loguru import logger

from sigrok import genai
from sigrok.config import settings
from sigrok.genai import GenAILlamaCpp, SIGROK_PERSONALITY_SYSTEM_PROMPT


class Misc(commands.Cog):
    bot: commands.Bot
    # Scheduled @schedule jobs: cap how many human turns feed the SLM (plus bot lines in window).
    _DEFERRED_SCHEDULE_RECENT_HUMAN_TURNS = 5
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

    async def _send_ping_returning_message(self, message: Message, text: str) -> Message:
        """Same routing as _send_response_to_ping but returns the sent Message for edits."""
        text = text[:1999]
        ref = message.to_reference(fail_if_not_exists=False)
        try:
            return await message.channel.send(text, reference=ref, mention_author=False)
        except Forbidden:
            try:
                return await message.channel.send(
                    text, reference=message, mention_author=False
                )
            except Forbidden:
                logger.warning(
                    "Reply blocked (likely no Read Message History); sending plain message. "
                    f"channel_id={message.channel.id} guild_id={getattr(message.guild, 'id', None)}"
                )
                return await message.channel.send(text)

    @staticmethod
    async def _safe_edit_mention_message(msg: Message, content: str) -> None:
        content = (content or "…")[:1999]
        try:
            await msg.edit(content=content)
        except HTTPException as exc:
            if getattr(exc, "status", None) == 429:
                await asyncio.sleep(2.0)
                try:
                    await msg.edit(content=content)
                except Exception as e2:
                    logger.warning(f"Streaming reply edit retry failed: {e2}")
            else:
                logger.warning(f"Streaming reply edit failed: {exc}")

    @staticmethod
    def _mention_streaming_eligible(has_images: bool) -> bool:
        # Streaming disabled globally: tools (search_web/fetch_url) only fire on
        # the non-streaming path, and bulk delivery is fine.
        return False

    def _post_process_mention_reply(self, text: str) -> str:
        text = self._strip_transcript_format(text)
        text = re.sub(r"^sigrok:\s*", "", text, flags=re.IGNORECASE).strip()
        text = self._strip_bot_mention(text)
        return self._normalize_bot_response(text)

    def _mention_reply_is_failure(self, question: str, r: str) -> bool:
        r = r.strip()
        return (
            not r
            or r.lower() == question.lower().strip()
            or (len(r) < 200 and r in SIGROK_PERSONALITY_SYSTEM_PROMPT)
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

    async def _streaming_mention_turn(
        self,
        message: Message,
        question: str,
        target_user_ids: Optional[set[int]],
        retry_hint: Optional[str],
        typing_ctx: Optional[object] = None,
        history_before: Optional[datetime] = None,
        *,
        recent_context_human_turns: Optional[int] = None,
        merge_reply_chain: bool = True,
    ) -> tuple[Optional[Message], str]:
        """
        Stream one llama.cpp completion into a reply message. Returns (reply_msg, processed_text).
        processed_text empty means hard failure (placeholder already updated when possible).
        typing_ctx: if provided, an already-entered typing context to exit after first post.
        """
        interval = float(settings.genai.discord_streaming.edit_interval_seconds)
        agen = genai.client.answer_message_question_streaming(
            message,
            question,
            target_user_ids,
            retry_hint=retry_hint,
            history_before=history_before,
            recent_context_human_turns=recent_context_human_turns,
            merge_reply_chain=merge_reply_chain,
        )
        _strip = getattr(
            genai.client, "_strip_think_block", lambda c: (c or "").strip()
        )

        try:
            first = await agen.__anext__()
        except StopAsyncIteration:
            return None, ""
        reply_msg: Optional[Message] = None
        owns_typing = typing_ctx is None
        if owns_typing:
            typing_ctx = message.channel.typing()
            await typing_ctx.__aenter__()
        typing_active = True

        def _visible(raw: str) -> str:
            text = _strip(raw)
            return text.strip() or "…"

        async def _stop_typing() -> None:
            nonlocal typing_active
            if typing_active and typing_ctx is not None:
                try:
                    await typing_ctx.__aexit__(None, None, None)
                except Exception:
                    pass
                typing_active = False

        try:
            delta_count = 0
            typing_start = time.monotonic()
            acc = first
            async for delta in agen:
                acc += delta
                delta_count += 1
                now = time.monotonic()
                if reply_msg is None:
                    if now - typing_start >= interval:
                        reply_msg = await self._send_ping_returning_message(message, _visible(acc))
                        logger.info(
                            f"Streaming first post for message={message.id} "
                            f"after {now - typing_start:.1f}s, {delta_count} deltas, {len(acc)} chars"
                        )
                        await _stop_typing()
                        last_edit = now
                else:
                    if now - last_edit >= interval:
                        await self._safe_edit_mention_message(reply_msg, _visible(acc))
                        last_edit = now
            await _stop_typing()
            logger.info(
                f"Streaming done for message={message.id}: "
                f"{delta_count} deltas, {len(acc)} chars total, posted={'yes' if reply_msg else 'no'}"
            )
        except Exception as exc:
            logger.error(f"Streaming mention failed: {exc}")
            await _stop_typing()
            if reply_msg is not None:
                await self._safe_edit_mention_message(
                    reply_msg, "I couldn't answer that right now."
                )
            return reply_msg, ""

        final_raw = acc.strip()
        if not final_raw:
            if reply_msg is not None:
                await self._safe_edit_mention_message(
                    reply_msg, "I couldn't answer that right now."
                )
            return reply_msg, ""

        out = self._post_process_mention_reply(_strip(final_raw))
        if reply_msg is None:
            reply_msg = await self._send_ping_returning_message(message, out[:1999])
        else:
            await self._safe_edit_mention_message(reply_msg, out[:1999])
        return reply_msg, out

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
        logger.info(
            f"Mention received in guild={getattr(message.guild, 'id', None)} "
            f"channel={getattr(message.channel, 'id', None)} message={message.id}: {question}"
        )
        if not question:
            if has_images:
                question = "Describe this image."
            else:
                async with message.channel.typing():
                    await self._send_response_to_ping(message, "Ask me something after the ping.")
                return

        target_user_ids = self._extract_target_user_ids(message, question)
        target_user_ids = target_user_ids | {message.author.id}
        is_reply_chain = bool(message.reference and message.reference.message_id)
        if not has_images and is_reply_chain:
            try:
                ref = await message.channel.fetch_message(message.reference.message_id)
                has_images = genai.client._message_has_images(ref)
            except Exception:
                pass

        stream_ok = self._mention_streaming_eligible(has_images)
        reply_msg: Optional[Message] = None
        response = ""

        if stream_ok:
            typing_ctx = message.channel.typing()
            await typing_ctx.__aenter__()
            try:
                reply_msg, response = await self._streaming_mention_turn(
                    message, question, target_user_ids or None,
                    retry_hint=None, typing_ctx=typing_ctx,
                )
            except RuntimeError as exc:
                logger.warning(f"Mention streaming unavailable ({exc}); falling back.")
                try:
                    await typing_ctx.__aexit__(None, None, None)
                except Exception:
                    pass
                stream_ok = False

            if stream_ok and response and not self._mention_reply_is_failure(question, response):
                if not has_images and self._response_needs_retry(
                    question, response, is_reply_chain=is_reply_chain
                ):
                    logger.info(
                        f"Retrying streaming mention reply for message={message.id} "
                        "due to likely stale or assistant-style response."
                    )
                    if reply_msg is not None:
                        try:
                            await reply_msg.delete()
                        except Exception:
                            pass
                    reply_msg, response = await self._streaming_mention_turn(
                        message,
                        question,
                        target_user_ids or None,
                        retry_hint=(
                            "Answer only the latest message. Keep it brief and in character. "
                            "If ambiguous, ask one short clarifying question."
                        ),
                    )
                r = response.strip()
                if self._mention_reply_is_failure(question, r):
                    await self._react_to_failed_llm_response(message)
                    if reply_msg is not None:
                        try:
                            await reply_msg.delete()
                        except Exception:
                            pass
                    return
                return

            if stream_ok and reply_msg is not None:
                try:
                    await reply_msg.delete()
                except Exception:
                    pass

        async with message.channel.typing():
            response = await genai.client.answer_message_question(
                message, question, target_user_ids or None
            )
        response = self._post_process_mention_reply(response)
        if not has_images and self._response_needs_retry(
            question, response, is_reply_chain=is_reply_chain
        ):
            logger.info(
                f"Retrying mention reply for message={message.id} due to likely stale or assistant-style response."
            )
            async with message.channel.typing():
                response = await genai.client.answer_message_question(
                    message,
                    question,
                    target_user_ids or None,
                    retry_hint=(
                        "Answer only the latest message. Keep it brief and in character. "
                        "If ambiguous, ask one short clarifying question."
                    ),
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
            async with message.channel.typing():
                await self._send_response_to_ping(message, "Ask me something after the ping.")
            return

        llm_question = self._deferred_self_post_llm_prompt(question)
        effective_history_before = (
            history_before if history_before is not None else datetime.now(timezone.utc)
        )

        target_user_ids = self._extract_target_user_ids(message, question)
        target_user_ids = target_user_ids | {message.author.id}
        is_reply_chain = bool(message.reference and message.reference.message_id)
        if not has_images and is_reply_chain:
            try:
                ref = await message.channel.fetch_message(message.reference.message_id)
                has_images = genai.client._message_has_images(ref)
            except Exception:
                pass

        stream_ok = self._mention_streaming_eligible(has_images)
        reply_msg: Optional[Message] = None
        response = ""

        if stream_ok:
            typing_ctx = message.channel.typing()
            await typing_ctx.__aenter__()
            try:
                reply_msg, response = await self._streaming_mention_turn(
                    message,
                    llm_question,
                    target_user_ids or None,
                    retry_hint=None,
                    typing_ctx=typing_ctx,
                    history_before=effective_history_before,
                    recent_context_human_turns=self._DEFERRED_SCHEDULE_RECENT_HUMAN_TURNS,
                    merge_reply_chain=False,
                )
            except RuntimeError as exc:
                logger.warning(f"Mention streaming unavailable ({exc}); falling back.")
                try:
                    await typing_ctx.__aexit__(None, None, None)
                except Exception:
                    pass
                stream_ok = False

            if stream_ok and response and not self._mention_reply_is_failure(llm_question, response):
                if not has_images and self._response_needs_retry(
                    llm_question, response, is_reply_chain=is_reply_chain
                ):
                    logger.info(
                        f"Retrying streaming mention reply for message={message.id} "
                        "due to likely stale or assistant-style response."
                    )
                    if reply_msg is not None:
                        try:
                            await reply_msg.delete()
                        except Exception:
                            pass
                    reply_msg, response = await self._streaming_mention_turn(
                        message,
                        llm_question,
                        target_user_ids or None,
                        retry_hint=None,
                        history_before=effective_history_before,
                        recent_context_human_turns=self._DEFERRED_SCHEDULE_RECENT_HUMAN_TURNS,
                        merge_reply_chain=False,
                    )
                r = response.strip()
                if self._mention_reply_is_failure(llm_question, r):
                    await self._react_to_failed_llm_response(message)
                    if reply_msg is not None:
                        try:
                            await reply_msg.delete()
                        except Exception:
                            pass
                    return
                return

            if stream_ok and reply_msg is not None:
                try:
                    await reply_msg.delete()
                except Exception:
                    pass

        async with message.channel.typing():
            response = await genai.client.answer_message_question(
                message,
                llm_question,
                target_user_ids or None,
                history_before=effective_history_before,
                recent_context_human_turns=self._DEFERRED_SCHEDULE_RECENT_HUMAN_TURNS,
                merge_reply_chain=False,
            )
        response = self._post_process_mention_reply(response)
        if not has_images and self._response_needs_retry(
            llm_question, response, is_reply_chain=is_reply_chain
        ):
            logger.info(
                f"Retrying mention reply for message={message.id} due to likely stale or assistant-style response."
            )
            async with message.channel.typing():
                response = await genai.client.answer_message_question(
                    message,
                    llm_question,
                    target_user_ids or None,
                    retry_hint=None,
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
