import re
from datetime import datetime

from discord import Message
from discord.ext import commands
from loguru import logger

from iqbot import db, genai
from iqbot.config import settings


class Misc(commands.Cog):
    bot: commands.Bot

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
            entry.guild == message.guild.id and entry.channel == channel_id
            for entry in settings.bot.whitelist
        )

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

    def _extract_target_user_ids(self, message: Message, question: str) -> set[int]:
        target_user_ids = {
            member.id
            for member in message.mentions
            if self.bot.user is None or member.id != self.bot.user.id
        }
        if message.guild is None:
            return target_user_ids

        lowered_question = question.lower()
        for member in message.guild.members:
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

    async def _handle_bot_mention(self, message: Message) -> None:
        question = self._strip_bot_mention(message.clean_content)
        if not question:
            await message.reply("Ask me something after the ping.", mention_author=False)
            return

        target_user_ids = self._extract_target_user_ids(message, question)
        async with message.channel.typing():
            response = await genai.client.answer_message_question(
                message, question, target_user_ids or None
            )

        snippet = await self._format_reply_chain_snippet(message)
        final_text = f"{snippet}\n\n{response}" if snippet else response
        await message.reply(final_text[:1999], mention_author=False)

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
        if message.author.bot or not self._is_whitelisted_channel(message):
            return

        if self.bot.user and self.bot.user in message.mentions:
            await self._handle_bot_mention(message)
            return

        await self._handle_respect_scoring(message)


def setup(bot):
    bot.add_cog(Misc(bot))
