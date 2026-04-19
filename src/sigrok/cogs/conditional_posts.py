"""Conditional and scheduled Discord posts.

Implements three mechanisms:

1. TOML `event_posts` — fire on `on_member_join` / `on_member_remove`.
2. TOML `timed_posts` — periodic channel posts on a fixed interval.
3. `@Sigrok @schedule ...` mentions — controller-only one-shot deferred
   replies (Discord timestamp token `<t:UNIX[:style]>`) or repeating cron
   posts (5-field cron expression). Persisted in the
   `schedule_mention_jobs` table so jobs survive restart.

NOTE: This cog was rebuilt from `misc.py`, `db.py`, and the original
plan in `.cursor/plans/conditional_discord_posts_4cd6843c.plan.md`
after the on-disk source was lost. Behavior matches the plan but
should be reviewed before relying on it in production.
"""

from __future__ import annotations

import asyncio
import re
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

from discord import Member, Message, TextChannel
from discord.ext import commands, tasks
from loguru import logger

try:
    from croniter import croniter
except Exception:  # pragma: no cover - dependency optional at import time
    croniter = None  # type: ignore[assignment]

from sigrok import db
from sigrok.config import settings


_DISCORD_TIMESTAMP_RE = re.compile(r"<t:(?P<ts>-?\d+)(?::[a-zA-Z])?>")
# Cron tokens: 5 whitespace-separated fields composed of digits, *, /, ,, -.
_CRON_TOKEN = r"[\d\*\/,\-]+"
_CRON_RE = re.compile(
    rf"^(?P<cron>{_CRON_TOKEN}\s+{_CRON_TOKEN}\s+{_CRON_TOKEN}\s+{_CRON_TOKEN}\s+{_CRON_TOKEN})(?:\s+|$)"
)
_SCHEDULE_MARKER_RE = re.compile(r"(?i)@schedule\b")


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _to_naive_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


def _to_aware_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _new_job_id() -> str:
    return secrets.token_hex(8)


def _format_user_template(template: str, member: Member) -> str:
    return (
        template.replace("{user_mention}", member.mention)
        .replace("{user}", member.display_name)
        .replace("{guild}", member.guild.name if member.guild else "")
    )


class ConditionalPosts(commands.Cog):
    """TOML join/leave + interval posts and `@schedule` mention jobs."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self._timed_state: dict[tuple[int, int], datetime] = {}
        self._ack_to_job: dict[int, str] = {}
        self._lock = asyncio.Lock()
        self.scheduler_loop.start()

    def cog_unload(self) -> None:
        self.scheduler_loop.cancel()

    # ------------------------------------------------------------------
    # Authorization
    # ------------------------------------------------------------------

    def _controller_ids(self) -> set[int]:
        ids = getattr(settings.bot, "schedule_controller_user_ids", None) or []
        return {int(x) for x in ids}

    def _is_controller(self, user_id: int) -> bool:
        return user_id in self._controller_ids()

    def _is_whitelisted_guild(self, guild_id: Optional[int]) -> bool:
        if guild_id is None:
            return False
        return any(entry.guild == guild_id for entry in settings.bot.whitelist)

    # ------------------------------------------------------------------
    # TOML event_posts
    # ------------------------------------------------------------------

    @commands.Cog.listener()
    async def on_member_join(self, member: Member) -> None:
        await self._handle_member_event(member, "join")

    @commands.Cog.listener()
    async def on_member_remove(self, member: Member) -> None:
        await self._handle_member_event(member, "leave")

    async def _handle_member_event(self, member: Member, event: str) -> None:
        if member.guild is None:
            return
        if not self._is_whitelisted_guild(member.guild.id):
            return
        for rule in settings.bot.event_posts:
            if rule.guild != member.guild.id or rule.on != event:
                continue
            if rule.ignore_bots and member.bot:
                continue
            channel = member.guild.get_channel(rule.channel)
            if not isinstance(channel, TextChannel):
                continue
            try:
                await channel.send(_format_user_template(rule.message, member))
            except Exception as exc:
                logger.warning(f"event_post send failed guild={rule.guild} channel={rule.channel}: {exc}")

    # ------------------------------------------------------------------
    # Scheduler loop (timed_posts + persisted @schedule jobs)
    # ------------------------------------------------------------------

    @tasks.loop(seconds=20)
    async def scheduler_loop(self) -> None:
        try:
            await self._tick_timed_posts()
        except Exception as exc:
            logger.warning(f"timed_posts tick failed: {exc}")
        try:
            await self._tick_schedule_jobs()
        except Exception as exc:
            logger.warning(f"schedule_jobs tick failed: {exc}")

    @scheduler_loop.before_loop
    async def _before_loop(self) -> None:
        await self.bot.wait_until_ready()

    async def _tick_timed_posts(self) -> None:
        now = _now_utc()
        for rule in settings.bot.timed_posts:
            if not self._is_whitelisted_guild(rule.guild):
                continue
            key = (rule.guild, rule.channel)
            last = self._timed_state.get(key)
            interval = timedelta(minutes=max(1, int(rule.interval_minutes)))
            if last is None:
                self._timed_state[key] = now
                continue
            if now - last < interval:
                continue
            channel = self._resolve_text_channel(rule.guild, rule.channel)
            if channel is None:
                continue
            try:
                await channel.send(rule.message)
                self._timed_state[key] = now
            except Exception as exc:
                logger.warning(f"timed_post send failed guild={rule.guild} channel={rule.channel}: {exc}")

    async def _tick_schedule_jobs(self) -> None:
        now_naive = _to_naive_utc(_now_utc())
        try:
            jobs = await db.list_schedule_mention_jobs()
        except Exception as exc:
            logger.warning(f"list_schedule_mention_jobs failed: {exc}")
            return
        for job in jobs:
            if not self._is_whitelisted_guild(job.guild_id):
                continue
            if job.kind == "once":
                if job.due_at is None or job.due_at > now_naive:
                    continue
                await self._fire_once_job(job)
            elif job.kind == "cron":
                if job.next_fire is None or job.next_fire > now_naive:
                    continue
                await self._fire_cron_job(job)
            else:
                logger.warning(f"unknown schedule job kind: {job.kind} (job_id={job.job_id})")

    async def _fire_once_job(self, job) -> None:  # type: ignore[no-untyped-def]
        async with self._lock:
            try:
                channel = self._resolve_text_channel(job.guild_id, job.channel_id)
                if channel is None:
                    logger.warning(f"once job channel missing guild={job.guild_id} channel={job.channel_id}")
                    await db.delete_schedule_mention_job(job.job_id)
                    return
                source: Optional[Message] = None
                try:
                    source = await channel.fetch_message(job.message_id)
                except Exception:
                    source = None
                misc = self.bot.get_cog("Misc")
                if misc is None or source is None:
                    try:
                        await channel.send(job.prompt)
                    except Exception as exc:
                        logger.warning(f"once job fallback send failed: {exc}")
                else:
                    try:
                        await misc.run_deferred_mention_reply(source, job.prompt)
                    except Exception as exc:
                        logger.warning(f"once job deferred reply failed: {exc}")
            finally:
                if job.ack_message_id is not None:
                    self._ack_to_job.pop(job.ack_message_id, None)
                try:
                    await db.delete_schedule_mention_job(job.job_id)
                except Exception as exc:
                    logger.warning(f"delete once job failed: {exc}")

    async def _fire_cron_job(self, job) -> None:  # type: ignore[no-untyped-def]
        async with self._lock:
            channel = self._resolve_text_channel(job.guild_id, job.channel_id)
            if channel is not None:
                try:
                    await channel.send(job.prompt)
                except Exception as exc:
                    logger.warning(f"cron job send failed job_id={job.job_id}: {exc}")
            next_fire = self._next_cron_fire(job.cron_expr or "", _now_utc())
            if next_fire is None:
                logger.warning(f"cron job has no next fire, deleting job_id={job.job_id}")
                if job.ack_message_id is not None:
                    self._ack_to_job.pop(job.ack_message_id, None)
                await db.delete_schedule_mention_job(job.job_id)
                return
            try:
                await db.update_schedule_cron_next_fire(job.job_id, next_fire)
            except Exception as exc:
                logger.warning(f"update next_fire failed job_id={job.job_id}: {exc}")

    def _resolve_text_channel(self, guild_id: int, channel_id: int) -> Optional[TextChannel]:
        guild = self.bot.get_guild(guild_id)
        if guild is None:
            return None
        channel = guild.get_channel(channel_id)
        if isinstance(channel, TextChannel):
            return channel
        return None

    @staticmethod
    def _next_cron_fire(cron_expr: str, base: datetime) -> Optional[datetime]:
        if croniter is None or not cron_expr.strip():
            return None
        try:
            itr = croniter(cron_expr.strip(), _to_aware_utc(base))
            return itr.get_next(datetime)
        except Exception as exc:
            logger.warning(f"invalid cron expr '{cron_expr}': {exc}")
            return None

    # ------------------------------------------------------------------
    # @schedule mention parsing (called from misc._handle_bot_mention)
    # ------------------------------------------------------------------

    async def handle_schedule_mention(self, message: Message) -> bool:
        """Return True when this cog handled the mention (suppresses normal reply)."""
        if message.guild is None:
            return False
        if not self._is_whitelisted_guild(message.guild.id):
            return False

        raw = message.content or ""
        marker = _SCHEDULE_MARKER_RE.search(raw)
        if marker is None:
            return False

        if not self._is_controller(message.author.id):
            try:
                await message.reply("not authorized to schedule", mention_author=False)
            except Exception:
                pass
            return True

        tail = raw[marker.end():].lstrip()
        if not tail:
            await self._usage_error(message)
            return True

        # Try Discord timestamp token first: <t:UNIX[:style]>
        ts_match = _DISCORD_TIMESTAMP_RE.match(tail)
        if ts_match is not None:
            unix_ts = int(ts_match.group("ts"))
            prompt = tail[ts_match.end():].strip()
            if not prompt:
                await self._usage_error(message)
                return True
            due = datetime.fromtimestamp(unix_ts, tz=timezone.utc)
            if due <= _now_utc():
                try:
                    await message.reply("that timestamp is in the past", mention_author=False)
                except Exception:
                    pass
                return True
            await self._create_once_job(message, due, prompt)
            return True

        # Else try cron: 5 fields
        cron_match = _CRON_RE.match(tail)
        if cron_match is not None:
            cron_expr = cron_match.group("cron").strip()
            prompt = tail[cron_match.end():].strip()
            if not prompt:
                await self._usage_error(message)
                return True
            next_fire = self._next_cron_fire(cron_expr, _now_utc())
            if next_fire is None:
                try:
                    await message.reply("bad cron expression", mention_author=False)
                except Exception:
                    pass
                return True
            await self._create_cron_job(message, cron_expr, next_fire, prompt)
            return True

        await self._usage_error(message)
        return True

    async def _usage_error(self, message: Message) -> None:
        try:
            await message.reply(
                "usage: `@schedule <t:UNIX[:F]> <prompt>` for one-shot, "
                "or `@schedule <min hr dom mon dow> <prompt>` for cron",
                mention_author=False,
            )
        except Exception:
            pass

    async def _create_once_job(self, message: Message, due: datetime, prompt: str) -> None:
        assert message.guild is not None
        job_id = _new_job_id()
        try:
            ack = await message.reply(
                f"scheduled one-shot for <t:{int(due.timestamp())}:F> (id `{job_id}`)\n"
                f"reply `stop` to this message to cancel",
                mention_author=False,
            )
        except Exception as exc:
            logger.warning(f"failed to send ack for once job: {exc}")
            ack = None
        try:
            await db.insert_schedule_mention_job(
                job_id=job_id,
                kind="once",
                guild_id=message.guild.id,
                channel_id=message.channel.id,
                message_id=message.id,
                creator_id=message.author.id,
                prompt=prompt,
                due_at=due,
                cron_expr=None,
                next_fire=None,
                ack_message_id=ack.id if ack is not None else None,
            )
        except Exception as exc:
            logger.warning(f"insert once job failed: {exc}")
            return
        if ack is not None:
            self._ack_to_job[ack.id] = job_id

    async def _create_cron_job(
        self,
        message: Message,
        cron_expr: str,
        next_fire: datetime,
        prompt: str,
    ) -> None:
        assert message.guild is not None
        job_id = _new_job_id()
        try:
            ack = await message.reply(
                f"scheduled cron `{cron_expr}` (next <t:{int(next_fire.timestamp())}:R>, id `{job_id}`)\n"
                f"reply `stop` to this message to cancel",
                mention_author=False,
            )
        except Exception as exc:
            logger.warning(f"failed to send ack for cron job: {exc}")
            ack = None
        try:
            await db.insert_schedule_mention_job(
                job_id=job_id,
                kind="cron",
                guild_id=message.guild.id,
                channel_id=message.channel.id,
                message_id=message.id,
                creator_id=message.author.id,
                prompt=prompt,
                due_at=None,
                cron_expr=cron_expr,
                next_fire=next_fire,
                ack_message_id=ack.id if ack is not None else None,
            )
        except Exception as exc:
            logger.warning(f"insert cron job failed: {exc}")
            return
        if ack is not None:
            self._ack_to_job[ack.id] = job_id

    # ------------------------------------------------------------------
    # `stop` reply cancellation
    # ------------------------------------------------------------------

    @commands.Cog.listener()
    async def on_message(self, message: Message) -> None:
        if message.author.bot:
            return
        if message.reference is None or message.reference.message_id is None:
            return
        if (message.content or "").strip().lower() != "stop":
            return

        ack_id = message.reference.message_id
        job_id = self._ack_to_job.get(ack_id)
        if job_id is None:
            try:
                jobs = await db.list_schedule_mention_jobs()
            except Exception:
                jobs = []
            for j in jobs:
                if j.ack_message_id == ack_id:
                    job_id = j.job_id
                    break
        if job_id is None:
            return

        if not (
            self._is_controller(message.author.id)
            or await self._is_creator_async(message.author.id, job_id)
            or self._has_admin_perms(message)
        ):
            return

        try:
            await db.delete_schedule_mention_job(job_id)
        except Exception as exc:
            logger.warning(f"delete job on stop failed job_id={job_id}: {exc}")
            return
        self._ack_to_job.pop(ack_id, None)
        try:
            await message.reply(f"stopped `{job_id}`", mention_author=False)
        except Exception:
            pass

    async def _is_creator_async(self, user_id: int, job_id: str) -> bool:
        try:
            jobs = await db.list_schedule_mention_jobs()
        except Exception:
            return False
        for j in jobs:
            if j.job_id == job_id:
                return j.creator_id == user_id
        return False

    @staticmethod
    def _has_admin_perms(message: Message) -> bool:
        try:
            perms = message.author.guild_permissions  # type: ignore[union-attr]
        except Exception:
            return False
        return bool(getattr(perms, "administrator", False) or getattr(perms, "manage_guild", False))


def setup(bot: commands.Bot) -> None:
    bot.add_cog(ConditionalPosts(bot))
