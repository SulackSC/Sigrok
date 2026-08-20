import asyncio
import secrets
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from functools import wraps
from pprint import pformat
from typing import Any, AsyncIterator, Optional

from loguru import logger
from sqlalchemy import (
    CheckConstraint,
    String,
    Text,
    UniqueConstraint,
    and_,
    delete,
    insert,
    or_,
    text,
    update,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.future import select
from sqlalchemy.orm import Mapped, declarative_base, mapped_column

from sigrok.config import settings

Base = declarative_base()

engine = create_async_engine(
    settings.database.url,
    echo=settings.database.echo,
    pool_size=settings.database.pool_size,
    max_overflow=settings.database.max_overflow,
    pool_recycle=settings.database.pool_recycle,
    pool_timeout=settings.database.pool_timeout,
)

async_session = async_sessionmaker(engine, expire_on_commit=False)


class User(Base):
    __tablename__ = "users"
    __table_args__ = (UniqueConstraint("guild_id", "user_id", name="uq_user_guild_user"),)
    __allow_unmapped__ = True

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    guild_id: Mapped[int] = mapped_column(index=True)
    user_id: Mapped[int] = mapped_column(index=True)
    rating: Mapped[Optional[int]] = mapped_column(default=100)
    is_present: Mapped[bool] = mapped_column(default=True)

    def __repr__(self):
        return pformat(self.to_dict())

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "guild_id": self.guild_id,
            "user_id": self.user_id,
            "rating": self.rating,
            "is_present": self.is_present,
        }


class UserRelationship(Base):
    """Guild-local, game-like stance toward a Discord user.

    Keep this table numeric/enum-only. It must never become a profile or a store
    for biographical information.
    """

    __tablename__ = "user_relationships"
    __table_args__ = (
        UniqueConstraint(
            "guild_id", "user_id", name="uq_user_relationship_guild_user"
        ),
        CheckConstraint("affinity BETWEEN 0 AND 200", name="ck_relationship_affinity"),
        CheckConstraint("trust BETWEEN 0 AND 100", name="ck_relationship_trust"),
        CheckConstraint(
            "disposition IN ('ally', 'neutral', 'rival', 'ignore')",
            name="ck_relationship_disposition",
        ),
        CheckConstraint(
            "roast_level BETWEEN 0 AND 3", name="ck_relationship_roast_level"
        ),
        CheckConstraint(
            "engagement_weight BETWEEN 0 AND 100",
            name="ck_relationship_engagement",
        ),
        CheckConstraint(
            "last_vibe IS NULL OR last_vibe IN "
            "('chill', 'friendly', 'spicy', 'hostile', 'correction', 'dismissive')",
            name="ck_relationship_last_vibe",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    guild_id: Mapped[int] = mapped_column(index=True)
    user_id: Mapped[int] = mapped_column(index=True)
    affinity: Mapped[int] = mapped_column(default=100)
    trust: Mapped[int] = mapped_column(default=50)
    disposition: Mapped[str] = mapped_column(String(16), default="neutral")
    roast_level: Mapped[int] = mapped_column(default=1)
    engagement_weight: Mapped[int] = mapped_column(default=100)
    last_vibe: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    updated_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "guild_id": self.guild_id,
            "user_id": self.user_id,
            "affinity": self.affinity,
            "trust": self.trust,
            "disposition": self.disposition,
            "roast_level": self.roast_level,
            "engagement_weight": self.engagement_weight,
            "last_vibe": self.last_vibe,
            "updated_at": (
                self.updated_at.replace(tzinfo=timezone.utc).isoformat()
                if self.updated_at is not None
                else None
            ),
        }


class ChannelRelationshipRun(Base):
    """Atomic lease/cooldown state for a channel's daily reflection pass."""

    __tablename__ = "channel_relationship_runs"

    channel_id: Mapped[int] = mapped_column(primary_key=True)
    guild_id: Mapped[int] = mapped_column(index=True)
    last_attempt_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    last_success_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    lease_until: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    lease_token: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)


class ScheduleMentionJob(Base):
    """Persisted @Sigrok @schedule mention jobs (one-shot and cron)."""

    __tablename__ = "schedule_mention_jobs"
    __allow_unmapped__ = True

    job_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    kind: Mapped[str] = mapped_column(String(8), index=True)  # "once" | "cron"
    guild_id: Mapped[int] = mapped_column(index=True)
    channel_id: Mapped[int] = mapped_column(index=True)
    message_id: Mapped[int] = mapped_column(index=True)
    creator_id: Mapped[int] = mapped_column(index=True)
    prompt: Mapped[str] = mapped_column(Text())
    due_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    cron_expr: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    next_fire: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    ack_message_id: Mapped[Optional[int]] = mapped_column(nullable=True, index=True)

    def __repr__(self) -> str:
        return f"<ScheduleMentionJob {self.job_id} {self.kind}>"


SUPERSIGROK_ACTIVE_STATUSES = frozenset({"active", "trialing"})


class SupersigrokSubscription(Base):
    """Stripe subscription ↔ Discord user for SuperSigrok entitlements."""

    __tablename__ = "supersigrok_subscriptions"

    discord_user_id: Mapped[int] = mapped_column(primary_key=True)
    stripe_customer_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True, index=True)
    stripe_subscription_id: Mapped[Optional[str]] = mapped_column(
        String(128), nullable=True, index=True
    )
    status: Mapped[str] = mapped_column(String(32), default="incomplete", index=True)
    current_period_end: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    created_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    updated_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "discord_user_id": self.discord_user_id,
            "stripe_customer_id": self.stripe_customer_id,
            "stripe_subscription_id": self.stripe_subscription_id,
            "status": self.status,
            "current_period_end": (
                self.current_period_end.replace(tzinfo=timezone.utc).isoformat()
                if self.current_period_end is not None
                else None
            ),
            "created_at": (
                self.created_at.replace(tzinfo=timezone.utc).isoformat()
                if self.created_at is not None
                else None
            ),
            "updated_at": (
                self.updated_at.replace(tzinfo=timezone.utc).isoformat()
                if self.updated_at is not None
                else None
            ),
        }

    @property
    def is_active(self) -> bool:
        return self.status in SUPERSIGROK_ACTIVE_STATUSES


class SubscriberGuild(Base):
    """One guild per SuperSigrok sponsor (not in settings.toml whitelist)."""

    __tablename__ = "subscriber_guilds"
    __table_args__ = (
        UniqueConstraint("sponsor_user_id", name="uq_subscriber_guild_sponsor"),
    )

    guild_id: Mapped[int] = mapped_column(primary_key=True)
    sponsor_user_id: Mapped[int] = mapped_column(index=True)
    added_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    grace_until: Mapped[Optional[datetime]] = mapped_column(nullable=True, index=True)
    grace_notified: Mapped[bool] = mapped_column(default=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "guild_id": self.guild_id,
            "sponsor_user_id": self.sponsor_user_id,
            "added_at": (
                self.added_at.replace(tzinfo=timezone.utc).isoformat()
                if self.added_at is not None
                else None
            ),
            "grace_until": (
                self.grace_until.replace(tzinfo=timezone.utc).isoformat()
                if self.grace_until is not None
                else None
            ),
            "grace_notified": bool(self.grace_notified),
        }


class SponsorAlreadyHasGuildError(RuntimeError):
    """Raised when a SuperSigrok user tries to sponsor a second guild."""

    def __init__(self, existing_guild_id: int) -> None:
        self.existing_guild_id = int(existing_guild_id)
        super().__init__(
            f"sponsor already has guild {self.existing_guild_id}"
        )


class GuildAlreadySponsoredError(RuntimeError):
    """Raised when a guild is already claimed by another SuperSigrok sponsor."""

    def __init__(self, sponsor_user_id: int) -> None:
        self.sponsor_user_id = int(sponsor_user_id)
        super().__init__(
            f"guild already sponsored by user {self.sponsor_user_id}"
        )


class FreeUserUsage(Base):
    """Global free-tier @mention quota window per Discord user."""

    __tablename__ = "free_user_usage"

    user_id: Mapped[int] = mapped_column(primary_key=True)
    window_started_at: Mapped[datetime] = mapped_column()
    full_replies: Mapped[int] = mapped_column(default=0)
    lockout_reply_sent: Mapped[bool] = mapped_column(default=False)


_schedule_tables_created = False
_users_schema_ensured = False
_relationship_tables_created = False
_supersigrok_tables_created = False
_supersigrok_tables_lock = asyncio.Lock()
_subscriber_guild_tables_created = False
_subscriber_guild_tables_lock = asyncio.Lock()
_free_usage_tables_created = False
_free_usage_tables_lock = asyncio.Lock()


def _to_naive_utc(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


async def ensure_user_schema() -> None:
    """Dedupe users and add unique index on existing SQLite databases."""
    global _users_schema_ensured
    if _users_schema_ensured:
        return
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.execute(
            text(
                """
                DELETE FROM users
                WHERE id NOT IN (
                    SELECT MIN(id) FROM users GROUP BY guild_id, user_id
                )
                """
            )
        )
        await conn.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_user_guild_user "
                "ON users (guild_id, user_id)"
            )
        )
    _users_schema_ensured = True


async def ensure_schedule_tables() -> None:
    """Create schedule_mention_jobs if missing (does not drop existing tables)."""
    global _schedule_tables_created
    if _schedule_tables_created:
        return
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    _schedule_tables_created = True


async def ensure_relationship_tables() -> None:
    """Create relationship and reflection-run tables without altering users."""
    global _relationship_tables_created
    if _relationship_tables_created:
        return
    async with engine.begin() as conn:
        # Cooldown moved from guild-scoped to channel-scoped; drop the old table.
        await conn.execute(text("DROP TABLE IF EXISTS guild_relationship_runs"))
        await conn.run_sync(Base.metadata.create_all)
    _relationship_tables_created = True


async def ensure_supersigrok_tables() -> None:
    """Create supersigrok_subscriptions if missing (safe under concurrent startup)."""
    global _supersigrok_tables_created
    async with _supersigrok_tables_lock:
        if _supersigrok_tables_created:
            return
        try:
            async with engine.begin() as conn:

                def _create(sync_conn) -> None:
                    SupersigrokSubscription.__table__.create(
                        sync_conn, checkfirst=True
                    )

                await conn.run_sync(_create)
        except Exception as exc:
            # Concurrent create_all from other ensure_* helpers can race on SQLite.
            msg = str(exc).lower()
            if "already exists" not in msg:
                raise
            logger.warning(
                f"supersigrok_subscriptions already present during ensure: {exc}"
            )
        _supersigrok_tables_created = True


async def read_supersigrok_subscription(
    discord_user_id: int,
) -> Optional[SupersigrokSubscription]:
    await ensure_supersigrok_tables()
    async with get_session() as session:
        result = await session.execute(
            select(SupersigrokSubscription).where(
                SupersigrokSubscription.discord_user_id == int(discord_user_id)
            )
        )
        return result.scalar_one_or_none()


async def read_supersigrok_by_subscription_id(
    stripe_subscription_id: str,
) -> Optional[SupersigrokSubscription]:
    await ensure_supersigrok_tables()
    async with get_session() as session:
        result = await session.execute(
            select(SupersigrokSubscription).where(
                SupersigrokSubscription.stripe_subscription_id
                == stripe_subscription_id
            )
        )
        return result.scalar_one_or_none()


async def read_supersigrok_by_customer_id(
    stripe_customer_id: str,
) -> Optional[SupersigrokSubscription]:
    await ensure_supersigrok_tables()
    async with get_session() as session:
        result = await session.execute(
            select(SupersigrokSubscription).where(
                SupersigrokSubscription.stripe_customer_id == stripe_customer_id
            )
        )
        return result.scalar_one_or_none()


async def list_active_supersigrok_discord_ids() -> list[int]:
    await ensure_supersigrok_tables()
    async with get_session() as session:
        result = await session.execute(
            select(SupersigrokSubscription.discord_user_id).where(
                SupersigrokSubscription.status.in_(tuple(SUPERSIGROK_ACTIVE_STATUSES))
            )
        )
        return [int(row[0]) for row in result.all()]


async def upsert_supersigrok_subscription(
    *,
    discord_user_id: int,
    stripe_customer_id: Optional[str] = None,
    stripe_subscription_id: Optional[str] = None,
    status: str,
    current_period_end: Optional[datetime] = None,
    now: Optional[datetime] = None,
) -> SupersigrokSubscription:
    await ensure_supersigrok_tables()
    now_naive = _to_naive_utc(now or datetime.now(timezone.utc))
    period_end = _to_naive_utc(current_period_end)
    uid = int(discord_user_id)
    async with get_session() as session:
        result = await session.execute(
            select(SupersigrokSubscription).where(
                SupersigrokSubscription.discord_user_id == uid
            )
        )
        row = result.scalar_one_or_none()
        if row is None:
            row = SupersigrokSubscription(
                discord_user_id=uid,
                stripe_customer_id=stripe_customer_id,
                stripe_subscription_id=stripe_subscription_id,
                status=status,
                current_period_end=period_end,
                created_at=now_naive,
                updated_at=now_naive,
            )
            session.add(row)
        else:
            if stripe_customer_id is not None:
                row.stripe_customer_id = stripe_customer_id
            if stripe_subscription_id is not None:
                row.stripe_subscription_id = stripe_subscription_id
            row.status = status
            if current_period_end is not None or period_end is not None:
                row.current_period_end = period_end
            row.updated_at = now_naive
        await session.commit()
        await session.refresh(row)
        return row


async def ensure_subscriber_guild_tables() -> None:
    """Create subscriber_guilds if missing."""
    global _subscriber_guild_tables_created
    async with _subscriber_guild_tables_lock:
        if _subscriber_guild_tables_created:
            return
        try:
            async with engine.begin() as conn:

                def _create(sync_conn) -> None:
                    SubscriberGuild.__table__.create(sync_conn, checkfirst=True)

                await conn.run_sync(_create)
                await conn.execute(
                    text(
                        "CREATE UNIQUE INDEX IF NOT EXISTS "
                        "uq_subscriber_guild_sponsor "
                        "ON subscriber_guilds (sponsor_user_id)"
                    )
                )
        except Exception as exc:
            msg = str(exc).lower()
            if "already exists" not in msg:
                raise
            logger.warning(f"subscriber_guilds already present during ensure: {exc}")
        _subscriber_guild_tables_created = True


async def ensure_free_usage_tables() -> None:
    """Create free_user_usage if missing."""
    global _free_usage_tables_created
    async with _free_usage_tables_lock:
        if _free_usage_tables_created:
            return
        try:
            async with engine.begin() as conn:

                def _create(sync_conn) -> None:
                    FreeUserUsage.__table__.create(sync_conn, checkfirst=True)

                await conn.run_sync(_create)
        except Exception as exc:
            msg = str(exc).lower()
            if "already exists" not in msg:
                raise
            logger.warning(f"free_user_usage already present during ensure: {exc}")
        _free_usage_tables_created = True


async def read_subscriber_guild(guild_id: int) -> Optional[SubscriberGuild]:
    await ensure_subscriber_guild_tables()
    async with get_session() as session:
        result = await session.execute(
            select(SubscriberGuild).where(SubscriberGuild.guild_id == int(guild_id))
        )
        return result.scalar_one_or_none()


async def list_subscriber_guilds() -> list[SubscriberGuild]:
    await ensure_subscriber_guild_tables()
    async with get_session() as session:
        result = await session.execute(select(SubscriberGuild))
        return list(result.scalars().all())


async def list_subscriber_guilds_for_sponsor(sponsor_user_id: int) -> list[SubscriberGuild]:
    await ensure_subscriber_guild_tables()
    async with get_session() as session:
        result = await session.execute(
            select(SubscriberGuild).where(
                SubscriberGuild.sponsor_user_id == int(sponsor_user_id)
            )
        )
        return list(result.scalars().all())


async def read_subscriber_guild_for_sponsor(
    sponsor_user_id: int,
) -> Optional[SubscriberGuild]:
    """Return the single sponsored guild for this user, if any."""
    rows = await list_subscriber_guilds_for_sponsor(sponsor_user_id)
    return rows[0] if rows else None


async def upsert_subscriber_guild(
    *,
    guild_id: int,
    sponsor_user_id: int,
    now: Optional[datetime] = None,
    clear_grace: bool = True,
) -> SubscriberGuild:
    """Create/update a subscriber guild row without one-per-sponsor checks."""
    await ensure_subscriber_guild_tables()
    now_naive = _to_naive_utc(now or datetime.now(timezone.utc))
    gid = int(guild_id)
    sid = int(sponsor_user_id)
    async with get_session() as session:
        result = await session.execute(
            select(SubscriberGuild).where(SubscriberGuild.guild_id == gid)
        )
        row = result.scalar_one_or_none()
        if row is None:
            row = SubscriberGuild(
                guild_id=gid,
                sponsor_user_id=sid,
                added_at=now_naive,
                grace_until=None,
                grace_notified=False,
            )
            session.add(row)
        else:
            row.sponsor_user_id = sid
            if clear_grace:
                row.grace_until = None
                row.grace_notified = False
        await session.commit()
        await session.refresh(row)
        return row


async def assign_subscriber_guild(
    *,
    guild_id: int,
    sponsor_user_id: int,
    now: Optional[datetime] = None,
) -> SubscriberGuild:
    """Assign this guild to the sponsor; enforces one guild per SuperSigrok sub."""
    await ensure_subscriber_guild_tables()
    gid = int(guild_id)
    sid = int(sponsor_user_id)

    existing_for_sponsor = await read_subscriber_guild_for_sponsor(sid)
    if existing_for_sponsor is not None and int(existing_for_sponsor.guild_id) != gid:
        raise SponsorAlreadyHasGuildError(int(existing_for_sponsor.guild_id))

    existing_for_guild = await read_subscriber_guild(gid)
    if (
        existing_for_guild is not None
        and int(existing_for_guild.sponsor_user_id) != sid
    ):
        raise GuildAlreadySponsoredError(int(existing_for_guild.sponsor_user_id))

    return await upsert_subscriber_guild(
        guild_id=gid, sponsor_user_id=sid, now=now, clear_grace=True
    )


async def delete_subscriber_guild(guild_id: int) -> bool:
    await ensure_subscriber_guild_tables()
    async with get_session() as session:
        result = await session.execute(
            delete(SubscriberGuild).where(SubscriberGuild.guild_id == int(guild_id))
        )
        await session.commit()
        return bool(result.rowcount)


async def delete_subscriber_guild_for_sponsor(sponsor_user_id: int) -> Optional[int]:
    """Remove the sponsor's guild row. Returns guild_id if one was deleted."""
    row = await read_subscriber_guild_for_sponsor(int(sponsor_user_id))
    if row is None:
        return None
    await delete_subscriber_guild(int(row.guild_id))
    return int(row.guild_id)


async def start_subscriber_guild_grace_for_sponsor(
    sponsor_user_id: int,
    *,
    grace_days: int,
    now: Optional[datetime] = None,
) -> list[SubscriberGuild]:
    """Set grace_until on all guilds sponsored by this user (if not already in grace)."""
    await ensure_subscriber_guild_tables()
    now_dt = now or datetime.now(timezone.utc)
    now_naive = _to_naive_utc(now_dt)
    grace_until = _to_naive_utc(now_dt + timedelta(days=max(0, int(grace_days))))
    updated: list[SubscriberGuild] = []
    async with get_session() as session:
        result = await session.execute(
            select(SubscriberGuild).where(
                SubscriberGuild.sponsor_user_id == int(sponsor_user_id)
            )
        )
        rows = list(result.scalars().all())
        for row in rows:
            if row.grace_until is None:
                row.grace_until = grace_until
                row.grace_notified = False
                updated.append(row)
        await session.commit()
        for row in updated:
            await session.refresh(row)
        return updated


async def clear_subscriber_guild_grace_for_sponsor(
    sponsor_user_id: int,
) -> int:
    """Clear grace on all guilds for a resubscribed sponsor. Returns rows cleared."""
    await ensure_subscriber_guild_tables()
    async with get_session() as session:
        result = await session.execute(
            select(SubscriberGuild).where(
                SubscriberGuild.sponsor_user_id == int(sponsor_user_id)
            )
        )
        rows = list(result.scalars().all())
        cleared = 0
        for row in rows:
            if row.grace_until is not None or row.grace_notified:
                row.grace_until = None
                row.grace_notified = False
                cleared += 1
        await session.commit()
        return cleared


async def mark_subscriber_guild_grace_notified(guild_id: int) -> None:
    await ensure_subscriber_guild_tables()
    async with get_session() as session:
        result = await session.execute(
            select(SubscriberGuild).where(SubscriberGuild.guild_id == int(guild_id))
        )
        row = result.scalar_one_or_none()
        if row is None:
            return
        row.grace_notified = True
        await session.commit()


async def list_expired_grace_guilds(
    *, now: Optional[datetime] = None
) -> list[SubscriberGuild]:
    await ensure_subscriber_guild_tables()
    now_naive = _to_naive_utc(now or datetime.now(timezone.utc))
    async with get_session() as session:
        result = await session.execute(
            select(SubscriberGuild).where(
                SubscriberGuild.grace_until.is_not(None),
                SubscriberGuild.grace_until <= now_naive,
            )
        )
        return list(result.scalars().all())


async def list_grace_guilds_needing_notify() -> list[SubscriberGuild]:
    await ensure_subscriber_guild_tables()
    async with get_session() as session:
        result = await session.execute(
            select(SubscriberGuild).where(
                SubscriberGuild.grace_until.is_not(None),
                SubscriberGuild.grace_notified.is_(False),
            )
        )
        return list(result.scalars().all())


class FreeUsageDecision:
    """Result of checking / consuming a free-user reply slot."""

    __slots__ = ("allowed", "send_lockout_reply", "remaining_seconds", "silent")

    def __init__(
        self,
        *,
        allowed: bool,
        send_lockout_reply: bool,
        remaining_seconds: float,
        silent: bool,
    ) -> None:
        self.allowed = allowed
        self.send_lockout_reply = send_lockout_reply
        self.remaining_seconds = remaining_seconds
        self.silent = silent


async def check_and_consume_free_usage(
    user_id: int,
    *,
    replies_per_window: int,
    window_minutes: int,
    now: Optional[datetime] = None,
) -> FreeUsageDecision:
    """Atomically decide whether a free user may get a full reply.

    - Under quota → allowed=True (consumes one slot).
    - Over quota, first time → send_lockout_reply=True.
    - Over quota, already notified → silent=True.
    """
    await ensure_free_usage_tables()
    now_dt = now or datetime.now(timezone.utc)
    now_naive = _to_naive_utc(now_dt)
    assert now_naive is not None
    window = timedelta(minutes=max(1, int(window_minutes)))
    limit = max(0, int(replies_per_window))
    uid = int(user_id)

    async with get_session() as session:
        result = await session.execute(
            select(FreeUserUsage).where(FreeUserUsage.user_id == uid)
        )
        row = result.scalar_one_or_none()
        if row is None:
            row = FreeUserUsage(
                user_id=uid,
                window_started_at=now_naive,
                full_replies=0,
                lockout_reply_sent=False,
            )
            session.add(row)
            await session.flush()

        window_end = row.window_started_at + window
        if now_naive >= window_end:
            row.window_started_at = now_naive
            row.full_replies = 0
            row.lockout_reply_sent = False
            window_end = now_naive + window

        remaining = max(0.0, (window_end - now_naive).total_seconds())

        if row.full_replies < limit:
            row.full_replies += 1
            await session.commit()
            return FreeUsageDecision(
                allowed=True,
                send_lockout_reply=False,
                remaining_seconds=remaining,
                silent=False,
            )

        if not row.lockout_reply_sent:
            row.lockout_reply_sent = True
            await session.commit()
            return FreeUsageDecision(
                allowed=False,
                send_lockout_reply=True,
                remaining_seconds=remaining,
                silent=False,
            )

        await session.commit()
        return FreeUsageDecision(
            allowed=False,
            send_lockout_reply=False,
            remaining_seconds=remaining,
            silent=True,
        )


async def read_relationship(
    guild_id: int, user_id: int
) -> Optional[UserRelationship]:
    await ensure_relationship_tables()
    async with get_session() as session:
        result = await session.execute(
            select(UserRelationship).where(
                UserRelationship.guild_id == guild_id,
                UserRelationship.user_id == user_id,
            )
        )
        return result.scalar_one_or_none()


async def read_relationships_by_ids(
    guild_id: int, user_ids: list[int]
) -> list[UserRelationship]:
    if not user_ids:
        return []
    await ensure_relationship_tables()
    unique_ids = sorted(set(user_ids))
    async with get_session() as session:
        result = await session.execute(
            select(UserRelationship).where(
                UserRelationship.guild_id == guild_id,
                UserRelationship.user_id.in_(unique_ids),
            )
        )
        return list(result.scalars().all())


async def apply_relationship_updates(
    guild_id: int,
    updates: list[dict[str, Any]],
    *,
    now: Optional[datetime] = None,
) -> list[UserRelationship]:
    """Apply already-validated, bounded relationship deltas in one transaction."""
    if not updates:
        return []
    await ensure_relationship_tables()
    now_naive = _to_naive_utc(now or datetime.now(timezone.utc))
    user_ids = sorted(
        {
            int(item["user_id"])
            for item in updates
            if isinstance(item.get("user_id"), int)
        }
    )
    if not user_ids:
        return []

    async with get_session() as session:
        result = await session.execute(
            select(UserRelationship).where(
                UserRelationship.guild_id == guild_id,
                UserRelationship.user_id.in_(user_ids),
            )
        )
        rows = {row.user_id: row for row in result.scalars().all()}
        changed_rows: list[UserRelationship] = []

        for item in updates:
            user_id = item.get("user_id")
            if not isinstance(user_id, int) or user_id not in user_ids:
                continue
            row = rows.get(user_id)
            if row is None:
                row = UserRelationship(
                    guild_id=guild_id,
                    user_id=user_id,
                    affinity=100,
                    trust=50,
                    disposition="neutral",
                    roast_level=1,
                    engagement_weight=100,
                )
                rows[user_id] = row
                session.add(row)

            changed = False
            delta_specs = (
                ("affinity", "affinity_delta", 0, 200, 10),
                ("trust", "trust_delta", 0, 100, 5),
                ("roast_level", "roast_level_delta", 0, 3, 1),
                (
                    "engagement_weight",
                    "engagement_weight_delta",
                    0,
                    100,
                    10,
                ),
            )
            for attr, key, lower, upper, max_delta in delta_specs:
                raw_delta = item.get(key)
                if not isinstance(raw_delta, int):
                    continue
                delta = max(-max_delta, min(max_delta, raw_delta))
                old_value = int(getattr(row, attr))
                new_value = max(lower, min(upper, old_value + delta))
                if new_value != old_value:
                    setattr(row, attr, new_value)
                    changed = True

            disposition = item.get("disposition")
            if disposition in {"ally", "neutral", "rival", "ignore"}:
                if row.disposition != disposition:
                    row.disposition = disposition
                    changed = True

            last_vibe = item.get("last_vibe")
            if last_vibe in {
                "chill",
                "friendly",
                "spicy",
                "hostile",
                "correction",
                "dismissive",
            }:
                if row.last_vibe != last_vibe:
                    row.last_vibe = last_vibe
                    changed = True

            if changed:
                row.updated_at = now_naive
                changed_rows.append(row)

        if changed_rows:
            await session.commit()
        else:
            await session.rollback()
        return changed_rows


async def set_relationship_field(
    guild_id: int,
    user_id: int,
    field: str,
    value: int | str | None,
    *,
    now: Optional[datetime] = None,
) -> UserRelationship:
    """Set one numeric/enum field (no free-text / personal-info fields)."""
    await ensure_relationship_tables()
    now_naive = _to_naive_utc(now or datetime.now(timezone.utc))
    async with get_session() as session:
        result = await session.execute(
            select(UserRelationship).where(
                UserRelationship.guild_id == guild_id,
                UserRelationship.user_id == user_id,
            )
        )
        row = result.scalar_one_or_none()
        if row is None:
            row = UserRelationship(
                guild_id=guild_id,
                user_id=user_id,
                affinity=100,
                trust=50,
                disposition="neutral",
                roast_level=1,
                engagement_weight=100,
            )
            session.add(row)

        if field == "affinity":
            row.affinity = max(0, min(200, int(value)))
        elif field == "trust":
            row.trust = max(0, min(100, int(value)))
        elif field == "roast_level":
            row.roast_level = max(0, min(3, int(value)))
        elif field == "engagement_weight":
            row.engagement_weight = max(0, min(100, int(value)))
        elif field == "disposition" and value in {
            "ally",
            "neutral",
            "rival",
            "ignore",
        }:
            row.disposition = str(value)
        elif field == "last_vibe" and (
            value is None
            or value
            in {
                "chill",
                "friendly",
                "spicy",
                "hostile",
                "correction",
                "dismissive",
            }
        ):
            row.last_vibe = str(value) if value is not None else None
        else:
            raise ValueError(f"Unsupported relationship field/value: {field}={value!r}")

        row.updated_at = now_naive
        await session.commit()
        return row


async def reset_relationship(guild_id: int, user_id: int) -> bool:
    await ensure_relationship_tables()
    async with get_session() as session:
        result = await session.execute(
            delete(UserRelationship).where(
                UserRelationship.guild_id == guild_id,
                UserRelationship.user_id == user_id,
            )
        )
        await session.commit()
        return bool(result.rowcount)


async def try_claim_relationship_run(
    guild_id: int,
    channel_id: int,
    *,
    now: Optional[datetime] = None,
    cooldown: timedelta = timedelta(hours=24),
    lease_duration: timedelta = timedelta(minutes=30),
) -> Optional[str]:
    """Atomically claim this channel's reflection slot.

    Cooldown is based on attempts, not only successes, so a failing provider
    cannot trigger an expensive request on every subsequent ping in the channel.
    Other channels in the same guild remain independent.
    """
    await ensure_relationship_tables()
    now_naive = _to_naive_utc(now or datetime.now(timezone.utc))
    assert now_naive is not None
    eligible_before = now_naive - cooldown
    lease_until = now_naive + lease_duration
    token = secrets.token_hex(16)

    async with get_session() as session:
        await session.execute(
            insert(ChannelRelationshipRun)
            .values(channel_id=channel_id, guild_id=guild_id)
            .prefix_with("OR IGNORE")
        )
        result = await session.execute(
            update(ChannelRelationshipRun)
            .where(
                ChannelRelationshipRun.channel_id == channel_id,
                or_(
                    ChannelRelationshipRun.last_attempt_at.is_(None),
                    ChannelRelationshipRun.last_attempt_at <= eligible_before,
                ),
                or_(
                    ChannelRelationshipRun.lease_until.is_(None),
                    ChannelRelationshipRun.lease_until <= now_naive,
                ),
            )
            .values(
                guild_id=guild_id,
                last_attempt_at=now_naive,
                lease_until=lease_until,
                lease_token=token,
            )
        )
        await session.commit()
        return token if result.rowcount == 1 else None


async def complete_relationship_run(
    channel_id: int,
    lease_token: str,
    *,
    now: Optional[datetime] = None,
) -> bool:
    await ensure_relationship_tables()
    now_naive = _to_naive_utc(now or datetime.now(timezone.utc))
    async with get_session() as session:
        result = await session.execute(
            update(ChannelRelationshipRun)
            .where(
                ChannelRelationshipRun.channel_id == channel_id,
                ChannelRelationshipRun.lease_token == lease_token,
            )
            .values(
                last_success_at=now_naive,
                lease_until=None,
                lease_token=None,
            )
        )
        await session.commit()
        return result.rowcount == 1


async def release_relationship_run(channel_id: int, lease_token: str) -> bool:
    """Release a failed lease while retaining the 24-hour attempt cooldown."""
    await ensure_relationship_tables()
    async with get_session() as session:
        result = await session.execute(
            update(ChannelRelationshipRun)
            .where(
                ChannelRelationshipRun.channel_id == channel_id,
                ChannelRelationshipRun.lease_token == lease_token,
            )
            .values(lease_until=None, lease_token=None)
        )
        await session.commit()
        return result.rowcount == 1


async def read_relationship_run(
    channel_id: int,
) -> Optional[ChannelRelationshipRun]:
    await ensure_relationship_tables()
    async with get_session() as session:
        result = await session.execute(
            select(ChannelRelationshipRun).where(
                ChannelRelationshipRun.channel_id == channel_id
            )
        )
        return result.scalar_one_or_none()

async def insert_schedule_mention_job(
    *,
    job_id: str,
    kind: str,
    guild_id: int,
    channel_id: int,
    message_id: int,
    creator_id: int,
    prompt: str,
    due_at: Optional[datetime],
    cron_expr: Optional[str],
    next_fire: Optional[datetime],
    ack_message_id: Optional[int],
) -> None:
    await ensure_schedule_tables()
    row = ScheduleMentionJob(
        job_id=job_id,
        kind=kind,
        guild_id=guild_id,
        channel_id=channel_id,
        message_id=message_id,
        creator_id=creator_id,
        prompt=prompt,
        due_at=_to_naive_utc(due_at),
        cron_expr=cron_expr,
        next_fire=_to_naive_utc(next_fire),
        ack_message_id=ack_message_id,
    )
    async with get_session() as session:
        session.add(row)
        await session.commit()


async def delete_schedule_mention_job(job_id: str) -> None:
    await ensure_schedule_tables()
    async with get_session() as session:
        result = await session.execute(
            select(ScheduleMentionJob).where(ScheduleMentionJob.job_id == job_id)
        )
        row = result.scalar_one_or_none()
        if row is not None:
            await session.delete(row)
            await session.commit()


async def list_schedule_mention_jobs() -> list[ScheduleMentionJob]:
    await ensure_schedule_tables()
    async with get_session() as session:
        result = await session.execute(select(ScheduleMentionJob))
        return list(result.scalars().all())


async def read_due_schedule_jobs(now: datetime) -> list[ScheduleMentionJob]:
    """Return one-shot and cron jobs that are due at or before `now`."""
    await ensure_schedule_tables()
    now_naive = _to_naive_utc(now)
    async with get_session() as session:
        result = await session.execute(
            select(ScheduleMentionJob).where(
                or_(
                    and_(
                        ScheduleMentionJob.kind == "once",
                        ScheduleMentionJob.due_at.is_not(None),
                        ScheduleMentionJob.due_at <= now_naive,
                    ),
                    and_(
                        ScheduleMentionJob.kind == "cron",
                        ScheduleMentionJob.next_fire.is_not(None),
                        ScheduleMentionJob.next_fire <= now_naive,
                    ),
                )
            )
        )
        return list(result.scalars().all())


async def update_schedule_cron_next_fire(job_id: str, next_fire: datetime) -> None:
    await ensure_schedule_tables()
    async with get_session() as session:
        result = await session.execute(
            select(ScheduleMentionJob).where(ScheduleMentionJob.job_id == job_id)
        )
        row = result.scalar_one_or_none()
        if row is not None:
            row.next_fire = _to_naive_utc(next_fire)
            await session.commit()


async def bump_once_schedule_job_due(job_id: str, delay: timedelta) -> None:
    """Push a one-shot job forward after a failed delivery attempt (avoids tight retry loops)."""
    await ensure_schedule_tables()
    new_due = datetime.now(timezone.utc) + delay
    async with get_session() as session:
        result = await session.execute(
            select(ScheduleMentionJob).where(
                ScheduleMentionJob.job_id == job_id,
                ScheduleMentionJob.kind == "once",
            )
        )
        row = result.scalar_one_or_none()
        if row is not None:
            row.due_at = _to_naive_utc(new_due)
            await session.commit()


def db_logger(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            elapsed_time = time.time() - start_time
            logger.info(f"{func.__name__} executed in {elapsed_time:.2f} seconds")
            return result
        except Exception:
            logger.exception(f"Error in {func.__name__}")
            raise

    return wrapper


@asynccontextmanager
async def get_session():
    async with async_session() as session:
        # async with session.begin():
        yield session


@db_logger
async def add_user(user: User) -> None:
    async with get_session() as session:
        session.add(user)
        await session.commit()


@db_logger
async def read_user(guild_id: int, user_id: int) -> Optional[User]:
    async with get_session() as session:
        result = await session.execute(
            select(User).where(User.guild_id == guild_id, User.user_id == user_id)
        )
        return result.scalar_one_or_none()


@db_logger
async def read_or_add_user(guild_id: int, user_id: int) -> User:
    await ensure_user_schema()
    user = await read_user(guild_id, user_id)
    if user is not None:
        logger.info(f"Existing user found: {user}")
        return user
    user = User(guild_id=guild_id, user_id=user_id, rating=100)
    try:
        await add_user(user)
        logger.info(f"New user created: {user}")
        return user
    except IntegrityError:
        existing = await read_user(guild_id, user_id)
        if existing is None:
            raise
        logger.info(f"Existing user found after conflict: {existing}")
        return existing


@db_logger
async def read_or_add_users(guild_id: int, user_ids: list[int]) -> list[User]:
    await ensure_user_schema()
    if not user_ids:
        return []
    async with get_session() as session:
        stmt = select(User).where(User.guild_id == guild_id, User.user_id.in_(user_ids))
        result = await session.execute(stmt)
        existing_users = {user.user_id: user for user in result.scalars().all()}

        missing_user_ids = set(user_ids) - existing_users.keys()
        new_users = [
            User(guild_id=guild_id, user_id=user_id, rating=100)
            for user_id in missing_user_ids
        ]

        if new_users:
            session.add_all(new_users)
            try:
                await session.commit()
            except IntegrityError:
                await session.rollback()
                result = await session.execute(stmt)
                existing_users = {user.user_id: user for user in result.scalars().all()}
                return [existing_users[uid] for uid in user_ids if uid in existing_users]
        return [existing_users[uid] for uid in user_ids if uid in existing_users] + new_users


@db_logger
async def read_users_by_ids(guild_id: int, user_ids: list[int]) -> list[User]:
    if not user_ids:
        return []
    await ensure_user_schema()
    async with get_session() as session:
        stmt = select(User).where(User.guild_id == guild_id, User.user_id.in_(user_ids))
        result = await session.execute(stmt)
        return list(result.scalars().all())


@db_logger
async def upsert_user_rating(guild_id: int, user_id: int, rating: int) -> User:
    async with get_session() as session:
        stmt = select(User).where(User.user_id == user_id, User.guild_id == guild_id)
        result = await session.execute(stmt)
        user = result.scalar_one_or_none()

        if user:
            user.rating = rating
        else:
            user = User(user_id=user_id, guild_id=guild_id, rating=rating)
            session.add(user)

        await session.commit()
    return user


@db_logger
async def adjust_user_rating(guild_id: int, user_id: int, delta: int) -> User:
    async with get_session() as session:
        stmt = select(User).where(User.user_id == user_id, User.guild_id == guild_id)
        result = await session.execute(stmt)
        user = result.scalar_one_or_none()

        if user is None:
            user = User(user_id=user_id, guild_id=guild_id, rating=max(100 + delta, 0))
            session.add(user)
        else:
            base = user.rating if user.rating is not None else 100
            user.rating = max(base + delta, 0)

        await session.commit()
    return user


async def read_top_ratings(guild_id: int) -> AsyncIterator[User]:
    async with get_session() as session:
        stmt = select(User).where(User.guild_id == guild_id, User.is_present)
        stmt = stmt.order_by(User.rating.desc(), User.user_id.asc())
        result = await session.stream(stmt)
        async for user in result.scalars():
            yield user


async def read_bottom_ratings(guild_id: int) -> AsyncIterator[User]:
    async with get_session() as session:
        stmt = select(User).where(User.guild_id == guild_id, User.is_present)
        stmt = stmt.order_by(User.rating.asc(), User.user_id.asc())
        result = await session.stream(stmt)
        async for user in result.scalars():
            yield user


@db_logger
async def read_present_users(guild_id: int) -> list[User]:
    async with get_session() as session:
        stmt = select(User).where(User.guild_id == guild_id, User.is_present)
        stmt = stmt.order_by(User.rating.desc(), User.user_id.asc())
        result = await session.execute(stmt)
        return list(result.scalars().all())


@db_logger
async def remove_user(guild_id: int, user_id: int) -> None:
    async with get_session() as session:
        stmt = select(User).where(User.user_id == user_id, User.guild_id == guild_id)
        result = await session.execute(stmt)
        user = result.scalar_one_or_none()

        if user:
            await session.delete(user)
            await session.commit()
            logger.info(f"User {user} removed from the database")
        else:
            logger.warning(f"User {user_id} not found in the database")


async def async_main():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


if __name__ == "__main__":
    asyncio.run(async_main())
