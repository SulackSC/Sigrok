"""Tests for SuperSigrok invite links, subscriber guilds, grace, and free rate limits."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from sigrok import db, supersigrok
from sigrok.billing import stripe_billing
from sigrok.config import settings


async def _install_temp_db(tmp_path, monkeypatch):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'commercial.db'}")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(db, "engine", engine)
    monkeypatch.setattr(db, "async_session", sessions)
    monkeypatch.setattr(db, "_supersigrok_tables_created", False)
    monkeypatch.setattr(db, "_subscriber_guild_tables_created", False)
    monkeypatch.setattr(db, "_free_usage_tables_created", False)
    async with engine.begin() as conn:
        await conn.run_sync(db.Base.metadata.create_all)
    return engine


def test_invite_url_shape() -> None:
    url = supersigrok.invite_url(123456789012345678, permissions=3263680)
    assert url.startswith("https://discord.com/oauth2/authorize?")
    assert "client_id=123456789012345678" in url
    assert "permissions=3263680" in url
    assert "scope=bot" in url


def test_humanize_duration() -> None:
    assert supersigrok.humanize_duration(30) == "a minute"
    assert supersigrok.humanize_duration(60) == "1 minute"
    assert "minute" in supersigrok.humanize_duration(2400)


def test_everyone_until_does_not_authorize_guild_sponsor() -> None:
    supersigrok.clear_runtime_grants()
    saved_users = list(settings.bot.supersigrok.user_ids)
    saved_roles = list(settings.bot.supersigrok.role_ids)
    saved_everyone = settings.bot.supersigrok.everyone_until
    try:
        settings.bot.supersigrok.user_ids = []
        settings.bot.supersigrok.role_ids = []
        settings.bot.supersigrok.everyone_until = datetime.now(timezone.utc) + timedelta(
            hours=1
        )
        assert supersigrok.is_supersigrok(user_id=999) is True
        assert supersigrok.can_sponsor_guild(user_id=999) is False
    finally:
        settings.bot.supersigrok.user_ids = saved_users
        settings.bot.supersigrok.role_ids = saved_roles
        settings.bot.supersigrok.everyone_until = saved_everyone
        supersigrok.clear_runtime_grants()


@pytest.mark.asyncio
async def test_free_usage_quota_and_lockout(tmp_path, monkeypatch) -> None:
    await _install_temp_db(tmp_path, monkeypatch)
    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    for _ in range(3):
        d = await db.check_and_consume_free_usage(
            7, replies_per_window=3, window_minutes=60, now=now
        )
        assert d.allowed is True

    d = await db.check_and_consume_free_usage(
        7, replies_per_window=3, window_minutes=60, now=now
    )
    assert d.allowed is False
    assert d.send_lockout_reply is True
    assert d.silent is False

    d2 = await db.check_and_consume_free_usage(
        7, replies_per_window=3, window_minutes=60, now=now
    )
    assert d2.allowed is False
    assert d2.send_lockout_reply is False
    assert d2.silent is True

    later = now + timedelta(hours=2)
    d3 = await db.check_and_consume_free_usage(
        7, replies_per_window=3, window_minutes=60, now=later
    )
    assert d3.allowed is True


@pytest.mark.asyncio
async def test_misc_rate_limit_cheap_then_silent(tmp_path, monkeypatch) -> None:
    await _install_temp_db(tmp_path, monkeypatch)
    from sigrok.cogs.misc import Misc

    bot = MagicMock()
    bot.user = SimpleNamespace(id=1, name="Sigrok")
    cog = Misc(bot)

    saved_users = list(settings.bot.supersigrok.user_ids)
    saved_everyone = settings.bot.supersigrok.everyone_until
    saved_limit = settings.bot.supersigrok.free_replies_per_window
    supersigrok.clear_runtime_grants()
    try:
        settings.bot.supersigrok.user_ids = []
        settings.bot.supersigrok.everyone_until = None
        settings.bot.supersigrok.free_replies_per_window = 1

        author = SimpleNamespace(id=55, bot=False, roles=[], name="freebie")
        channel = AsyncMock()

        class _TypingCM:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return None

        channel.typing = MagicMock(return_value=_TypingCM())
        message = SimpleNamespace(
            author=author,
            guild=SimpleNamespace(id=999),
            channel=channel,
            content="<@1> hello",
            clean_content="@Sigrok hello",
            mentions=[bot.user],
            reference=None,
            id=100,
            to_reference=MagicMock(return_value=None),
        )

        monkeypatch.setattr(cog, "_strip_bot_mention", lambda content: "hello")
        monkeypatch.setattr(cog, "_strip_bot_mention_raw", lambda content: "hello")
        from sigrok import genai as genai_mod

        monkeypatch.setattr(genai_mod.client, "_message_has_images", lambda _m: False)

        full = AsyncMock(return_value="ok full reply that is definitely fine")
        monkeypatch.setattr(genai_mod.client, "answer_message_question", full)
        monkeypatch.setattr(cog, "_maybe_schedule_relationship_reflection", AsyncMock())
        monkeypatch.setattr(
            "sigrok.channel_rules.fetch_channel_rules", AsyncMock(return_value="")
        )
        monkeypatch.setattr(cog, "_send_response_to_ping", AsyncMock())
        monkeypatch.setattr(cog, "_react_to_failed_llm_response", AsyncMock())

        await cog._handle_bot_mention(message)
        full.assert_awaited_once()

        cheap = AsyncMock(return_value="can't put more effort into you for the next hour")
        monkeypatch.setattr(genai_mod.client, "answer_rate_limit_cooldown", cheap)
        send = AsyncMock()
        monkeypatch.setattr(cog, "_send_response_to_ping", send)
        await cog._handle_bot_mention(message)
        cheap.assert_awaited_once()
        send.assert_awaited_once()

        cheap.reset_mock()
        send.reset_mock()
        await cog._handle_bot_mention(message)
        cheap.assert_not_awaited()
        send.assert_not_awaited()

        supersigrok.grant_user(55)
        full.reset_mock()
        await cog._handle_bot_mention(message)
        full.assert_awaited_once()
    finally:
        settings.bot.supersigrok.user_ids = saved_users
        settings.bot.supersigrok.everyone_until = saved_everyone
        settings.bot.supersigrok.free_replies_per_window = saved_limit
        supersigrok.clear_runtime_grants()


@pytest.mark.asyncio
async def test_subscriber_guild_auth_and_grace(tmp_path, monkeypatch) -> None:
    await _install_temp_db(tmp_path, monkeypatch)
    from sigrok.config import WhitelistEntry

    supersigrok.clear_runtime_grants()
    saved_users = list(settings.bot.supersigrok.user_ids)
    saved_everyone = settings.bot.supersigrok.everyone_until
    saved_whitelist = list(settings.bot.whitelist)
    try:
        settings.bot.supersigrok.user_ids = []
        settings.bot.supersigrok.everyone_until = None
        settings.bot.whitelist = [WhitelistEntry(guild=111, channel=0, roles=[])]

        assert await supersigrok.guild_is_authorized(111) is True
        assert await supersigrok.guild_is_authorized(222) is False

        supersigrok.grant_user(42)
        await db.upsert_subscriber_guild(guild_id=222, sponsor_user_id=42)
        assert await supersigrok.guild_is_authorized(222) is True

        await stripe_billing.apply_subscription_entitlement(
            discord_user_id=42,
            stripe_customer_id="cus_x",
            stripe_subscription_id="sub_x",
            status="canceled",
        )
        assert supersigrok.can_sponsor_guild(user_id=42) is False
        row = await db.read_subscriber_guild(222)
        assert row is not None
        assert row.grace_until is not None
        assert await supersigrok.guild_is_authorized(222) is True

        expired = await db.list_expired_grace_guilds(
            now=datetime.now(timezone.utc) + timedelta(days=30)
        )
        assert any(g.guild_id == 222 for g in expired)

        # Force expired in DB and confirm unauthorized
        async with db.get_session() as session:
            from sqlalchemy.future import select

            result = await session.execute(
                select(db.SubscriberGuild).where(db.SubscriberGuild.guild_id == 222)
            )
            g = result.scalar_one()
            g.grace_until = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(
                days=1
            )
            await session.commit()
        assert await supersigrok.guild_is_authorized(222) is False

        supersigrok.grant_user(42)
        await db.upsert_subscriber_guild(guild_id=222, sponsor_user_id=42)
        await db.start_subscriber_guild_grace_for_sponsor(42, grace_days=3)
        await stripe_billing.apply_subscription_entitlement(
            discord_user_id=42,
            stripe_customer_id="cus_x",
            stripe_subscription_id="sub_x",
            status="active",
            current_period_end=datetime.now(timezone.utc) + timedelta(days=30),
        )
        row = await db.read_subscriber_guild(222)
        assert row is not None
        assert row.grace_until is None
        assert await supersigrok.guild_is_authorized(222) is True
    finally:
        settings.bot.supersigrok.user_ids = saved_users
        settings.bot.supersigrok.everyone_until = saved_everyone
        settings.bot.whitelist = saved_whitelist
        supersigrok.clear_runtime_grants()


@pytest.mark.asyncio
async def test_one_server_per_sponsor(tmp_path, monkeypatch) -> None:
    await _install_temp_db(tmp_path, monkeypatch)
    await db.assign_subscriber_guild(guild_id=100, sponsor_user_id=7)
    # Same guild re-assign is fine
    again = await db.assign_subscriber_guild(guild_id=100, sponsor_user_id=7)
    assert again.guild_id == 100

    with pytest.raises(db.SponsorAlreadyHasGuildError) as exc:
        await db.assign_subscriber_guild(guild_id=200, sponsor_user_id=7)
    assert exc.value.existing_guild_id == 100

    await db.assign_subscriber_guild(guild_id=200, sponsor_user_id=8)
    with pytest.raises(db.GuildAlreadySponsoredError):
        await db.assign_subscriber_guild(guild_id=200, sponsor_user_id=9)

    deleted = await db.delete_subscriber_guild_for_sponsor(7)
    assert deleted == 100
    assert await db.read_subscriber_guild_for_sponsor(7) is None
    # Slot freed — can take a new server
    await db.assign_subscriber_guild(guild_id=300, sponsor_user_id=7)


@pytest.mark.asyncio
async def test_dm_leave_command(tmp_path, monkeypatch) -> None:
    await _install_temp_db(tmp_path, monkeypatch)
    from sigrok.cogs.supersigrok_billing import SuperSigrokCog

    supersigrok.clear_runtime_grants()
    saved_users = list(settings.bot.supersigrok.user_ids)
    saved_everyone = settings.bot.supersigrok.everyone_until
    try:
        settings.bot.supersigrok.user_ids = []
        settings.bot.supersigrok.everyone_until = None
        supersigrok.grant_user(77)
        await db.assign_subscriber_guild(guild_id=555, sponsor_user_id=77)

        guild = MagicMock()
        guild.id = 555
        guild.name = "Test Guild"
        guild.leave = AsyncMock()

        bot = MagicMock()
        bot.user = SimpleNamespace(id=1, name="Sigrok")
        bot.get_guild = MagicMock(return_value=guild)
        bot.loop = MagicMock()

        cog = SuperSigrokCog(bot)
        author = SimpleNamespace(id=77, send=AsyncMock())
        ctx = SimpleNamespace(
            author=author,
            guild=None,
            send=AsyncMock(),
        )
        await SuperSigrokCog.supersigrok_leave.callback(cog, ctx)
        guild.leave.assert_awaited_once()
        assert await db.read_subscriber_guild_for_sponsor(77) is None
        author.send.assert_awaited()
        sent = author.send.await_args.args[0].lower()
        assert "left" in sent or "cleared" in sent
    finally:
        settings.bot.supersigrok.user_ids = saved_users
        settings.bot.supersigrok.everyone_until = saved_everyone
        supersigrok.clear_runtime_grants()
