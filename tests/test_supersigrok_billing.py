"""Tests for SuperSigrok Stripe entitlements, DMs, NSFW bypass, and badge."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from sigrok import db, supersigrok
from sigrok.billing import stripe_billing
from sigrok.config import settings
from sigrok.streaming.response import NSFW_FILTER_REPLACEMENT, apply_nsfw_filter


async def _install_temp_db(tmp_path, monkeypatch):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'supersigrok.db'}")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(db, "engine", engine)
    monkeypatch.setattr(db, "async_session", sessions)
    monkeypatch.setattr(db, "_supersigrok_tables_created", False)
    async with engine.begin() as conn:
        await conn.run_sync(db.Base.metadata.create_all)
    return engine


@pytest.mark.asyncio
async def test_subscription_upsert_and_hydrate(tmp_path, monkeypatch) -> None:
    await _install_temp_db(tmp_path, monkeypatch)
    supersigrok.clear_runtime_grants()
    saved_users = list(settings.bot.supersigrok.user_ids)
    saved_everyone = settings.bot.supersigrok.everyone_until
    try:
        settings.bot.supersigrok.user_ids = []
        settings.bot.supersigrok.everyone_until = None

        row = await db.upsert_supersigrok_subscription(
            discord_user_id=42,
            stripe_customer_id="cus_test",
            stripe_subscription_id="sub_test",
            status="active",
            current_period_end=datetime.now(timezone.utc) + timedelta(days=30),
        )
        assert row.is_active is True
        assert supersigrok.is_supersigrok(user_id=42) is False  # not hydrated yet

        n = await supersigrok.hydrate_subscription_grants()
        assert n == 1
        assert supersigrok.is_supersigrok(user_id=42) is True

        await db.upsert_supersigrok_subscription(
            discord_user_id=42,
            status="canceled",
        )
        # Hydrate does not revoke; apply_subscription_entitlement does.
        await stripe_billing.apply_subscription_entitlement(
            discord_user_id=42,
            stripe_customer_id="cus_test",
            stripe_subscription_id="sub_test",
            status="canceled",
        )
        assert supersigrok.is_supersigrok(user_id=42) is False
    finally:
        settings.bot.supersigrok.user_ids = saved_users
        settings.bot.supersigrok.everyone_until = saved_everyone
        supersigrok.clear_runtime_grants()


@pytest.mark.asyncio
async def test_apply_checkout_completed_grants(tmp_path, monkeypatch) -> None:
    await _install_temp_db(tmp_path, monkeypatch)
    supersigrok.clear_runtime_grants()
    saved_users = list(settings.bot.supersigrok.user_ids)
    saved_everyone = settings.bot.supersigrok.everyone_until
    try:
        settings.bot.supersigrok.user_ids = []
        settings.bot.supersigrok.everyone_until = None

        event = {
            "type": "checkout.session.completed",
            "data": {
                "object": {
                    "client_reference_id": "99",
                    "customer": "cus_99",
                    "subscription": None,
                    "metadata": {"discord_user_id": "99"},
                }
            },
        }
        await stripe_billing.handle_stripe_event(event)
        assert supersigrok.is_supersigrok(user_id=99) is True
        row = await db.read_supersigrok_subscription(99)
        assert row is not None
        assert row.stripe_customer_id == "cus_99"
        assert row.status == "active"
    finally:
        settings.bot.supersigrok.user_ids = saved_users
        settings.bot.supersigrok.everyone_until = saved_everyone
        supersigrok.clear_runtime_grants()


@pytest.mark.asyncio
async def test_subscription_updated_revokes_past_due(tmp_path, monkeypatch) -> None:
    await _install_temp_db(tmp_path, monkeypatch)
    supersigrok.clear_runtime_grants()
    saved_users = list(settings.bot.supersigrok.user_ids)
    saved_everyone = settings.bot.supersigrok.everyone_until
    try:
        settings.bot.supersigrok.user_ids = []
        settings.bot.supersigrok.everyone_until = None
        await stripe_billing.apply_subscription_entitlement(
            discord_user_id=7,
            stripe_customer_id="cus_7",
            stripe_subscription_id="sub_7",
            status="active",
        )
        assert supersigrok.is_supersigrok(user_id=7) is True

        event = {
            "type": "customer.subscription.updated",
            "data": {
                "object": {
                    "id": "sub_7",
                    "customer": "cus_7",
                    "status": "past_due",
                    "metadata": {"discord_user_id": "7"},
                    "current_period_end": int(
                        (datetime.now(timezone.utc) + timedelta(days=1)).timestamp()
                    ),
                }
            },
        }
        await stripe_billing.handle_stripe_event(event)
        assert supersigrok.is_supersigrok(user_id=7) is False
        row = await db.read_supersigrok_subscription(7)
        assert row is not None
        assert row.status == "past_due"
    finally:
        settings.bot.supersigrok.user_ids = saved_users
        settings.bot.supersigrok.everyone_until = saved_everyone
        supersigrok.clear_runtime_grants()


@pytest.mark.asyncio
async def test_checkout_session_includes_discord_metadata(monkeypatch) -> None:
    captured: dict = {}

    class FakeSessions:
        def create(self, params):
            captured.update(params)
            return SimpleNamespace(url="https://checkout.stripe.test/session")

    class FakeCheckout:
        sessions = FakeSessions()

    class FakeV1:
        checkout = FakeCheckout()

    class FakeClient:
        v1 = FakeV1()

    monkeypatch.setattr(stripe_billing, "_stripe_client", lambda: FakeClient())
    monkeypatch.setattr(
        settings.tokens, "stripe_secret_key", "sk_test_x", raising=False
    )
    monkeypatch.setattr(settings.bot.supersigrok, "stripe_price_id", "")
    monkeypatch.setattr(settings.bot.supersigrok, "stripe_amount_cents", 500)
    monkeypatch.setattr(settings.bot.supersigrok, "stripe_currency", "usd")
    monkeypatch.setattr(settings.bot.supersigrok, "stripe_product_name", "SuperSigrok")
    monkeypatch.setattr(
        settings.bot.supersigrok,
        "stripe_success_url",
        "https://example.com/ok",
    )
    monkeypatch.setattr(
        settings.bot.supersigrok,
        "stripe_cancel_url",
        "https://example.com/cancel",
    )

    async def fake_read(_uid):
        return None

    monkeypatch.setattr(db, "read_supersigrok_subscription", fake_read)

    url = await stripe_billing.create_checkout_session(123456789012345678)
    assert url.startswith("https://checkout.stripe.test/")
    assert captured["client_reference_id"] == "123456789012345678"
    assert captured["metadata"]["discord_user_id"] == "123456789012345678"
    assert captured["subscription_data"]["metadata"]["discord_user_id"] == (
        "123456789012345678"
    )
    item = captured["line_items"][0]
    assert "price" not in item
    assert item["price_data"]["unit_amount"] == 500
    assert item["price_data"]["currency"] == "usd"
    assert item["price_data"]["recurring"]["interval"] == "month"
    assert item["price_data"]["product_data"]["name"] == "SuperSigrok"


@pytest.mark.asyncio
async def test_checkout_uses_price_id_when_set(monkeypatch) -> None:
    captured: dict = {}

    class FakeSessions:
        def create(self, params):
            captured.update(params)
            return SimpleNamespace(url="https://checkout.stripe.test/session")

    class FakeCheckout:
        sessions = FakeSessions()

    class FakeV1:
        checkout = FakeCheckout()

    class FakeClient:
        v1 = FakeV1()

    monkeypatch.setattr(stripe_billing, "_stripe_client", lambda: FakeClient())
    monkeypatch.setattr(
        settings.tokens, "stripe_secret_key", "sk_test_x", raising=False
    )
    monkeypatch.setattr(settings.bot.supersigrok, "stripe_price_id", "price_test")

    async def fake_read(_uid):
        return None

    monkeypatch.setattr(db, "read_supersigrok_subscription", fake_read)

    await stripe_billing.create_checkout_session(1)
    assert captured["line_items"][0]["price"] == "price_test"


def test_badge_reply_prefixes_once() -> None:
    assert supersigrok.badge_reply("hello") == "🧠 hello"
    assert supersigrok.badge_reply("🧠 already") == "🧠 already"
    assert supersigrok.badge_reply("  ") == ""


def test_nsfw_still_filters_non_supersigrok() -> None:
    assert apply_nsfw_filter("here is some porn for you") == NSFW_FILTER_REPLACEMENT


@pytest.mark.asyncio
async def test_dm_gate_upsell_and_entitled(monkeypatch) -> None:
    from sigrok.cogs.misc import Misc

    bot = MagicMock()
    bot.user = SimpleNamespace(id=1, name="Sigrok")
    cog = Misc(bot)
    monkeypatch.setattr(settings.bot, "prefix", ".")

    # Non-entitled DM → upsell
    channel = AsyncMock()
    author = SimpleNamespace(id=999, bot=False, roles=[])
    message = SimpleNamespace(
        author=author,
        guild=None,
        content="hey",
        channel=channel,
        mentions=[],
        id=1,
    )
    saved_users = list(settings.bot.supersigrok.user_ids)
    saved_everyone = settings.bot.supersigrok.everyone_until
    supersigrok.clear_runtime_grants()
    try:
        settings.bot.supersigrok.user_ids = []
        settings.bot.supersigrok.everyone_until = None
        await cog.on_message(message)
        channel.send.assert_awaited()
        assert "SuperSigrok-only" in channel.send.await_args.args[0]
        assert ".supersigrok" in channel.send.await_args.args[0]

        # Prefix command ignored by DM LLM path
        channel.reset_mock()
        message.content = ".supersigrok"
        await cog.on_message(message)
        channel.send.assert_not_awaited()

        # Entitled DM → handler
        handled = AsyncMock()
        monkeypatch.setattr(cog, "_handle_bot_mention", handled)
        supersigrok.grant_user(999)
        message.content = "what's up"
        await cog.on_message(message)
        handled.assert_awaited_once()
    finally:
        settings.bot.supersigrok.user_ids = saved_users
        settings.bot.supersigrok.everyone_until = saved_everyone
        supersigrok.clear_runtime_grants()


def test_finish_reply_helpers_nsfw_and_badge() -> None:
    """Document the misc reply contract for SuperSigrok vs normal."""
    text = "here is some porn for you"
    # Normal path
    filtered = apply_nsfw_filter(text)
    assert filtered == NSFW_FILTER_REPLACEMENT
    # SuperSigrok path skips filter then badges
    entitled = supersigrok.badge_reply(text)
    assert entitled.startswith("🧠 ")
    assert "porn" in entitled
