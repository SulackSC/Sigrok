"""Stripe billing helpers for SuperSigrok subscriptions."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Optional

import stripe
from aiohttp import web
from loguru import logger

from sigrok import db
from sigrok.config import settings
from sigrok.supersigrok import grant_user, revoke_user

_REVOKED_STATUSES = frozenset(
    {"canceled", "unpaid", "past_due", "incomplete_expired", "paused"}
)


class StripeNotConfiguredError(RuntimeError):
    pass


def stripe_configured() -> bool:
    """True when Checkout can run (secret key + price id or default amount)."""
    tokens = settings.tokens
    if not tokens.stripe_secret_key:
        return False
    cfg = settings.bot.supersigrok
    if cfg.stripe_price_id:
        return True
    return int(cfg.stripe_amount_cents) > 0


def _stripe_client() -> stripe.StripeClient:
    key = settings.tokens.stripe_secret_key
    if not key:
        raise StripeNotConfiguredError("tokens.stripe_secret_key is not set")
    return stripe.StripeClient(key)


def _require_checkout_config() -> None:
    if not settings.tokens.stripe_secret_key:
        raise StripeNotConfiguredError("tokens.stripe_secret_key is not set")
    cfg = settings.bot.supersigrok
    if not cfg.stripe_price_id and int(cfg.stripe_amount_cents) <= 0:
        raise StripeNotConfiguredError(
            "Set bot.supersigrok.stripe_price_id or stripe_amount_cents"
        )


def _checkout_line_items() -> list[dict[str, Any]]:
    cfg = settings.bot.supersigrok
    if cfg.stripe_price_id:
        return [{"price": cfg.stripe_price_id, "quantity": 1}]
    return [
        {
            "quantity": 1,
            "price_data": {
                "currency": (cfg.stripe_currency or "usd").lower(),
                "unit_amount": int(cfg.stripe_amount_cents),
                "recurring": {"interval": "month"},
                "product_data": {
                    "name": cfg.stripe_product_name or "SuperSigrok",
                    "description": "Max Thinking replies, DMs, add Sigrok to your server, and NSFW filter bypass",
                },
            },
        }
    ]

def _period_end_from_subscription(sub: Any) -> Optional[datetime]:
    raw = getattr(sub, "current_period_end", None)
    if raw is None and isinstance(sub, dict):
        raw = sub.get("current_period_end")
    if raw is None:
        return None
    try:
        return datetime.fromtimestamp(int(raw), tz=timezone.utc)
    except (TypeError, ValueError, OSError):
        return None


def _status_from_subscription(sub: Any) -> str:
    status = getattr(sub, "status", None)
    if status is None and isinstance(sub, dict):
        status = sub.get("status")
    return str(status or "incomplete")


def _id_of(obj: Any) -> Optional[str]:
    if obj is None:
        return None
    if isinstance(obj, str):
        return obj
    value = getattr(obj, "id", None)
    if value is not None:
        return str(value)
    if isinstance(obj, dict) and obj.get("id") is not None:
        return str(obj["id"])
    return None


async def create_checkout_session(discord_user_id: int) -> str:
    """Create a Stripe Checkout Session and return the hosted URL."""
    _require_checkout_config()
    cfg = settings.bot.supersigrok
    uid = str(int(discord_user_id))
    row = await db.read_supersigrok_subscription(int(discord_user_id))
    customer_id = row.stripe_customer_id if row else None

    def _create_with_customer() -> str:
        client = _stripe_client()
        params: dict[str, Any] = {
            "mode": "subscription",
            "line_items": _checkout_line_items(),
            "success_url": cfg.stripe_success_url,
            "cancel_url": cfg.stripe_cancel_url,
            "client_reference_id": uid,
            "metadata": {"discord_user_id": uid},
            "subscription_data": {"metadata": {"discord_user_id": uid}},
        }
        if customer_id:
            params["customer"] = customer_id
        session = client.v1.checkout.sessions.create(params)
        url = getattr(session, "url", None) or (
            session.get("url") if isinstance(session, dict) else None
        )
        if not url:
            raise RuntimeError("Stripe Checkout Session missing url")
        return str(url)

    return await asyncio.to_thread(_create_with_customer)


async def create_portal_session(discord_user_id: int) -> str:
    """Create a Stripe Customer Portal session URL."""
    if not settings.tokens.stripe_secret_key:
        raise StripeNotConfiguredError("tokens.stripe_secret_key is not set")
    row = await db.read_supersigrok_subscription(int(discord_user_id))
    if row is None or not row.stripe_customer_id:
        raise StripeNotConfiguredError(
            "No Stripe customer on file. Subscribe with .supersigrok buy first."
        )
    customer_id = row.stripe_customer_id
    return_url = settings.bot.supersigrok.stripe_success_url

    def _create() -> str:
        client = _stripe_client()
        session = client.v1.billing_portal.sessions.create(
            {
                "customer": customer_id,
                "return_url": return_url,
            }
        )
        url = getattr(session, "url", None) or (
            session.get("url") if isinstance(session, dict) else None
        )
        if not url:
            raise RuntimeError("Stripe Customer Portal session missing url")
        return str(url)

    return await asyncio.to_thread(_create)


async def apply_subscription_entitlement(
    *,
    discord_user_id: int,
    stripe_customer_id: Optional[str],
    stripe_subscription_id: Optional[str],
    status: str,
    current_period_end: Optional[datetime] = None,
) -> None:
    row = await db.upsert_supersigrok_subscription(
        discord_user_id=int(discord_user_id),
        stripe_customer_id=stripe_customer_id,
        stripe_subscription_id=stripe_subscription_id,
        status=status,
        current_period_end=current_period_end,
    )
    if row.is_active:
        grant_user(int(discord_user_id))
        await db.clear_subscriber_guild_grace_for_sponsor(int(discord_user_id))
        logger.info(
            f"SuperSigrok granted discord_user_id={discord_user_id} "
            f"status={status} subscription={stripe_subscription_id}"
        )
        await _maybe_dm_welcome_invite(int(discord_user_id))
    else:
        revoke_user(int(discord_user_id))
        grace_days = int(settings.bot.supersigrok.guild_grace_days)
        updated = await db.start_subscriber_guild_grace_for_sponsor(
            int(discord_user_id), grace_days=grace_days
        )
        logger.info(
            f"SuperSigrok revoked discord_user_id={discord_user_id} "
            f"status={status} subscription={stripe_subscription_id} "
            f"guilds_in_grace={len(updated)}"
        )


_welcome_dm_sent: set[int] = set()
_notify_bot: Any = None


def set_notify_bot(bot: Any) -> None:
    """Register the Discord bot so webhook grants can DM invite links."""
    global _notify_bot
    _notify_bot = bot


async def _maybe_dm_welcome_invite(discord_user_id: int) -> None:
    """DM join/leave instructions once after a successful grant (best-effort)."""
    if discord_user_id in _welcome_dm_sent:
        return
    bot = _notify_bot
    if bot is None or bot.user is None:
        return

    user = bot.get_user(int(discord_user_id))
    if user is None:
        try:
            user = await bot.fetch_user(int(discord_user_id))
        except Exception:
            logger.warning(
                f"Could not fetch user {discord_user_id} for SuperSigrok welcome DM"
            )
            return
    prefix = settings.bot.prefix
    try:
        await user.send(
            "Welcome to SuperSigrok (Max Thinking + DMs + NSFW filter off).\n"
            f"One server included — DM `{prefix}supersigrok join` for the invite link, "
            f"or `{prefix}supersigrok leave` to pull me out later.\n"
            f"Billing: `{prefix}supersigrok manage`."
        )
        _welcome_dm_sent.add(int(discord_user_id))
    except Exception as exc:
        logger.warning(
            f"Could not DM SuperSigrok welcome to {discord_user_id}: {exc}"
        )


def _discord_user_id_from_mapping(obj: Any) -> Optional[int]:
    meta = getattr(obj, "metadata", None)
    if meta is None and isinstance(obj, dict):
        meta = obj.get("metadata") or {}
    if meta is None:
        meta = {}
    raw = None
    if hasattr(meta, "get"):
        raw = meta.get("discord_user_id")
    elif isinstance(meta, dict):
        raw = meta.get("discord_user_id")
    if raw is None:
        raw = getattr(obj, "client_reference_id", None)
        if raw is None and isinstance(obj, dict):
            raw = obj.get("client_reference_id")
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


async def handle_stripe_event(event: Any) -> None:
    event_type = getattr(event, "type", None) or (
        event.get("type") if isinstance(event, dict) else None
    )
    data_object = getattr(getattr(event, "data", None), "object", None)
    if data_object is None and isinstance(event, dict):
        data_object = (event.get("data") or {}).get("object")
    if not event_type or data_object is None:
        logger.warning("Stripe event missing type or data.object")
        return

    if event_type == "checkout.session.completed":
        await _handle_checkout_completed(data_object)
        return
    if event_type in {
        "customer.subscription.created",
        "customer.subscription.updated",
        "customer.subscription.deleted",
    }:
        await _handle_subscription_event(data_object)
        return
    logger.debug(f"Ignoring Stripe event type={event_type}")


async def _handle_checkout_completed(session: Any) -> None:
    discord_user_id = _discord_user_id_from_mapping(session)
    if discord_user_id is None:
        logger.warning("checkout.session.completed missing discord_user_id")
        return
    customer_id = _id_of(getattr(session, "customer", None) or (
        session.get("customer") if isinstance(session, dict) else None
    ))
    subscription_id = _id_of(
        getattr(session, "subscription", None)
        or (session.get("subscription") if isinstance(session, dict) else None)
    )
    status = "active"
    period_end = None
    if subscription_id and settings.tokens.stripe_secret_key:

        def _fetch_sub() -> Any:
            client = _stripe_client()
            return client.v1.subscriptions.retrieve(subscription_id)

        try:
            sub = await asyncio.to_thread(_fetch_sub)
            status = _status_from_subscription(sub)
            period_end = _period_end_from_subscription(sub)
            # Prefer subscription metadata if present.
            meta_uid = _discord_user_id_from_mapping(sub)
            if meta_uid is not None:
                discord_user_id = meta_uid
            customer_id = _id_of(getattr(sub, "customer", None)) or customer_id
        except Exception:
            logger.exception(
                f"Failed to retrieve subscription {subscription_id} after checkout"
            )
    await apply_subscription_entitlement(
        discord_user_id=discord_user_id,
        stripe_customer_id=customer_id,
        stripe_subscription_id=subscription_id,
        status=status,
        current_period_end=period_end,
    )


async def _handle_subscription_event(sub: Any) -> None:
    subscription_id = _id_of(sub)
    customer_id = _id_of(
        getattr(sub, "customer", None)
        or (sub.get("customer") if isinstance(sub, dict) else None)
    )
    status = _status_from_subscription(sub)
    if status in _REVOKED_STATUSES and status not in db.SUPERSIGROK_ACTIVE_STATUSES:
        # Keep status as reported by Stripe.
        pass
    period_end = _period_end_from_subscription(sub)
    discord_user_id = _discord_user_id_from_mapping(sub)
    if discord_user_id is None and subscription_id:
        row = await db.read_supersigrok_by_subscription_id(subscription_id)
        if row is not None:
            discord_user_id = int(row.discord_user_id)
    if discord_user_id is None and customer_id:
        row = await db.read_supersigrok_by_customer_id(customer_id)
        if row is not None:
            discord_user_id = int(row.discord_user_id)
    if discord_user_id is None:
        logger.warning(
            f"Subscription event missing discord_user_id "
            f"subscription={subscription_id} customer={customer_id}"
        )
        return
    await apply_subscription_entitlement(
        discord_user_id=discord_user_id,
        stripe_customer_id=customer_id,
        stripe_subscription_id=subscription_id,
        status=status,
        current_period_end=period_end,
    )


def construct_event(payload: bytes, sig_header: str) -> Any:
    secret = settings.tokens.stripe_webhook_secret
    if not secret:
        raise StripeNotConfiguredError("tokens.stripe_webhook_secret is not set")
    return stripe.Webhook.construct_event(payload, sig_header, secret)


class StripeWebhookServer:
    """aiohttp listener for Stripe webhooks (separate from Kick)."""

    def __init__(self) -> None:
        self._runner: Optional[web.AppRunner] = None

    @property
    def running(self) -> bool:
        return self._runner is not None

    async def start(self) -> None:
        if self._runner is not None:
            return
        if not settings.tokens.stripe_webhook_secret:
            logger.warning(
                "Stripe webhook not started: tokens.stripe_webhook_secret is empty"
            )
            return
        cfg = settings.bot.supersigrok
        app = web.Application()
        path = cfg.stripe_webhook_path.rstrip("/") or "/stripe/webhook"
        app.router.add_post(path, self._handler)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(
            self._runner, cfg.stripe_webhook_host, cfg.stripe_webhook_port
        )
        await site.start()
        logger.info(
            "Stripe webhook listening on http://{}:{}{}",
            cfg.stripe_webhook_host,
            cfg.stripe_webhook_port,
            path,
        )

    async def stop(self) -> None:
        if self._runner is None:
            return
        await self._runner.cleanup()
        self._runner = None

    async def _handler(self, request: web.Request) -> web.Response:
        payload = await request.read()
        sig_header = request.headers.get("Stripe-Signature", "")
        if not sig_header:
            return web.Response(status=400, text="missing Stripe-Signature")
        try:
            event = construct_event(payload, sig_header)
        except Exception as exc:
            logger.warning(f"Stripe webhook signature verification failed: {exc}")
            return web.Response(status=400, text="invalid signature")
        try:
            await handle_stripe_event(event)
        except Exception:
            logger.exception("Failed to handle Stripe webhook event")
            return web.Response(status=500, text="handler error")
        return web.Response(status=200, text="ok")
