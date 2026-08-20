"""SuperSigrok paid tier entitlements.

Grants come from:
- settings.toml user_ids / role_ids (comps)
- Stripe subscriptions persisted in SQLite (hydrated into runtime on boot)
- runtime grant_user / revoke_user (webhooks + admin)

Set bot.supersigrok.everyone_until to a future ISO datetime to temporarily grant Max
Thinking to all users (e.g. burn remaining OpenCode Go quota before a billing reset).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, Optional, Union
from urllib.parse import urlencode

from discord import Member, User
from loguru import logger

from sigrok.config import settings

# Runtime grants (Stripe / admin). Unioned with settings.toml.
_runtime_user_ids: set[int] = set()
_logged_everyone_until: Optional[datetime] = None

SUPERSIGROK_REPLY_BADGE = "🧠"

# Default canned line when the cheap rate-limit LLM call fails.
RATE_LIMIT_FALLBACK_REPLY = (
    "i can't put more effort into you for a while. try again later, or get SuperSigrok."
)


def granted_user_ids() -> set[int]:
    return set(settings.bot.supersigrok.user_ids) | _runtime_user_ids


def grant_user(user_id: int) -> None:
    """Mark a Discord user as SuperSigrok (payment / admin hook)."""
    _runtime_user_ids.add(int(user_id))


def revoke_user(user_id: int) -> None:
    """Revoke a runtime SuperSigrok grant. Does not remove settings.toml entries."""
    _runtime_user_ids.discard(int(user_id))


def clear_runtime_grants() -> None:
    _runtime_user_ids.clear()


async def hydrate_subscription_grants() -> int:
    """Load active Stripe subscription Discord IDs into the runtime grant set."""
    from sigrok import db

    await db.ensure_supersigrok_tables()
    ids = await db.list_active_supersigrok_discord_ids()
    for uid in ids:
        _runtime_user_ids.add(int(uid))
    logger.info(f"Hydrated {len(ids)} SuperSigrok subscription grant(s) from DB")
    return len(ids)


def _role_ids_of(author: Union[Member, User]) -> set[int]:
    roles = getattr(author, "roles", None)
    if not roles:
        return set()
    return {int(r.id) for r in roles}


def _as_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def everyone_max_thinking_active() -> bool:
    """True while bot.supersigrok.everyone_until is still in the future."""
    global _logged_everyone_until
    until = settings.bot.supersigrok.everyone_until
    if until is None:
        return False
    now = datetime.now(timezone.utc)
    active = now < _as_utc(until)
    if active and until != _logged_everyone_until:
        _logged_everyone_until = until
        logger.info(
            f"SuperSigrok everyone Max Thinking active until {_as_utc(until).isoformat()}"
        )
    return active


def is_comp_user(user_id: int) -> bool:
    return int(user_id) in set(settings.bot.supersigrok.user_ids)


def is_supersigrok(
    author: Union[Member, User, None] = None,
    *,
    user_id: Optional[int] = None,
    role_ids: Optional[Iterable[int]] = None,
) -> bool:
    """True if this Discord user should get Max Thinking replies."""
    if everyone_max_thinking_active():
        return True
    return can_sponsor_guild(author, user_id=user_id, role_ids=role_ids)


def can_sponsor_guild(
    author: Union[Member, User, None] = None,
    *,
    user_id: Optional[int] = None,
    role_ids: Optional[Iterable[int]] = None,
) -> bool:
    """True if this user may add/keep Sigrok on a subscriber server.

    Excludes the everyone_until promo — only real grants (Stripe, comps, roles).
    """
    cfg = settings.bot.supersigrok
    uid = int(user_id) if user_id is not None else (int(author.id) if author else None)
    if uid is not None and uid in granted_user_ids():
        return True
    wanted = {int(r) for r in cfg.role_ids}
    if not wanted:
        return False
    have = set(int(r) for r in role_ids) if role_ids is not None else (
        _role_ids_of(author) if author is not None else set()
    )
    return bool(wanted & have)


def invite_url(client_id: int, *, permissions: Optional[int] = None) -> str:
    """Build a Discord OAuth2 bot invite URL for DMing SuperSigrok users."""
    perms = (
        int(permissions)
        if permissions is not None
        else int(settings.bot.supersigrok.invite_permissions)
    )
    query = urlencode(
        {
            "client_id": str(int(client_id)),
            "permissions": str(perms),
            "scope": "bot",
        }
    )
    return f"https://discord.com/oauth2/authorize?{query}"


def humanize_duration(seconds: float) -> str:
    """Short human duration for rate-limit copy (e.g. '40 minutes')."""
    secs = max(0, int(seconds))
    if secs < 60:
        return "a minute"
    minutes = (secs + 59) // 60  # round up
    if minutes < 60:
        return "1 minute" if minutes == 1 else f"{minutes} minutes"
    hours = (minutes + 59) // 60
    if hours < 48:
        return "1 hour" if hours == 1 else f"{hours} hours"
    days = (hours + 23) // 24
    return "1 day" if days == 1 else f"{days} days"


def badge_reply(text: str) -> str:
    """Prefix a SuperSigrok LLM reply with the hero badge."""
    body = (text or "").strip()
    if not body:
        return body
    if body.startswith(SUPERSIGROK_REPLY_BADGE):
        return body
    return f"{SUPERSIGROK_REPLY_BADGE} {body}"


def toml_authorized_guilds() -> set[int]:
    return {entry.guild for entry in settings.bot.whitelist}


async def guild_is_authorized(guild_id: int) -> bool:
    """True if guild is in TOML whitelist or a still-valid subscriber guild."""
    from sigrok import db

    if int(guild_id) in toml_authorized_guilds():
        return True
    row = await db.read_subscriber_guild(int(guild_id))
    if row is None:
        return False
    if can_sponsor_guild(user_id=row.sponsor_user_id):
        return True
    if row.grace_until is not None and datetime.now(timezone.utc) < _as_utc(
        row.grace_until
    ):
        return True
    return False
