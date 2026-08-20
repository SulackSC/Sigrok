# type: ignore

import asyncio
import os

from discord import AuditLogAction, Forbidden, Intents
from discord.ext import commands
from loguru import logger

from sigrok import db
from sigrok.config import settings
from sigrok.supersigrok import (
    can_sponsor_guild,
    guild_is_authorized,
    toml_authorized_guilds,
)

for i in range(5):
    logger.add(f"logs/file{i}.log", rotation="10 MB")

intents = Intents(**settings.bot.intents.model_dump())
bot = commands.Bot(command_prefix=settings.bot.prefix, intents=intents)

if not os.path.exists("data.db"):
    logger.info("Initializing database...")
    asyncio.run(db.async_main())

for cog in settings.bot.cogs:
    logger.info(f"Loading {cog} cog...")
    bot.load_extension(f"sigrok.cogs.{cog}")


async def _find_bot_adder_user_id(guild) -> int | None:
    """Best-effort: who added the bot (needs View Audit Log)."""
    me = guild.me
    if me is None:
        return None
    try:
        async for entry in guild.audit_logs(
            limit=8, action=AuditLogAction.bot_add
        ):
            target = entry.target
            if target is not None and getattr(target, "id", None) == me.id:
                user = entry.user
                if user is not None:
                    return int(user.id)
    except Exception as exc:
        logger.warning(
            f"Could not read audit log for guild={guild.id} ({guild.name}): {exc}"
        )
    return None


async def _dm_user(user_id: int, content: str) -> None:
    user = bot.get_user(int(user_id))
    if user is None:
        try:
            user = await bot.fetch_user(int(user_id))
        except Exception:
            return
    try:
        await user.send(content)
    except (Forbidden, Exception) as exc:
        logger.warning(f"Could not DM user={user_id}: {exc}")


@bot.event
async def on_ready():
    await bot.wait_until_ready()
    await db.ensure_relationship_tables()
    await db.ensure_supersigrok_tables()
    await db.ensure_subscriber_guild_tables()
    await db.ensure_free_usage_tables()
    logger.info(f"{bot.user.name} ready and raring to go")

    for guild in list(bot.guilds):
        if await guild_is_authorized(guild.id):
            continue
        # Drop stale subscriber rows that are no longer valid.
        row = await db.read_subscriber_guild(guild.id)
        if row is not None:
            await db.delete_subscriber_guild(guild.id)
        logger.info(f"Leaving unauthorized guild: {guild.name} ({guild.id})")
        try:
            await guild.leave()
        except Exception:
            logger.exception(f"Failed to leave guild {guild.id}")

    # Keep Discord's registered slash-command list empty now that the bot no
    # longer exposes any slash commands.
    present_guilds = [
        guild.id for guild in bot.guilds if await guild_is_authorized(guild.id)
    ]
    try:
        await bot.sync_commands(guild_ids=present_guilds)
        logger.info(
            f"Cleared slash commands in {len(present_guilds)} authorized guild(s)."
        )
    except Exception as exc:
        logger.error(f"Failed to clear slash commands: {exc}")


@bot.event
async def on_guild_join(guild):
    if guild.id in toml_authorized_guilds():
        logger.info(f"Joined whitelisted guild: {guild.name} ({guild.id})")
        return

    existing = await db.read_subscriber_guild(guild.id)
    if existing is not None and await guild_is_authorized(guild.id):
        logger.info(
            f"Rejoined authorized subscriber guild: {guild.name} ({guild.id})"
        )
        return

    # Give Discord a moment to write the BOT_ADD audit entry.
    await asyncio.sleep(2.0)
    adder_id = await _find_bot_adder_user_id(guild)
    prefix = settings.bot.prefix
    if adder_id is not None and can_sponsor_guild(user_id=adder_id):
        try:
            await db.assign_subscriber_guild(
                guild_id=guild.id, sponsor_user_id=adder_id
            )
        except db.SponsorAlreadyHasGuildError as exc:
            logger.info(
                f"Rejecting guild {guild.id}: sponsor={adder_id} already has "
                f"guild={exc.existing_guild_id}"
            )
            await _dm_user(
                adder_id,
                f"SuperSigrok includes one server. You're already using guild "
                f"`{exc.existing_guild_id}`. "
                f"DM me `{prefix}supersigrok leave` first, then add me again.",
            )
            try:
                await guild.leave()
            except Exception:
                logger.exception(f"Failed to leave guild {guild.id}")
            return
        except db.GuildAlreadySponsoredError as exc:
            logger.info(
                f"Rejecting guild {guild.id}: already sponsored by {exc.sponsor_user_id}"
            )
            await _dm_user(
                adder_id,
                "That server is already claimed by another SuperSigrok sponsor.",
            )
            try:
                await guild.leave()
            except Exception:
                logger.exception(f"Failed to leave guild {guild.id}")
            return
        logger.info(
            f"Accepted subscriber guild {guild.name} ({guild.id}) "
            f"sponsored by user={adder_id}"
        )
        await _dm_user(
            adder_id,
            f"Sigrok is in **{guild.name}** now (your one SuperSigrok server).\n"
            f"DM `{prefix}supersigrok leave` anytime to pull me out.",
        )
        return

    logger.info(
        f"Auto-leaving unauthorized guild: {guild.name} ({guild.id}) "
        f"adder={adder_id}"
    )
    try:
        await guild.leave()
    except Exception:
        logger.exception(f"Failed to leave guild {guild.id}")


logger.info("Logging in...")
bot.run(settings.tokens.bot)
