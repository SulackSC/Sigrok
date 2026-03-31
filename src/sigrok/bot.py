# type: ignore

import asyncio
import os

from discord import Intents
from discord.ext import commands
from loguru import logger

from sigrok import db
from sigrok.config import settings

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


@bot.event
async def on_ready():
    await bot.wait_until_ready()
    logger.info(f"{bot.user.name} ready and raring to go")
    authorized_guilds = {entry.guild for entry in settings.bot.whitelist}
    for guild in bot.guilds:
        if guild.id not in authorized_guilds:
            logger.info(f"Leaving unauthorized guild: {guild.name} ({guild.id})")
            await guild.leave()

    # Keep Discord's registered slash-command list empty now that the bot no
    # longer exposes any slash commands.
    present_guilds = [
        guild.id for guild in bot.guilds if guild.id in authorized_guilds
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
    if guild.id not in {entry.guild for entry in settings.bot.whitelist}:
        logger.info(f"Auto-leaving unauthorized guild: {guild.name} ({guild.id})")
        await guild.leave()


logger.info("Logging in...")
bot.run(settings.tokens.bot)
