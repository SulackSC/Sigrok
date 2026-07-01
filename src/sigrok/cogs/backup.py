import asyncio
import gzip
import os
import re
import shutil
from datetime import datetime
from urllib.parse import urlparse

import discord
from discord.ext import commands, tasks
from loguru import logger

from sigrok.config import settings


def create_backup() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"backup_{timestamp}.sqlite3.gz"
    path = os.path.join(settings.database.backup_dir, filename)

    with (
        open(urlparse(settings.database.url).path.lstrip("/"), "rb") as f_in,
        gzip.open(path, "wb") as f_out,
    ):
        shutil.copyfileobj(f_in, f_out)

    logger.info(f"Created backup: {filename}")
    remove_old_backups()
    return filename


def remove_old_backups() -> None:
    """Keep only the newest `retention` backup files."""
    backup_dir = settings.database.backup_dir
    retention = max(0, int(settings.database.retention))
    backups: list[tuple[float, str]] = []
    for backup in os.listdir(backup_dir):
        if not re.match(r"^backup_\d+_\d+\.sqlite3\.gz$", backup):
            logger.warning(f"Skipping unrecognized backup filename: {backup}")
            continue
        full_path = os.path.join(backup_dir, backup)
        try:
            backups.append((os.path.getmtime(full_path), full_path))
        except OSError as exc:
            logger.warning(f"Failed to stat backup `{backup}`: {exc}")

    backups.sort(key=lambda row: row[0], reverse=True)
    for _, path in backups[retention:]:
        try:
            os.remove(path)
            logger.info(f"Deleted old backup: {os.path.basename(path)}")
        except OSError as exc:
            logger.warning(f"Failed to delete backup `{path}`: {exc}")


class Backup(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        os.makedirs(settings.database.backup_dir, exist_ok=True)
        self.backup_task.start()

    def cog_unload(self):
        self.backup_task.cancel()
        logger.info("Backup cog unloaded and task cancelled.")

    @tasks.loop(hours=24)
    async def backup_task(self):
        try:
            name = await asyncio.to_thread(create_backup)
            logger.info(f"Daily backup created: {name}")
        except Exception as e:
            logger.error(f"Scheduled backup failed: {e}")

    @backup_task.before_loop
    async def before_backup_task(self):
        await self.bot.wait_until_ready()
        await asyncio.sleep(24 * 60 * 60)


def setup(bot):
    bot.add_cog(Backup(bot))
