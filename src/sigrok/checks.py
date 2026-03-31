from discord.ext.commands import Context

from sigrok.config import settings


def bot_owner(ctx: Context):
    return ctx.author.id == settings.bot.owner.id


def bot_manager(ctx: Context):
    return (
        ctx.author.id == settings.bot.owner.id
        or ctx.author.guild_permissions.administrator  # type: ignore
        # TODO: add role check
    )
