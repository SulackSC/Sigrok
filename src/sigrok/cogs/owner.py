import tempfile

from discord import File
from discord.ext import commands
from discord.ext.commands import Context


class Owner(commands.Cog):
    bot: commands.Bot

    def __init__(self, bot):
        self.bot = bot

    async def respond_with_file(self, ctx: Context, content: str) -> None:
        out = tempfile.NamedTemporaryFile(
            dir=".", prefix="conversation.", suffix=".txt", delete=False
        )
        out.write(content.encode("utf-8"))
        out.flush()
        try:
            file = File(out.name, filename="conversation.txt")
            await ctx.respond(file=file)  # type: ignore
        except Exception as err:
            print(err)
        finally:
            out.close()


def setup(bot):
    bot.add_cog(Owner(bot))
