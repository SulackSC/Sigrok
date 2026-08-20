from typing import Optional

from discord import Forbidden, Member
from discord.ext import commands
from discord.ext.commands import Context

from sigrok import db
from sigrok.relationships import (
    DISPOSITIONS,
    LAST_VIBES,
    relationship_state_dict,
)


class Owner(commands.Cog):
    bot: commands.Bot

    def __init__(self, bot):
        self.bot = bot

    @staticmethod
    async def _send_private(ctx: Context, content: str) -> None:
        try:
            await ctx.author.send(content)
        except Forbidden:
            await ctx.send(
                "i couldn't dm you. enable dms for this server and try again",
                delete_after=8,
            )

    @commands.group(name="relationship", invoke_without_command=True)
    async def relationship(self, ctx: Context) -> None:
        """Private relationship controls; never used in normal replies."""
        await self._send_private(
            ctx,
            "`.relationship show [@user]`\n"
            "`.relationship set @user <field> <value>`\n"
            "`.relationship reset [@user]`\n"
            "fields: affinity, trust, disposition, roast_level, "
            "engagement_weight, last_vibe",
        )

    @relationship.command(name="show")
    async def relationship_show(
        self, ctx: Context, member: Optional[Member] = None
    ) -> None:
        if ctx.guild is None:
            return
        target = member or ctx.author
        row = await db.read_relationship(ctx.guild.id, target.id)
        state = relationship_state_dict(row)
        rendered = "\n".join(f"{key}: {value}" for key, value in state.items())
        await self._send_private(
            ctx,
            f"guild={ctx.guild.id} user={target.id}\n{rendered}",
        )

    @relationship.command(name="set")
    async def relationship_set(
        self,
        ctx: Context,
        member: Member,
        field: str,
        *,
        value: str,
    ) -> None:
        if ctx.guild is None:
            return
        field = field.strip().lower()
        raw_value = value.strip().lower()
        parsed: int | str | None
        if field in {"affinity", "trust", "roast_level", "engagement_weight"}:
            try:
                parsed = int(raw_value)
            except ValueError:
                await self._send_private(ctx, f"{field} needs an integer")
                return
        elif field == "disposition":
            if raw_value not in DISPOSITIONS:
                await self._send_private(
                    ctx, "disposition: ally, neutral, rival, or ignore"
                )
                return
            parsed = raw_value
        elif field == "last_vibe":
            if raw_value in {"none", "null", "clear"}:
                parsed = None
            elif raw_value in LAST_VIBES:
                parsed = raw_value
            else:
                await self._send_private(
                    ctx,
                    "last_vibe: chill, friendly, spicy, hostile, correction, "
                    "dismissive, or clear",
                )
                return
        else:
            await self._send_private(ctx, "that field is not editable")
            return

        row = await db.set_relationship_field(
            ctx.guild.id, member.id, field, parsed
        )
        await self._send_private(
            ctx,
            f"updated guild={ctx.guild.id} user={member.id}\n"
            + "\n".join(
                f"{key}: {item}"
                for key, item in relationship_state_dict(row).items()
            ),
        )

    @relationship.command(name="reset")
    async def relationship_reset(
        self, ctx: Context, member: Optional[Member] = None
    ) -> None:
        if ctx.guild is None:
            return
        target = member or ctx.author
        await db.reset_relationship(ctx.guild.id, target.id)
        await self._send_private(ctx, f"reset guild={ctx.guild.id} user={target.id}")


def setup(bot):
    bot.add_cog(Owner(bot))
