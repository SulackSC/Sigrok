"""SuperSigrok subscribe / manage commands and Stripe webhook lifecycle."""

from __future__ import annotations

from datetime import datetime, timezone

from discord import Forbidden
from discord.ext import commands, tasks
from discord.ext.commands import Context
from loguru import logger

from sigrok import db
from sigrok.billing import (
    StripeNotConfiguredError,
    StripeWebhookServer,
    create_checkout_session,
    create_portal_session,
    stripe_configured,
)
from sigrok.billing.stripe_billing import set_notify_bot
from sigrok.config import settings
from sigrok.supersigrok import (
    can_sponsor_guild,
    everyone_max_thinking_active,
    guild_is_authorized,
    hydrate_subscription_grants,
    invite_url,
    is_comp_user,
    is_supersigrok,
    toml_authorized_guilds,
)


class SuperSigrokCog(commands.Cog):
    bot: commands.Bot

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self._webhook = StripeWebhookServer()
        set_notify_bot(bot)

    def cog_unload(self) -> None:
        if self.guild_grace_sweeper.is_running():
            self.guild_grace_sweeper.cancel()
        if self._webhook.running:
            self.bot.loop.create_task(self._webhook.stop())

    @commands.Cog.listener()
    async def on_ready(self) -> None:
        try:
            await db.ensure_supersigrok_tables()
            await db.ensure_subscriber_guild_tables()
            await db.ensure_free_usage_tables()
            await hydrate_subscription_grants()
        except Exception:
            logger.exception("SuperSigrok DB init / grant hydrate failed")
        try:
            await self._webhook.start()
        except Exception:
            logger.exception("Failed to start Stripe webhook server")
        if not self.guild_grace_sweeper.is_running():
            self.guild_grace_sweeper.start()

    @staticmethod
    async def _send_private(ctx: Context, content: str) -> None:
        try:
            await ctx.author.send(content)
            if ctx.guild is not None:
                await ctx.send("check your DMs", delete_after=8)
        except Forbidden:
            await ctx.send(content, delete_after=60)

    def _invite_link(self) -> str:
        if self.bot.user is None:
            raise RuntimeError("Bot user is not ready yet")
        return invite_url(self.bot.user.id)

    async def _format_server_line(self, row: db.SubscriberGuild) -> str:
        guild = self.bot.get_guild(int(row.guild_id))
        name = guild.name if guild is not None else "unknown"
        grace = ""
        if row.grace_until is not None:
            grace = f" (grace until {row.grace_until.isoformat()}Z)"
        return f"**{name}** (`{row.guild_id}`){grace}"

    @commands.group(name="supersigrok", invoke_without_command=True)
    async def supersigrok_group(self, ctx: Context) -> None:
        """Subscribe to SuperSigrok (Max Thinking + DMs + one server)."""
        await self.supersigrok_buy(ctx)

    @supersigrok_group.command(name="buy")
    async def supersigrok_buy(self, ctx: Context) -> None:
        if not stripe_configured():
            await self._send_private(
                ctx,
                "SuperSigrok billing isn't configured yet "
                "(need tokens.stripe_secret_key in .secrets.toml).",
            )
            return
        if is_supersigrok(ctx.author) and not everyone_max_thinking_active():
            row = await db.read_supersigrok_subscription(ctx.author.id)
            if row is not None and row.is_active:
                prefix = settings.bot.prefix
                await self._send_private(
                    ctx,
                    "You're already on SuperSigrok. "
                    f"Use `{prefix}supersigrok manage` for billing.\n"
                    f"Your one server: `{prefix}supersigrok server`\n"
                    f"Add Sigrok: `{prefix}supersigrok join`\n"
                    f"Remove Sigrok: `{prefix}supersigrok leave`",
                )
                return
        try:
            url = await create_checkout_session(ctx.author.id)
        except StripeNotConfiguredError as exc:
            await self._send_private(ctx, str(exc))
            return
        except Exception:
            logger.exception("Failed to create Stripe Checkout session")
            await self._send_private(
                ctx, "Couldn't start checkout right now. Try again later."
            )
            return
        prefix = settings.bot.prefix
        await self._send_private(
            ctx,
            "SuperSigrok checkout (Max Thinking + DMs + one server, NSFW filter off):\n"
            f"{url}\n"
            f"After you subscribe, DM `{prefix}supersigrok join` to add me to a server "
            f"(and `{prefix}supersigrok leave` to pull me out).",
        )

    @supersigrok_group.command(name="join")
    async def supersigrok_join(self, ctx: Context) -> None:
        """DM an invite link to add Sigrok to your one sponsored server."""
        await self._join_invite(ctx)

    @supersigrok_group.command(name="invite")
    async def supersigrok_invite(self, ctx: Context) -> None:
        """Alias for join — DM the add-to-server link."""
        await self._join_invite(ctx)

    async def _join_invite(self, ctx: Context) -> None:
        if not can_sponsor_guild(ctx.author):
            await self.supersigrok_buy(ctx)
            return
        prefix = settings.bot.prefix
        existing = await db.read_subscriber_guild_for_sponsor(ctx.author.id)
        if existing is not None:
            line = await self._format_server_line(existing)
            await self._send_private(
                ctx,
                f"SuperSigrok includes **one** server. You're already on {line}.\n"
                f"DM `{prefix}supersigrok leave` to remove me first, then "
                f"`{prefix}supersigrok join` again.",
            )
            return
        try:
            link = self._invite_link()
        except RuntimeError:
            await self._send_private(ctx, "Bot isn't ready yet — try again in a moment.")
            return
        await self._send_private(
            ctx,
            "Add Sigrok to **one** server you manage (you must be the person who adds the bot):\n"
            f"{link}\n"
            f"If I bounce out, run `{prefix}supersigrok claim` in that server "
            "(Manage Server required).\n"
            f"Later: `{prefix}supersigrok leave` from DMs to pull me out.",
        )

    @supersigrok_group.command(name="leave")
    async def supersigrok_leave(self, ctx: Context) -> None:
        """Leave the sponsor's one server (usable from DMs)."""
        if not can_sponsor_guild(ctx.author) and not await db.read_subscriber_guild_for_sponsor(
            ctx.author.id
        ):
            await self._send_private(
                ctx,
                "No SuperSigrok server on file. "
                f"Subscribe with `{settings.bot.prefix}supersigrok buy`.",
            )
            return
        row = await db.read_subscriber_guild_for_sponsor(ctx.author.id)
        if row is None:
            await self._send_private(
                ctx,
                "I'm not in a server for you right now. "
                f"DM `{settings.bot.prefix}supersigrok join` to add me.",
            )
            return
        guild_id = int(row.guild_id)
        line = await self._format_server_line(row)
        guild = self.bot.get_guild(guild_id)
        await db.delete_subscriber_guild(guild_id)
        left = False
        if guild is not None:
            try:
                await guild.leave()
                left = True
            except Exception:
                logger.exception(f"Failed to leave sponsored guild {guild_id}")
        if left:
            await self._send_private(
                ctx,
                f"Left {line}.\n"
                f"DM `{settings.bot.prefix}supersigrok join` when you want me somewhere else.",
            )
        else:
            await self._send_private(
                ctx,
                f"Cleared your sponsored server record for {line}. "
                "I wasn't in that guild (or couldn't leave); you're free to join another.",
            )

    @supersigrok_group.command(name="server")
    async def supersigrok_server(self, ctx: Context) -> None:
        """Show the one SuperSigrok server for this user (DM-friendly)."""
        row = await db.read_subscriber_guild_for_sponsor(ctx.author.id)
        prefix = settings.bot.prefix
        if row is None:
            await self._send_private(
                ctx,
                "No sponsored server yet.\n"
                f"`{prefix}supersigrok join` — get the invite link\n"
                f"`{prefix}supersigrok leave` — remove me later",
            )
            return
        line = await self._format_server_line(row)
        await self._send_private(
            ctx,
            f"Your SuperSigrok server: {line}\n"
            f"`{prefix}supersigrok leave` — pull me out\n"
            f"`{prefix}supersigrok join` — only after leave (one server per sub)",
        )

    @supersigrok_group.command(name="claim")
    async def supersigrok_claim(self, ctx: Context) -> None:
        if ctx.guild is None:
            await self._send_private(
                ctx,
                "Run `.supersigrok claim` in the server you want to keep Sigrok in "
                "(or use `.supersigrok join` from DMs for the invite link).",
            )
            return
        if ctx.guild.id in toml_authorized_guilds():
            await ctx.send("this server is already on the official whitelist", delete_after=12)
            return
        if not can_sponsor_guild(ctx.author):
            await self._send_private(
                ctx,
                "Only SuperSigrok sponsors can claim a server. "
                f"Run `{settings.bot.prefix}supersigrok` to subscribe.",
            )
            return
        perms = getattr(ctx.author, "guild_permissions", None)
        if perms is None or not (perms.manage_guild or perms.administrator):
            await ctx.send(
                "you need Manage Server (or Administrator) to claim this guild",
                delete_after=12,
            )
            return
        try:
            await db.assign_subscriber_guild(
                guild_id=ctx.guild.id, sponsor_user_id=ctx.author.id
            )
        except db.SponsorAlreadyHasGuildError as exc:
            await self._send_private(
                ctx,
                f"You already sponsor guild `{exc.existing_guild_id}`. "
                f"DM `{settings.bot.prefix}supersigrok leave` first.",
            )
            return
        except db.GuildAlreadySponsoredError:
            await ctx.send(
                "this server is already claimed by another SuperSigrok sponsor",
                delete_after=12,
            )
            return
        await ctx.send(
            f"got it — this is your one SuperSigrok server ({ctx.author.mention}). "
            f"DM `{settings.bot.prefix}supersigrok leave` to remove me later.",
            delete_after=20,
        )

    @supersigrok_group.command(name="manage")
    async def supersigrok_manage(self, ctx: Context) -> None:
        await self._portal(ctx)

    @supersigrok_group.command(name="cancel")
    async def supersigrok_cancel(self, ctx: Context) -> None:
        await self._portal(ctx)

    async def _portal(self, ctx: Context) -> None:
        if not settings.tokens.stripe_secret_key:
            await self._send_private(
                ctx,
                "SuperSigrok billing isn't configured yet "
                "(need tokens.stripe_secret_key).",
            )
            return
        try:
            url = await create_portal_session(ctx.author.id)
        except StripeNotConfiguredError as exc:
            await self._send_private(ctx, str(exc))
            return
        except Exception:
            logger.exception("Failed to create Stripe Customer Portal session")
            await self._send_private(
                ctx, "Couldn't open the billing portal right now. Try again later."
            )
            return
        await self._send_private(ctx, f"Manage / cancel SuperSigrok:\n{url}")

    @supersigrok_group.command(name="status")
    async def supersigrok_status(self, ctx: Context) -> None:
        entitled = is_supersigrok(ctx.author)
        parts = [f"entitled: {'yes' if entitled else 'no'}"]
        if everyone_max_thinking_active():
            parts.append("source: everyone_until promo")
        elif is_comp_user(ctx.author.id):
            parts.append("source: settings.toml comp")
        else:
            row = await db.read_supersigrok_subscription(ctx.author.id)
            if row is not None:
                parts.append(f"source: stripe ({row.status})")
                if row.current_period_end is not None:
                    parts.append(f"period_end: {row.current_period_end.isoformat()}Z")
            elif entitled:
                roles = getattr(ctx.author, "roles", None) or []
                role_ids = {int(r.id) for r in roles}
                if role_ids & set(settings.bot.supersigrok.role_ids):
                    parts.append("source: discord role")
                else:
                    parts.append("source: runtime grant")
            else:
                parts.append("source: none")
        if can_sponsor_guild(ctx.author):
            parts.append("can_add_server: yes (one server)")
        else:
            parts.append("can_add_server: no")
        sponsored = await db.read_subscriber_guild_for_sponsor(ctx.author.id)
        if sponsored is not None:
            parts.append(f"server: {await self._format_server_line(sponsored)}")
        else:
            parts.append("server: none")
            if can_sponsor_guild(ctx.author):
                try:
                    parts.append(f"join: {self._invite_link()}")
                except RuntimeError:
                    pass
        await self._send_private(ctx, "\n".join(parts))

    @tasks.loop(minutes=15)
    async def guild_grace_sweeper(self) -> None:
        try:
            await self._sweep_subscriber_guild_grace()
        except Exception:
            logger.exception("subscriber guild grace sweeper failed")

    @guild_grace_sweeper.before_loop
    async def _before_grace_sweeper(self) -> None:
        await self.bot.wait_until_ready()

    async def _sweep_subscriber_guild_grace(self) -> None:
        await db.ensure_subscriber_guild_tables()
        now = datetime.now(timezone.utc)
        prefix = settings.bot.prefix

        for row in await db.list_grace_guilds_needing_notify():
            user = self.bot.get_user(int(row.sponsor_user_id))
            if user is None:
                try:
                    user = await self.bot.fetch_user(int(row.sponsor_user_id))
                except Exception:
                    user = None
            grace_days = int(settings.bot.supersigrok.guild_grace_days)
            if user is not None:
                try:
                    await user.send(
                        f"Your SuperSigrok lapsed, so Sigrok will leave guild "
                        f"`{row.guild_id}` in about {grace_days} day(s) unless you "
                        f"resubscribe (`{prefix}supersigrok buy`). "
                        f"Or DM `{prefix}supersigrok leave` to remove me now."
                    )
                except Exception as exc:
                    logger.warning(
                        f"Could not DM grace notice to {row.sponsor_user_id}: {exc}"
                    )
            await db.mark_subscriber_guild_grace_notified(row.guild_id)

        for row in await db.list_expired_grace_guilds(now=now):
            # Sponsor may have regained entitlement without clearing grace yet.
            if can_sponsor_guild(user_id=row.sponsor_user_id):
                await db.clear_subscriber_guild_grace_for_sponsor(row.sponsor_user_id)
                continue
            if await guild_is_authorized(row.guild_id):
                continue
            guild = self.bot.get_guild(int(row.guild_id))
            if guild is not None:
                try:
                    await guild.leave()
                    logger.info(
                        f"Left subscriber guild after grace expired: "
                        f"{guild.name} ({guild.id})"
                    )
                except Exception:
                    logger.exception(f"Failed to leave guild {row.guild_id} after grace")
            await db.delete_subscriber_guild(row.guild_id)


def setup(bot: commands.Bot) -> None:
    bot.add_cog(SuperSigrokCog(bot))
