# Sigrok — Discord channel permissions

Sigrok **tries** to send every answer as a **reply** to your `@Sigrok` ping. Discord requires **Read Message History** for that; without it, the bot still **posts the text** as a normal message (not threaded).

| Permission | Why |
|------------|-----|
| **Read Message History** | Needed for reply threading (API error `160002` without it) and for reading pinned channel rules. |
| **Send Messages** | To post at all. |
| **View Channel** | To see the channel. |
| **View Audit Log** | So SuperSigrok “add to server” can see who invited the bot (sponsor check). |
| **Add Reactions** | Fallback reactions when a reply fails. |

Optional but useful: **Embed Links**, **Attach Files**. Voice: **Connect**, **Speak** (for the `voice_rec` cog).

The SuperSigrok invite URL (`.supersigrok invite`) requests these via `bot.supersigrok.invite_permissions` (default `3263680`).

### Public Bot (Developer Portal)

For subscribers to add Sigrok to their own servers, enable **Public Bot** on the Discord application. Authorization still happens on join: only SuperSigrok sponsors (or TOML-whitelisted guilds) keep the bot; everyone else is left immediately.

### Channel rules (pinned)

Mods can pin a message that **starts and ends** with `@Sigrok`. The text between those mentions is injected into Sigrok’s system prompt for replies **in that channel only**. Example:

```
@Sigrok
no spoilers for the current season
keep answers under 3 sentences
@Sigrok
```

Any matching pin counts (Discord’s pin permission gates who can create them). Multiple matching pins are combined. Unpin or edit the pin to change the rules (cache refreshes within about a minute).

Configure these on the bot role or channel overrides (e.g. `#bots`).
