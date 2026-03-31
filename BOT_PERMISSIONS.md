# Sigrok — Discord channel permissions

Sigrok **tries** to send every answer as a **reply** to your `@Sigrok` ping. Discord requires **Read Message History** for that; without it, the bot still **posts the text** as a normal message (not threaded).

| Permission | Why |
|------------|-----|
| **Read Message History** | Needed for reply threading (API error `160002` without it). |
| **Send Messages** | To post at all. |
| **View Channel** | To see the channel. |

Optional but useful: **Embed Links**, **Attach Files** (if you add those features later).

Configure these on the bot role or channel overrides (e.g. `#bots`).
