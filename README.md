# Sigrok

Sigrok is a **Discord bot** built on [py-cord](https://github.com/Pycord-Development/pycord). It answers when pinged, using a **local or hosted language model** as the brain, with tunable prompts and optional web search. The codebase also includes **Bluesky** integration, **voice recording** chunks, **scheduled and conditional channel posts**, and **database backups**.

Upstream lineage: forked from [BrokenDesign/iqbot](https://github.com/BrokenDesign/iqbot). The legacy `users.iq` column was renamed to **`rating`**; the original IQ-bot wager flow has been removed.

The bot creates `data.db` on first run via `Base.metadata.create_all`. There are no schema migrations — if the model changes break an existing dev DB, delete `data.db` and let it re-init.

## Features

- **Whitelist** — Only configured guilds/channels are used; the bot leaves servers that are not allowed.
- **Generative replies** — `@Sigrok` in **any channel** of a **whitelisted guild**; backends include Ollama, llama.cpp (`llama-server`), OpenAI, and Anthropic (see configuration).
- **Social** — Optional Bluesky and X (Twitter) mention bots; replies when `@mentioned` with the same Sigrok personality as Discord.
- **Streaming chat** — Optional Twitch, YouTube Live Chat, and Kick integration (`streaming_chat` cog); replies on `@mention` only.
- **Voice** — Chunked recording from voice channels (see `voice_rec` cog).
- **Automation** — Cron-like and one-shot jobs, join/leave messages, timed posts (`conditional_posts` cog).
- **Data** — SQLite via SQLAlchemy, optional rolling backups (`backup` cog).

## Requirements

- **Python** 3.10+ (see `pyproject.toml`).
- A **Discord application** and bot token.
- For local models: **Ollama** or **llama.cpp** `llama-server` (OpenAI-compatible HTTP).

## Quick start

1. Clone the repo and install dependencies, e.g. with [Poetry](https://python-poetry.org/):

   ```bash
   poetry install
   ```

2. Copy **`settings.toml.example`** → **`settings.toml`** and set your Discord **owner** id, **whitelist** guild/channel ids, and other options.

3. Add **`.secrets.toml`** in the project root (not committed; see `.gitignore`) with at least the bot token:

   ```toml
   [tokens]
   bot = "<discord bot token>"
   gpt = "<openai api key, optional>"
   hf = "<huggingface token, optional>"
   anthropic = "<anthropic key, optional>"
   opencode_go = "<opencode go api key, optional>"
   ```

4. From the repo root, run the bot:

   ```bash
   poetry run python src/sigrok/bot.py
   ```

   On first run, if `data.db` is missing, the app initializes the database.

5. **Channel permissions** — See [`BOT_PERMISSIONS.md`](BOT_PERMISSIONS.md) so replies can thread correctly.

## Configuration

| File | Purpose |
|------|---------|
| `settings.toml` | Bot prefix, cogs, whitelist, `[genai]` model and tuning, social toggles (local only; gitignored). |
| `.secrets.toml` | API tokens merged over `settings.toml`. |
| `resources/sigrok_personality_prompt.txt` | Main Discord personality / system text for the model (see code). |

A `sigrok.service` systemd unit example lives under `deploy/systemd/`.

## Language models (`[genai]`)

In `settings.toml`, `[genai]` selects the backend via the **model prefix**:

- **`ollama/<tag>`** — HTTP `POST {base_url}/api/chat` (default Ollama port `11434`).
- **`llamacpp/<model_id>`** — OpenAI-compatible `POST .../v1/chat/completions` against [llama.cpp](https://github.com/ggerganov/llama.cpp) `llama-server` (this repo’s deploy notes often use port `8081` when `8080` is taken).
- **`opencode-go/<model_id>`** — [OpenCode Go](https://opencode.ai/docs/go/) hosted models (e.g. `deepseek-v4-flash`). Set `tokens.opencode_go` in `.secrets.toml`. Default `base_url` is `https://opencode.ai/zen/go` (normalized to `/v1`).

For llama.cpp, set `base_url` to the server root (e.g. `http://127.0.0.1:8081`) or a URL already ending in `/v1`; the client normalizes to a single `/v1`. A dummy key is fine on the wire; the client may send `sk-no-key-required`.

Example OpenCode Go + DeepSeek V4 Flash:

```toml
[genai]
model = "opencode-go/deepseek-v4-flash"
base_url = "https://opencode.ai/zen/go"
request_timeout = 180
```

Web search tooling needs `llama-server` built with tool support (often `--jinja`) and a chat template that matches your GGUF. See upstream server and function-calling docs.

Generation tuning accepts **`temperature`** / **`repeat_penalty`** or the legacy names **`ollama_temperature`** / **`ollama_repeat_penalty`**.

### llama.cpp runtime notes

- The project expects a llama.cpp `llama-server` endpoint compatible with `/v1/chat/completions`.
- In recent testing, using **Q4_1 KV cache quantization** provided the best memory/perf trade-off for long Discord context windows.
- Keep your model alias (`llamacpp/<alias>`) aligned with the server `-a` alias; mismatch here looks like model failure from the bot side.
- If tool calling is enabled, keep a tool-capable server build and compatible chat template; Sigrok now dedupes sources and applies stricter public-URL fetch safeguards before using tool results.

## Streaming chat (`[streaming]`)

Sigrok can reply in **Twitch**, **YouTube Live Chat**, and **Kick** when viewers `@mention` the configured bot handle. Web search tools are **disabled** on streaming platforms to keep latency acceptable.

1. Add `"streaming_chat"` to `[bot].cogs` (included in `settings.toml.example`).
2. Enable platforms under `[streaming]` in `settings.toml`.
3. Add credentials to `.secrets.toml`:

```toml
[tokens]
twitch_client_id = "<twitch dev app client id>"
twitch_client_secret = "<twitch dev app client secret>"
twitch_bot_access_token = "<bot user oauth token>"
twitch_bot_refresh_token = "<bot user refresh token>"

[tokens.youtube]
client_id = "<google oauth client id>"
client_secret = "<google oauth client secret>"
refresh_token = "<youtube channel oauth refresh token>"

[tokens.kick]
client_id = "<kick dev app client id>"
client_secret = "<kick dev app client secret>"
access_token = "<kick oauth access token>"
refresh_token = "<kick oauth refresh token>"
```

### Twitch setup

- Create a [Twitch Developer](https://dev.twitch.tv/console/apps) application.
- Create a dedicated bot Twitch account; authorize it with scopes `user:read:chat`, `user:write:chat`, and `user:bot`.
- Authorize the bot on each broadcaster channel (`channel:bot` scope from the broadcaster).
- Set `streaming.twitch.bot_user_id`, `owner_user_id`, `bot_username`, and `channels` (broadcaster logins).

### YouTube setup

- Enable **YouTube Data API v3** in Google Cloud; create OAuth credentials for a desktop or web app.
- Obtain a refresh token for the channel that will post chat messages.
- Set `streaming.youtube.channels` to YouTube channel IDs (`UC...`).

### Kick setup

- Create a [Kick Developer](https://kick.com/settings/developer) application.
- `receive_mode = "websocket"` connects via Pusher (no public URL required).
- `receive_mode = "webhook"` uses official EventSub; expose `webhook_port` via HTTPS (e.g. Cloudflare Tunnel) in production.

### Rate limits

Tune `global_reply_cooldown_seconds` and `per_user_cooldown_seconds` under `[streaming]` to avoid chat spam during busy streams.

## Social (`[social.bluesky]` / `[social.x]`)

Sigrok can reply on **Bluesky** and **X** when your account is `@mentioned`, using the same generative reply pipeline as Discord (with platform-specific prompts and character limits).

1. Add `"bluesky"` and/or `"x"` to `[bot].cogs`.
2. Enable the platform under `[social]` in `settings.toml`.
3. Add credentials to `.secrets.toml`:

```toml
[social.bluesky]
identifier = "your.handle.bsky.social"
password = "your_app_password"

[social.x]
bearer_token = "<oauth2 user access token>"
```

### Bluesky setup

- Create an [app password](https://bsky.app/settings/app-passwords) for the bot account.
- Set `social.bluesky.enabled = true` in `settings.toml`.

### X setup

- Create a [developer project and app](https://developer.x.com/en/portal/dashboard) with **OAuth 2.0 user context**.
- Required scopes: `tweet.read`, `tweet.write`, `users.read`.
- Generate a user access token for the bot account (the unbanned `@sigrok` account).
- Put the token in `.secrets.toml` as `social.x.bearer_token`.
- Set `social.x.enabled = true` in `settings.toml`.
- `user_id` and `username` are optional; the client resolves them via `GET /2/users/me` on first use.

On first startup, the poller bootstraps from current mentions without replying to history. Only new mentions after that get replies.

## License

GPLv2 — see [`LICENSE`](LICENSE).
