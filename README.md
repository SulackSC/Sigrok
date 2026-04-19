# Sigrok

Sigrok is a **Discord bot** built on [py-cord](https://github.com/Pycord-Development/pycord). It answers when pinged, using a **local or hosted language model** as the brain, with tunable prompts and optional web search. The codebase also includes **Bluesky** integration, **voice recording** chunks, **scheduled and conditional channel posts**, and **database backups**.

Upstream lineage: forked from [BrokenDesign/iqbot](https://github.com/BrokenDesign/iqbot). The legacy `users.iq` column was renamed to **`rating`**; the original IQ-bot wager flow has been removed.

Run **`alembic upgrade head`** after upgrading if you already have a `data.db` from an older checkout.

## Features

- **Whitelist** — Only configured guilds/channels are used; the bot leaves servers that are not allowed.
- **Generative replies** — `@Sigrok` in **any channel** of a **whitelisted guild**; backends include Ollama, llama.cpp (`llama-server`), OpenAI, and Anthropic (see configuration).
- **Social** — Optional Bluesky posting; optional X (Twitter) bearer token support in config.
- **Voice** — Chunked recording from voice channels (see `voice_rec` cog).
- **Automation** — Cron-like and one-shot jobs, join/leave messages, timed posts (`conditional_posts` cog).
- **Data** — SQLite via SQLAlchemy, Alembic migrations, optional rolling backups (`backup` cog).

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

Systemd unit examples live under `deploy/systemd/` (e.g. `sigrok.service`, `llama-server.service`).

## Language models (`[genai]`)

In `settings.toml`, `[genai]` selects the backend via the **model prefix**:

- **`ollama/<tag>`** — HTTP `POST {base_url}/api/chat` (default Ollama port `11434`).
- **`llamacpp/<model_id>`** — OpenAI-compatible `POST .../v1/chat/completions` against [llama.cpp](https://github.com/ggerganov/llama.cpp) `llama-server` (this repo’s deploy notes often use port `8081` when `8080` is taken).

For llama.cpp, set `base_url` to the server root (e.g. `http://127.0.0.1:8081`) or a URL already ending in `/v1`; the client normalizes to a single `/v1`. A dummy key is fine on the wire; the client may send `sk-no-key-required`.

Web search tooling needs `llama-server` built with tool support (often `--jinja`) and a chat template that matches your GGUF. See upstream server and function-calling docs.

Generation tuning accepts **`temperature`** / **`repeat_penalty`** or the legacy names **`ollama_temperature`** / **`ollama_repeat_penalty`**.

## License

GPLv2 — see [`LICENSE`](LICENSE).
