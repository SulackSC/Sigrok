# Sigrok

Discord bot with ELO system for discord debates with some shitty SLM as the judge :)

Forked from `BrokenDesign/iqbot`:
https://github.com/BrokenDesign/iqbot

Expects a `.secrets.toml` file containing the API keys:

```toml
[tokens]
bot = "<discord bot token>"
gpt = "<openai api token>"
```

## Local language model (Ollama or llama.cpp)

`settings.toml` `[genai]` chooses the backend via the model prefix:

- `ollama/<tag>` — HTTP `POST {base_url}/api/chat` (default Ollama port `11434`).
- `llamacpp/<model_id>` — OpenAI-compatible `POST .../v1/chat/completions` against [llama.cpp](https://github.com/ggerganov/llama.cpp) `llama-server` (this host uses `8081` because OpenWebUI already uses `8080`).

For llama.cpp, set `base_url` to the server root (e.g. `http://127.0.0.1:8081`) or already suffixed with `/v1`; the client normalizes to a single `/v1`. Use a dummy API key on the wire; the bot sends `sk-no-key-required`.

Web search tools need `llama-server` built with tool support, typically `--jinja`, plus a chat template that matches your GGUF. See the upstream server README and function-calling docs.

Generation tuning accepts either `temperature` / `repeat_penalty` or the legacy names `ollama_temperature` / `ollama_repeat_penalty`.

