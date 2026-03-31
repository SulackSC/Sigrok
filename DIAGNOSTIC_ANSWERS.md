# Sigrok Diagnostic Answers (for Gemini's questions)

## 1. The Context Pipeline

### How are you building the prompt?

**Not ChatML.** We use Ollama's native chat format:
- **System role:** `{"role": "system", "content": system_prompt}` — contains `SIGROK_PERSONALITY_SYSTEM_PROMPT` + task instructions
- **User role:** Single message with:
  ```
  Recent Discord conversation:
  [ID: 1484264224434819123 | sulack]: oh shit thank you
  [ID: 1484264490185789716 | sulack]: How far away is the moon
  ...

  Question: <user's question>
  ```

The conversation is a **plain string** built by `format_message()` → `_render_messages()` → `build_recent_context_for_message()`.

Transcript lines use **`Name: content`** (bot lines use **`Sigrok:`**). Long fenced ` ``` ` blocks are truncated to the first 10 lines per fence.

### How many previous messages are sent per request?

**5 human turns** (`settings.genai.question.recent_messages = 5`), plus **Sigrok’s own replies** that fall within that window (chronological). Reply chains are no longer expanded separately—channel order supplies context.

### Are you manually injecting log files into the prompt?

**No.** Logs are never fed into the prompt. The `[TRANSCRIPT LEAK]` label in our logs was our own annotation — we were marking where the **model output** contained transcript-format text. The leak comes from the model **imitating** the transcript format it sees in the user prompt.

---

## 2. The Model & Parameters

### Which SLM?

**qwen2.5:3b** (switched from qwen3:4b to avoid thinking-mode prompt leakage).

### Temperature and repeat_penalty?

**Neither is set.** The Ollama request only passes:
```json
{
  "model": "ollama/qwen2.5:3b",
  "messages": [...],
  "stream": false,
  "options": {"num_predict": 1024}
}
```
Ollama uses its defaults for temperature and repeat_penalty.

---

## 3. The Discord Integration Logic

### How is the bot's own history handled?

**Bot messages are excluded.** In `build_recent_context_for_message()`:
```python
async for message in source_message.channel.history(...):
    if message.author.bot:
        continue
    ...
```
So Sigrok's replies are **not** added to the context buffer. Only human messages are included.

### Are we using a "System" role for instructions?

**Yes.** The system prompt is sent as `{"role": "system", "content": "..."}`. It contains:
- `SIGROK_PERSONALITY_SYSTEM_PROMPT` (persona, style, directives)
- `[Sigrok Q&A task]` instructions (answer the question, don't use transcript format, etc.)

The user prompt only has the conversation transcript + the question.

---

## 4. The "Leak" Origin

### Where does `ID: 1484264224434819123` actually live?

**In the user prompt we send to Ollama.** It comes from `format_message()` in `genai.py`:

```python
def format_message(self, message: Message) -> str:
    content = message.content.strip() or "[no text]"
    if not reply_id:
        return f"[ID: {message.id} | {message.author.name}]: {content}"
    return f"[ID: {message.id} | {message.author.name} replying to {reply_id}]: {content}"
```

So the transcript we send looks like:
```
[ID: 1484264224434819123 | sulack]: oh shit thank you
```

The model sees this format and sometimes **outputs** it in its response (e.g. `ID: 1484264224434819123 | sulack]: bad`). That's the leak — the model is echoing/continuing the format instead of generating plain reply text.

**Root cause:** The model is trained to continue text. Seeing `[ID: X | user]: content` in the prompt, it sometimes produces another line in the same format instead of a standalone reply.

---

## Summary of Likely Fixes

1. **Change transcript format** — Use a format the model is less likely to copy (e.g. `User: content` instead of `[ID: ... | ...]: content`).
2. **Add temperature/repeat_penalty** — Tune to reduce repetition and format mimicry.
3. **Reduce context window** — 5 messages can include long HTML; consider 3 or truncating long messages.
4. **Post-process response** — We already strip `[ID: ... | ...]:` from the start; could extend to strip it anywhere in the response.
