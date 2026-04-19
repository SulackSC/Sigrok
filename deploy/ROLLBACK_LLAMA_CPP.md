# llama.cpp cutover + rollback

This host currently runs:
- `sigrok.service` (bot)
- `open-webui.service` (UI on `:8080`)
- llama.cpp **native** `llama-server` on `:8081` (see `/etc/systemd/system/llama-server.service`)

## 1) Install/enable llama.cpp system service

```bash
sudo install -m 644 /home/sulack/Documents/Sigrok/deploy/systemd/llama-server.service /etc/systemd/system/llama-server.service
sudo systemctl daemon-reload
sudo systemctl enable --now llama-server.service
sudo systemctl status llama-server.service --no-pager
curl -sS -H 'Authorization: Bearer sk-no-key-required' http://127.0.0.1:8081/v1/models
```

## 2) Point OpenWebUI at llama.cpp

```bash
sudo mkdir -p /etc/systemd/system/open-webui.service.d
sudo install -m 644 /home/sulack/Documents/Sigrok/deploy/systemd/open-webui-llamacpp.override.conf /etc/systemd/system/open-webui.service.d/llamacpp.conf
sudo systemctl daemon-reload
sudo systemctl restart open-webui.service
sudo systemctl status open-webui.service --no-pager
```

OpenWebUI remains at `http://127.0.0.1:8080` (or LAN IP `:8080`); only model backend changes.

## 3) Bot config (llama.cpp)

`settings.toml` should match the `-a` model alias from `llama-server.service`, e.g.:

```toml
model = "llamacpp/qwen3.6-35b-a3b"
base_url = "http://127.0.0.1:8081"
```

Restart bot when ready:

```bash
sudo systemctl restart sigrok.service
sudo systemctl status sigrok.service --no-pager
```

## Rollback (fast)

### Bot rollback to previous native llama.cpp model

1. Restore the **previous** `llama-server.service` `ExecStart` paths (Qwen3.5-27B GGUF + `-a qwen3.5-27b`), then:

```bash
sudo install -m 644 /home/sulack/Documents/Sigrok/deploy/systemd/llama-server.service /etc/systemd/system/llama-server.service
sudo systemctl daemon-reload
sudo systemctl restart llama-server.service
```

2. Edit `settings.toml` and restore:

```toml
model = "llamacpp/qwen3.5-27b"
base_url = "http://127.0.0.1:8081"
```

3. Restart the bot:

```bash
sudo systemctl restart sigrok.service
```

(If you keep a git branch or backup of `deploy/systemd/llama-server.service` from before the Qwen3.6 cutover, reinstall that file instead of the repo copy.)

### Bot rollback to Ollama (only if Ollama still runs on `:11434`)

```toml
model = "ollama/qwen3:14b"
base_url = "http://127.0.0.1:11434"
```

```bash
sudo systemctl restart sigrok.service
```

### OpenWebUI rollback to previous backend behavior

```bash
sudo rm -f /etc/systemd/system/open-webui.service.d/llamacpp.conf
sudo systemctl daemon-reload
sudo systemctl restart open-webui.service
```

### Stop llama.cpp service (optional on rollback)

```bash
sudo systemctl disable --now llama-server.service
```
