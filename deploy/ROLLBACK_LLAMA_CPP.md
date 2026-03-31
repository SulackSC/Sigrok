# llama.cpp cutover + rollback

This host currently runs:
- `sigrok.service` (bot)
- `open-webui.service` (UI on `:8080`)
- llama.cpp server on `:8081`

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

## 3) Bot config already switched to llama.cpp

`settings.toml` is set to:
- `model = "llamacpp//home/ollama/...sha256-a8cc..."`
- `base_url = "http://127.0.0.1:8081"`

Restart bot when ready:

```bash
sudo systemctl restart sigrok.service
sudo systemctl status sigrok.service --no-pager
```

## Rollback (fast)

### Bot rollback to Ollama

Edit `settings.toml` and restore:

```toml
model = "ollama/qwen3:14b"
base_url = "http://127.0.0.1:11434"
```

Then:

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
