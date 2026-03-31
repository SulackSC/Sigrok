#!/usr/bin/env bash
set -euo pipefail

echo "[1/3] Reverting OpenWebUI override"
sudo rm -f /etc/systemd/system/open-webui.service.d/llamacpp.conf

echo "[2/3] Stopping llama-server service"
sudo systemctl disable --now llama-server.service || true

echo "[3/3] Reloading and restarting OpenWebUI"
sudo systemctl daemon-reload
sudo systemctl restart open-webui.service

echo
echo "Now restore bot settings.toml to Ollama and restart bot:"
echo "  model = \"ollama/qwen3:14b\""
echo "  base_url = \"http://127.0.0.1:11434\""
echo "  sudo systemctl restart sigrok.service"
