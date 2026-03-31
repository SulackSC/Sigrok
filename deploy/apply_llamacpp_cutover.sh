#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/sulack/Documents/Sigrok"

echo "[1/4] Installing llama-server systemd unit"
sudo install -m 644 "$ROOT/deploy/systemd/llama-server.service" /etc/systemd/system/llama-server.service

echo "[2/4] Installing OpenWebUI llama.cpp override"
sudo mkdir -p /etc/systemd/system/open-webui.service.d
sudo install -m 644 "$ROOT/deploy/systemd/open-webui-llamacpp.override.conf" /etc/systemd/system/open-webui.service.d/llamacpp.conf

echo "[3/4] Reloading systemd and restarting dependencies"
sudo systemctl daemon-reload
sudo systemctl enable --now llama-server.service
sudo systemctl restart open-webui.service

echo "[4/4] Health checks"
curl -sS -H 'Authorization: Bearer sk-no-key-required' http://127.0.0.1:8081/v1/models
sudo systemctl --no-pager --full status llama-server.service
sudo systemctl --no-pager --full status open-webui.service

echo
echo "Cutover infra complete."
echo "Restart the bot when ready:"
echo "  sudo systemctl restart sigrok.service"
