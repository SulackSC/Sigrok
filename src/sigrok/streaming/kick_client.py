from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import Awaitable, Callable
from typing import Any, Optional

import aiohttp
from aiohttp import web
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from loguru import logger

from sigrok.config import KickStreamingSettings, KickTokenSettings, StreamingSettings
from sigrok.streaming.messages import StreamingChatMessage

OnKickMessage = Callable[[StreamingChatMessage, dict[str, Any]], Awaitable[None]]

_KICK_API_BASE = "https://api.kick.com/public/v1"
_KICK_OAUTH_BASE = "https://id.kick.com"
_PUSHER_APP_KEY = "32cbd69e4b950bf97679"
_PUSHER_CLUSTER = "us2"


class KickChatAdapter:
    def __init__(
        self,
        *,
        streaming: StreamingSettings,
        on_message: OnKickMessage,
    ) -> None:
        self._settings = streaming.kick
        self._token_settings: KickTokenSettings = KickTokenSettings()
        self._on_message = on_message
        self._tasks: list[asyncio.Task[None]] = []
        self._stop = asyncio.Event()
        self._webhook_runner: Optional[web.AppRunner] = None
        self._channel_meta: dict[str, dict[str, Any]] = {}
        self._chatroom_to_slug: dict[str, str] = {}
        self._webhook_public_key_pem: Optional[str] = None

    def configure_tokens(self, tokens: KickTokenSettings) -> None:
        self._token_settings = tokens

    @property
    def bot_username(self) -> str:
        return self._settings.bot_username

    @property
    def max_chars(self) -> int:
        return self._settings.max_chars

    async def start(self, *, tokens: KickTokenSettings) -> None:
        if not self._settings.enabled:
            return
        self._token_settings = tokens
        if self._settings.receive_mode == "webhook":
            await self._start_webhook()
        else:
            for channel in self._settings.channels:
                slug = channel.strip().lower()
                if not slug:
                    continue
                task = asyncio.create_task(self._listen_websocket(slug))
                self._tasks.append(task)
        logger.info("Kick chat adapter started (mode={}).", self._settings.receive_mode)

    async def stop(self) -> None:
        self._stop.set()
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        if self._webhook_runner is not None:
            await self._webhook_runner.cleanup()
            self._webhook_runner = None
        self._stop.clear()

    async def _fetch_chatroom_id(self, slug: str) -> Optional[int]:
        url = f"https://kick.com/api/v2/channels/{slug}"
        headers = {
            "Accept": "application/json",
            "User-Agent": "Sigrok/1.0",
        }
        timeout = aiohttp.ClientTimeout(total=20)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, headers=headers) as resp:
                if resp.status != 200:
                    logger.warning(f"Kick channel lookup failed for {slug}: HTTP {resp.status}")
                    return None
                data = await resp.json()
        chatroom = data.get("chatroom") or {}
        chatroom_id = chatroom.get("id")
        if chatroom_id is None:
            return None
        self._channel_meta[slug] = {
            "chatroom_id": str(chatroom_id),
            "channel_id": str(data.get("user_id") or data.get("id") or slug),
            "slug": slug,
        }
        self._chatroom_to_slug[str(chatroom_id)] = slug
        return int(chatroom_id)

    async def _listen_websocket(self, slug: str) -> None:
        while not self._stop.is_set():
            try:
                chatroom_id = await self._fetch_chatroom_id(slug)
                if chatroom_id is None:
                    await asyncio.sleep(20)
                    continue

                ws_url = (
                    f"wss://ws-{_PUSHER_CLUSTER}.pusher.com/app/{_PUSHER_APP_KEY}"
                    "?protocol=7&client=js&version=8.4.0-rc2&flash=false"
                )
                timeout = aiohttp.ClientTimeout(total=None, sock_read=120)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.ws_connect(ws_url, heartbeat=30) as ws:
                        subscribe = {
                            "event": "pusher:subscribe",
                            "data": {
                                "auth": "",
                                "channel": f"chatrooms.{chatroom_id}.v2",
                            },
                        }
                        await ws.send_str(json.dumps(subscribe))
                        logger.info(f"Kick websocket connected for {slug} (chatroom {chatroom_id})")
                        async for msg in ws:
                            if self._stop.is_set():
                                break
                            if msg.type != aiohttp.WSMsgType.TEXT:
                                continue
                            await self._handle_pusher_message(slug, msg.data)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(f"Kick websocket error for {slug}: {exc}")
                await asyncio.sleep(10)

    async def _handle_pusher_message(self, slug: str, raw: str) -> None:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return
        event = str(data.get("event") or "")
        if "ChatMessageEvent" not in event:
            return
        channel = str(data.get("channel") or "")
        parts = channel.split(".")
        chatroom_id = parts[1] if len(parts) > 1 else ""
        try:
            chat_message = json.loads(data.get("data") or "{}")
        except json.JSONDecodeError:
            return
        sender = chat_message.get("sender") or {}
        username = str(sender.get("username") or "unknown")
        content = str(chat_message.get("content") or "")
        message_id = str(chat_message.get("id") or chat_message.get("message_id") or "")
        meta = self._channel_meta.get(slug, {})
        message = StreamingChatMessage(
            platform="kick",
            channel_key=slug,
            message_id=message_id or f"{chatroom_id}:{username}:{len(content)}",
            author_id=str(sender.get("id") or username),
            author_name=username,
            author_display_name=username,
            content=content,
            created_at=str(chat_message.get("created_at") or StreamingChatMessage.now_iso()),
        )
        payload = {
            "slug": slug,
            "chatroom_id": chatroom_id,
            "channel_id": meta.get("channel_id"),
            "raw": chat_message,
        }
        await self._on_message(message, payload)

    async def _load_webhook_public_key(self) -> str:
        if self._settings.webhook_public_key.strip():
            return self._settings.webhook_public_key.strip()
        timeout = aiohttp.ClientTimeout(total=20)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get("https://api.kick.com/public/v1/public-key") as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    raise RuntimeError(
                        f"Kick public key fetch failed: HTTP {resp.status} {body[:200]}"
                    )
                data = await resp.json()
        public_key = str((data.get("data") or {}).get("public_key") or data.get("public_key") or "")
        if not public_key.strip():
            raise RuntimeError("Kick public key response was empty.")
        return public_key.strip()

    def _verify_webhook_signature(
        self,
        *,
        message_id: str,
        timestamp: str,
        body: bytes,
        signature_b64: str,
    ) -> bool:
        if not self._webhook_public_key_pem:
            return False
        public_key = serialization.load_pem_public_key(
            self._webhook_public_key_pem.encode("utf-8")
        )
        signed_payload = f"{message_id}.{timestamp}.".encode("utf-8") + body
        signature = base64.b64decode(signature_b64, validate=True)
        public_key.verify(  # type: ignore[union-attr]
            signature,
            signed_payload,
            padding.PKCS1v15(),
            hashes.SHA256(),
        )
        return True

    async def _start_webhook(self) -> None:
        try:
            self._webhook_public_key_pem = await self._load_webhook_public_key()
        except Exception as exc:
            raise RuntimeError(
                "Kick webhook mode requires a valid public key; "
                f"set streaming.kick.webhook_public_key or fix API access: {exc}"
            ) from exc

        app = web.Application()
        path = self._settings.webhook_path.rstrip("/") or "/kick/eventsub"
        app.router.add_post(path, self._webhook_handler)
        self._webhook_runner = web.AppRunner(app)
        await self._webhook_runner.setup()
        site = web.TCPSite(
            self._webhook_runner,
            self._settings.webhook_host,
            self._settings.webhook_port,
        )
        await site.start()
        logger.info(
            "Kick webhook listening on http://{}:{}{}",
            self._settings.webhook_host,
            self._settings.webhook_port,
            path,
        )

    async def _webhook_handler(self, request: web.Request) -> web.Response:
        raw_body = await request.read()
        message_id = request.headers.get("Kick-Event-Message-Id", "")
        timestamp = request.headers.get("Kick-Event-Message-Timestamp", "")
        signature = request.headers.get("Kick-Event-Signature", "")
        if not (message_id and timestamp and signature):
            return web.Response(status=401, text="missing kick signature headers")
        try:
            self._verify_webhook_signature(
                message_id=message_id,
                timestamp=timestamp,
                body=raw_body,
                signature_b64=signature,
            )
        except Exception as exc:
            logger.warning(f"Kick webhook signature verification failed: {exc}")
            return web.Response(status=401, text="invalid signature")

        try:
            body = json.loads(raw_body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return web.Response(status=400, text="invalid json")
        event_type = str(body.get("type") or body.get("event") or "")
        if "chat" not in event_type.lower():
            return web.Response(status=200, text="ok")
        data = body.get("data") or body.get("payload") or body
        sender = data.get("sender") or data.get("user") or {}
        username = str(sender.get("username") or sender.get("slug") or "unknown")
        content = str(data.get("content") or data.get("message") or "")
        slug = str(data.get("channel_slug") or data.get("slug") or "")
        if not slug and self._settings.channels:
            slug = self._settings.channels[0].strip().lower()
        message_id = str(data.get("id") or data.get("message_id") or "")
        message = StreamingChatMessage(
            platform="kick",
            channel_key=slug or "unknown",
            message_id=message_id or f"webhook:{username}",
            author_id=str(sender.get("id") or username),
            author_name=username,
            author_display_name=username,
            content=content,
            created_at=StreamingChatMessage.now_iso(),
        )
        await self._on_message(message, data)
        return web.Response(status=200, text="ok")

    async def _access_token(self) -> str:
        if self._token_settings.access_token:
            return self._token_settings.access_token
        raise RuntimeError("Kick access_token missing in tokens.kick config.")

    async def send_message(self, slug: str, text: str) -> None:
        meta = self._channel_meta.get(slug.lower())
        if meta is None:
            await self._fetch_chatroom_id(slug.lower())
            meta = self._channel_meta.get(slug.lower())
        channel_id = (meta or {}).get("channel_id")
        if not channel_id:
            raise RuntimeError(f"Kick channel_id unknown for {slug}")

        token = await self._access_token()
        url = f"{_KICK_API_BASE}/chat"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload = {
            "broadcaster_user_id": int(channel_id) if str(channel_id).isdigit() else channel_id,
            "content": text,
            "type": "bot",
        }
        timeout = aiohttp.ClientTimeout(total=20)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, headers=headers, json=payload) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    raise RuntimeError(f"Kick send failed: {resp.status} {body[:300]}")
