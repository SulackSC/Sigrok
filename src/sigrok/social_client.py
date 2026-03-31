from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import re
from typing import Any, Optional

import aiohttp

from sigrok.config import BlueskySettings, XSettings

_REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=30)


def _truncate_text(text: str, *, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3] + "..."


def _now_iso_z() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def _normalize_post_text(text: str, *, max_chars: int) -> str:
    return _truncate_text(text.replace("\r\n", "\n").strip(), max_chars=max_chars)


_ANGLE_LINK_RE = re.compile(r"<(https?://[^>\s]+)>", flags=re.IGNORECASE)
_URL_RE = re.compile(r"https?://[^\s]+", flags=re.IGNORECASE)


def _strip_link_markup(text: str) -> str:
    return _ANGLE_LINK_RE.sub(r"\1", text)


def _trim_to_boundary(text: str, *, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    limit = max_chars - 3
    clipped = text[:limit]
    boundary = max(clipped.rfind("\n"), clipped.rfind(" "))
    if boundary >= max(0, limit - 40):
        clipped = clipped[:boundary]
    return clipped.rstrip(" ,;:\n\t") + "..."


def _normalize_bluesky_text(text: str, *, max_chars: int) -> str:
    text = _strip_link_markup(text.replace("\r\n", "\n").strip())
    split_match = re.search(r"\n\s*sources:\n", text, flags=re.IGNORECASE)
    if split_match is not None:
        text = text[: split_match.start()].strip()

    if len(text) <= max_chars:
        return text

    return _trim_to_boundary(text, max_chars=max_chars)


def _build_link_facets(text: str) -> list[dict[str, Any]]:
    facets: list[dict[str, Any]] = []
    for match in _URL_RE.finditer(text):
        raw_url = match.group(0)
        url = raw_url.rstrip(".,;!?")
        end_offset = len(url)
        if url.endswith(")") and "(" not in url:
            url = url[:-1]
            end_offset = len(url)
        if not url:
            continue
        start = match.start()
        end = start + end_offset
        facets.append(
            {
                "$type": "app.bsky.richtext.facet",
                "index": {
                    "byteStart": len(text[:start].encode("utf-8")),
                    "byteEnd": len(text[:end].encode("utf-8")),
                },
                "features": [
                    {
                        "$type": "app.bsky.richtext.facet#link",
                        "uri": url,
                    }
                ],
            }
        )
    return facets


class SocialHttpError(RuntimeError):
    pass


async def _get_json(
    url: str,
    *,
    params: Optional[dict[str, Any]] = None,
    headers: Optional[dict[str, str]] = None,
    error_prefix: str,
) -> dict[str, Any]:
    async with aiohttp.ClientSession(timeout=_REQUEST_TIMEOUT) as session:
        async with session.get(url, params=params, headers=headers) as resp:
            if resp.status >= 400:
                body = await resp.text()
                raise SocialHttpError(f"{error_prefix}: {resp.status} {body[:500]}")
            return await resp.json()


async def _post_json(
    url: str,
    *,
    payload: dict[str, Any],
    headers: Optional[dict[str, str]] = None,
    error_prefix: str,
) -> dict[str, Any]:
    async with aiohttp.ClientSession(timeout=_REQUEST_TIMEOUT) as session:
        async with session.post(url, json=payload, headers=headers) as resp:
            if resp.status >= 400:
                body = await resp.text()
                raise SocialHttpError(f"{error_prefix}: {resp.status} {body[:500]}")
            content_type = (resp.headers.get("Content-Type") or "").lower()
            if "application/json" in content_type:
                return await resp.json()
            body = await resp.text()
            return {} if not body.strip() else {"raw_body": body}


def _parse_profile_view(profile: dict[str, Any]) -> tuple[str, str, str]:
    did = str(profile.get("did") or "")
    handle = str(profile.get("handle") or "")
    display_name = str(profile.get("displayName") or handle or did)
    return did, handle, display_name


@dataclass
class _BlueskySession:
    access_jwt: str
    did: str
    handle: str


@dataclass
class BlueskyPost:
    uri: str
    cid: str
    author_did: str
    author_handle: str
    author_display_name: str
    text: str
    created_at: str
    reply_to_uri: Optional[str] = None

    def to_genai_message(self) -> dict[str, Any]:
        return {
            "id": self.uri,
            "author_id": self.author_did,
            "author_name": self.author_handle,
            "author_display_name": self.author_display_name,
            "author_handle": self.author_handle,
            "author_is_bot": False,
            "created_at": self.created_at,
            "reply_to_message_id": self.reply_to_uri,
            "content": self.text.strip() or "[no text]",
            "attachments": [],
        }


@dataclass
class BlueskyNotification:
    uri: str
    cid: str
    reason: str
    indexed_at: str
    author_did: str
    author_handle: str
    author_display_name: str


def _parse_post_view(post_view: dict[str, Any]) -> Optional[BlueskyPost]:
    uri = str(post_view.get("uri") or "")
    cid = str(post_view.get("cid") or "")
    record = post_view.get("record") or {}
    author = post_view.get("author") or {}
    author_did, author_handle, author_display_name = _parse_profile_view(author)
    text = str(record.get("text") or "")
    created_at = str(record.get("createdAt") or post_view.get("indexedAt") or "")
    reply = record.get("reply") or {}
    parent = reply.get("parent") or {}
    reply_to_uri = str(parent.get("uri") or "") or None
    if not (uri and cid and author_did and author_handle):
        return None
    return BlueskyPost(
        uri=uri,
        cid=cid,
        author_did=author_did,
        author_handle=author_handle,
        author_display_name=author_display_name,
        text=text,
        created_at=created_at,
        reply_to_uri=reply_to_uri,
    )


def _flatten_parent_chain(thread_node: dict[str, Any]) -> list[BlueskyPost]:
    node_type = str(thread_node.get("$type") or "")
    if "threadViewPost" not in node_type:
        return []

    messages: list[BlueskyPost] = []
    parent = thread_node.get("parent")
    if isinstance(parent, dict):
        messages.extend(_flatten_parent_chain(parent))

    post = _parse_post_view(thread_node.get("post") or {})
    if post is not None:
        messages.append(post)
    return messages


class BlueskyClient:
    def __init__(self, settings: BlueskySettings):
        self.settings = settings
        self._session: Optional[_BlueskySession] = None

    async def _ensure_session(self) -> _BlueskySession:
        if self._session is not None:
            return self._session

        if not self.settings.identifier or not self.settings.password:
            raise SocialHttpError("Bluesky is not configured (identifier/password missing).")

        url = f"{self.settings.api_base_url.rstrip('/')}/xrpc/com.atproto.server.createSession"
        payload = {"identifier": self.settings.identifier, "password": self.settings.password}
        data = await _post_json(
            url,
            payload=payload,
            error_prefix="Bluesky createSession failed",
        )

        self._session = _BlueskySession(
            access_jwt=data["accessJwt"],
            did=data["did"],
            handle=str(data.get("handle") or self.settings.identifier),
        )
        return self._session

    async def get_own_handle(self) -> str:
        return (await self._ensure_session()).handle

    async def list_notifications(self, *, limit: int = 50) -> list[BlueskyNotification]:
        sess = await self._ensure_session()
        url = (
            f"{self.settings.api_base_url.rstrip('/')}"
            "/xrpc/app.bsky.notification.listNotifications"
        )
        data = await _get_json(
            url,
            params={"limit": max(1, min(limit, 100))},
            headers={"Authorization": f"Bearer {sess.access_jwt}"},
            error_prefix="Bluesky listNotifications failed",
        )
        notifications: list[BlueskyNotification] = []
        for raw in data.get("notifications") or []:
            author_did, author_handle, author_display_name = _parse_profile_view(
                raw.get("author") or {}
            )
            uri = str(raw.get("uri") or "")
            cid = str(raw.get("cid") or "")
            reason = str(raw.get("reason") or "")
            indexed_at = str(raw.get("indexedAt") or "")
            if not (uri and cid and reason and indexed_at and author_did and author_handle):
                continue
            notifications.append(
                BlueskyNotification(
                    uri=uri,
                    cid=cid,
                    reason=reason,
                    indexed_at=indexed_at,
                    author_did=author_did,
                    author_handle=author_handle,
                    author_display_name=author_display_name,
                )
            )
        return notifications

    async def get_post_thread(
        self,
        uri: str,
        *,
        parent_height: Optional[int] = None,
        depth: int = 0,
    ) -> list[BlueskyPost]:
        sess = await self._ensure_session()
        url = (
            f"{self.settings.api_base_url.rstrip('/')}"
            "/xrpc/app.bsky.feed.getPostThread"
        )
        data = await _get_json(
            url,
            params={
                "uri": uri,
                "depth": max(0, depth),
                "parentHeight": max(
                    0, parent_height or self.settings.thread_parent_height
                ),
            },
            headers={"Authorization": f"Bearer {sess.access_jwt}"},
            error_prefix="Bluesky getPostThread failed",
        )
        thread = data.get("thread") or {}
        return _flatten_parent_chain(thread)

    async def mark_notifications_seen(self) -> None:
        sess = await self._ensure_session()
        url = (
            f"{self.settings.api_base_url.rstrip('/')}"
            "/xrpc/app.bsky.notification.updateSeen"
        )
        await _post_json(
            url,
            payload={"seenAt": _now_iso_z()},
            headers={"Authorization": f"Bearer {sess.access_jwt}"},
            error_prefix="Bluesky updateSeen failed",
        )

    def _build_post_record(
        self,
        text: str,
        *,
        reply: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        normalized_text = _normalize_bluesky_text(
            text, max_chars=int(self.settings.max_chars)
        )
        record: dict[str, Any] = {
            "$type": "app.bsky.feed.post",
            "text": normalized_text,
            "createdAt": _now_iso_z(),
        }
        facets = _build_link_facets(normalized_text)
        if facets:
            record["facets"] = facets
        if reply is not None:
            record["reply"] = reply
        return record

    async def post_text(self, text: str) -> str:
        settings = self.settings

        sess = await self._ensure_session()
        url = f"{settings.api_base_url.rstrip('/')}/xrpc/com.atproto.repo.createRecord"
        payload = {
            "repo": sess.did,
            "collection": "app.bsky.feed.post",
            "record": self._build_post_record(text),
        }
        headers = {"Authorization": f"Bearer {sess.access_jwt}"}
        data = await _post_json(
            url,
            payload=payload,
            headers=headers,
            error_prefix="Bluesky post failed",
        )

        uri = str(data.get("uri") or "")
        if not uri:
            raise SocialHttpError("Bluesky post succeeded but no URI was returned.")
        return uri

    async def reply_to_post(
        self,
        text: str,
        *,
        parent: BlueskyPost,
        root: Optional[BlueskyPost] = None,
    ) -> str:
        settings = self.settings

        sess = await self._ensure_session()
        url = f"{settings.api_base_url.rstrip('/')}/xrpc/com.atproto.repo.createRecord"
        root_post = root or parent
        payload = {
            "repo": sess.did,
            "collection": "app.bsky.feed.post",
            "record": self._build_post_record(
                text,
                reply={
                    "root": {"uri": root_post.uri, "cid": root_post.cid},
                    "parent": {"uri": parent.uri, "cid": parent.cid},
                },
            ),
        }
        headers = {"Authorization": f"Bearer {sess.access_jwt}"}
        data = await _post_json(
            url,
            payload=payload,
            headers=headers,
            error_prefix="Bluesky reply failed",
        )

        uri = str(data.get("uri") or "")
        if not uri:
            raise SocialHttpError("Bluesky reply succeeded but no URI was returned.")
        return uri


class XClient:
    def __init__(self, settings: XSettings):
        self.settings = settings

    async def post_text(self, text: str) -> str:
        text = _normalize_post_text(text, max_chars=int(self.settings.max_chars))

        if not self.settings.bearer_token:
            raise SocialHttpError("X is not configured (bearer_token missing).")

        url = f"{self.settings.api_base_url.rstrip('/')}/2/tweets"
        headers = {
            "Authorization": f"Bearer {self.settings.bearer_token}",
            "Content-Type": "application/json",
        }
        payload = {"text": text}
        data = await _post_json(
            url,
            payload=payload,
            headers=headers,
            error_prefix="X tweet failed",
        )

        tweet_id = str((data.get("data") or {}).get("id") or "")
        if not tweet_id:
            raise SocialHttpError("X post succeeded but no tweet id was returned.")

        return f"https://x.com/i/status/{tweet_id}"

