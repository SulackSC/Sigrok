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
    session: Optional[aiohttp.ClientSession] = None,
) -> dict[str, Any]:
    owns_session = session is None
    if session is None:
        session = aiohttp.ClientSession(timeout=_REQUEST_TIMEOUT)
    try:
        async with session.get(url, params=params, headers=headers) as resp:
            if resp.status >= 400:
                body = await resp.text()
                raise SocialHttpError(f"{error_prefix}: {resp.status} {body[:500]}")
            return await resp.json()
    finally:
        if owns_session:
            await session.close()


async def _post_json(
    url: str,
    *,
    payload: dict[str, Any],
    headers: Optional[dict[str, str]] = None,
    error_prefix: str,
    session: Optional[aiohttp.ClientSession] = None,
) -> dict[str, Any]:
    owns_session = session is None
    if session is None:
        session = aiohttp.ClientSession(timeout=_REQUEST_TIMEOUT)
    try:
        async with session.post(url, json=payload, headers=headers) as resp:
            if resp.status >= 400:
                body = await resp.text()
                raise SocialHttpError(f"{error_prefix}: {resp.status} {body[:500]}")
            content_type = (resp.headers.get("Content-Type") or "").lower()
            if "application/json" in content_type:
                return await resp.json()
            body = await resp.text()
            return {} if not body.strip() else {"raw_body": body}
    finally:
        if owns_session:
            await session.close()


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
        self._http: Optional[aiohttp.ClientSession] = None

    async def _http_session(self) -> aiohttp.ClientSession:
        if self._http is None or self._http.closed:
            self._http = aiohttp.ClientSession(timeout=_REQUEST_TIMEOUT)
        return self._http

    async def close(self) -> None:
        if self._http is not None:
            await self._http.close()
            self._http = None

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
            session=await self._http_session(),
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
            session=await self._http_session(),
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
            session=await self._http_session(),
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
            session=await self._http_session(),
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
            session=await self._http_session(),
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
            session=await self._http_session(),
        )

        uri = str(data.get("uri") or "")
        if not uri:
            raise SocialHttpError("Bluesky reply succeeded but no URI was returned.")
        return uri


def _normalize_x_text(text: str, *, max_chars: int) -> str:
    text = _strip_link_markup(text.replace("\r\n", "\n").strip())
    split_match = re.search(r"\n\s*sources:\n", text, flags=re.IGNORECASE)
    if split_match is not None:
        text = text[: split_match.start()].strip()

    if len(text) <= max_chars:
        return text
    return _trim_to_boundary(text, max_chars=max_chars)


def _index_users_by_id(users: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for user in users:
        user_id = str(user.get("id") or "")
        if user_id:
            indexed[user_id] = user
    return indexed


def _index_tweets_by_id(tweets: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for tweet in tweets:
        tweet_id = str(tweet.get("id") or "")
        if tweet_id:
            indexed[tweet_id] = tweet
    return indexed


def _parse_x_user(user: dict[str, Any]) -> tuple[str, str, str]:
    user_id = str(user.get("id") or "")
    username = str(user.get("username") or "")
    display_name = str(user.get("name") or username or user_id)
    return user_id, username, display_name


def _reply_parent_tweet_id(tweet: dict[str, Any]) -> Optional[str]:
    for ref in tweet.get("referenced_tweets") or []:
        if str(ref.get("type") or "") == "replied_to":
            tweet_id = str(ref.get("id") or "")
            if tweet_id:
                return tweet_id
    return None


def _parse_x_tweet(
    tweet: dict[str, Any],
    *,
    users_by_id: dict[str, dict[str, Any]],
) -> Optional["XTweet"]:
    tweet_id = str(tweet.get("id") or "")
    author_id = str(tweet.get("author_id") or "")
    text = str(tweet.get("text") or "")
    created_at = str(tweet.get("created_at") or "")
    conversation_id = str(tweet.get("conversation_id") or tweet_id)
    in_reply_to_user_id = str(tweet.get("in_reply_to_user_id") or "") or None
    reply_to_tweet_id = _reply_parent_tweet_id(tweet)
    author = users_by_id.get(author_id) or {}
    author_user_id, author_username, author_display_name = _parse_x_user(author)
    if not (tweet_id and author_id and author_username):
        return None
    return XTweet(
        id=tweet_id,
        text=text,
        author_id=author_user_id or author_id,
        author_username=author_username,
        author_display_name=author_display_name,
        created_at=created_at,
        conversation_id=conversation_id,
        in_reply_to_user_id=in_reply_to_user_id,
        reply_to_tweet_id=reply_to_tweet_id,
    )


@dataclass
class XTweet:
    id: str
    text: str
    author_id: str
    author_username: str
    author_display_name: str
    created_at: str
    conversation_id: str
    in_reply_to_user_id: Optional[str] = None
    reply_to_tweet_id: Optional[str] = None

    def to_genai_message(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "author_id": self.author_id,
            "author_name": self.author_username,
            "author_display_name": self.author_display_name,
            "author_handle": self.author_username,
            "author_is_bot": False,
            "created_at": self.created_at,
            "reply_to_message_id": self.reply_to_tweet_id,
            "content": self.text.strip() or "[no text]",
            "attachments": [],
        }


class XClient:
    _TWEET_FIELDS = (
        "created_at,conversation_id,author_id,referenced_tweets,in_reply_to_user_id"
    )
    _TWEET_EXPANSIONS = "author_id,referenced_tweets.id"
    _USER_FIELDS = "username,name"

    def __init__(self, settings: XSettings):
        self.settings = settings
        self._http: Optional[aiohttp.ClientSession] = None
        self._resolved_user_id = settings.user_id.strip()
        self._resolved_username = settings.username.strip().lstrip("@").lower()

    async def _http_session(self) -> aiohttp.ClientSession:
        if self._http is None or self._http.closed:
            self._http = aiohttp.ClientSession(timeout=_REQUEST_TIMEOUT)
        return self._http

    async def close(self) -> None:
        if self._http is not None:
            await self._http.close()
            self._http = None

    def _auth_headers(self) -> dict[str, str]:
        if not self.settings.bearer_token:
            raise SocialHttpError("X is not configured (bearer_token missing).")
        return {
            "Authorization": f"Bearer {self.settings.bearer_token}",
            "Content-Type": "application/json",
        }

    async def _ensure_identity(self) -> tuple[str, str]:
        if self._resolved_user_id and self._resolved_username:
            return self._resolved_user_id, self._resolved_username

        url = f"{self.settings.api_base_url.rstrip('/')}/2/users/me"
        data = await _get_json(
            url,
            params={"user.fields": self._USER_FIELDS},
            headers=self._auth_headers(),
            error_prefix="X users/me failed",
            session=await self._http_session(),
        )
        user = data.get("data") or {}
        user_id, username, _display_name = _parse_x_user(user)
        if not (user_id and username):
            raise SocialHttpError("X users/me succeeded but user identity was missing.")
        self._resolved_user_id = user_id
        self._resolved_username = username.lower()
        return self._resolved_user_id, self._resolved_username

    async def get_own_username(self) -> str:
        return (await self._ensure_identity())[1]

    def _parse_tweet_response(self, data: dict[str, Any]) -> list[XTweet]:
        users_by_id = _index_users_by_id(list(data.get("includes", {}).get("users") or []))
        tweets: list[XTweet] = []
        for raw in data.get("data") or []:
            if not isinstance(raw, dict):
                continue
            tweet = _parse_x_tweet(raw, users_by_id=users_by_id)
            if tweet is not None:
                tweets.append(tweet)
        return tweets

    async def list_mentions(
        self,
        *,
        since_id: Optional[str] = None,
        max_results: int = 20,
    ) -> list[XTweet]:
        user_id, _username = await self._ensure_identity()
        params: dict[str, Any] = {
            "max_results": max(5, min(max_results, 100)),
            "tweet.fields": self._TWEET_FIELDS,
            "expansions": self._TWEET_EXPANSIONS,
            "user.fields": self._USER_FIELDS,
        }
        if since_id:
            params["since_id"] = since_id
        url = f"{self.settings.api_base_url.rstrip('/')}/2/users/{user_id}/mentions"
        data = await _get_json(
            url,
            params=params,
            headers=self._auth_headers(),
            error_prefix="X list mentions failed",
            session=await self._http_session(),
        )
        return self._parse_tweet_response(data)

    async def get_tweet(self, tweet_id: str) -> Optional[XTweet]:
        url = f"{self.settings.api_base_url.rstrip('/')}/2/tweets/{tweet_id}"
        data = await _get_json(
            url,
            params={
                "tweet.fields": self._TWEET_FIELDS,
                "expansions": self._TWEET_EXPANSIONS,
                "user.fields": self._USER_FIELDS,
            },
            headers=self._auth_headers(),
            error_prefix="X get tweet failed",
            session=await self._http_session(),
        )
        raw = data.get("data")
        if not isinstance(raw, dict):
            return None
        users_by_id = _index_users_by_id(list(data.get("includes", {}).get("users") or []))
        included_tweets = _index_tweets_by_id(
            list(data.get("includes", {}).get("tweets") or [])
        )
        users_by_id.update(
            {
                str(tweet.get("author_id") or ""): users_by_id.get(
                    str(tweet.get("author_id") or ""), {}
                )
                for tweet in included_tweets.values()
            }
        )
        return _parse_x_tweet(raw, users_by_id=users_by_id)

    async def get_reply_chain(
        self,
        tweet: XTweet,
        *,
        parent_height: Optional[int] = None,
    ) -> list[XTweet]:
        chain = [tweet]
        current = tweet
        remaining = max(0, parent_height or self.settings.thread_parent_height)
        while remaining > 0 and current.reply_to_tweet_id:
            parent = await self.get_tweet(current.reply_to_tweet_id)
            if parent is None:
                break
            chain.insert(0, parent)
            current = parent
            remaining -= 1
        return chain

    async def post_text(self, text: str) -> str:
        return await self._create_tweet(text)

    async def reply_to_tweet(self, text: str, *, in_reply_to_tweet_id: str) -> str:
        return await self._create_tweet(
            text,
            in_reply_to_tweet_id=in_reply_to_tweet_id,
        )

    async def _create_tweet(
        self,
        text: str,
        *,
        in_reply_to_tweet_id: Optional[str] = None,
    ) -> str:
        text = _normalize_x_text(text, max_chars=int(self.settings.max_chars))
        url = f"{self.settings.api_base_url.rstrip('/')}/2/tweets"
        payload: dict[str, Any] = {"text": text}
        if in_reply_to_tweet_id:
            payload["reply"] = {"in_reply_to_tweet_id": in_reply_to_tweet_id}
        data = await _post_json(
            url,
            payload=payload,
            headers=self._auth_headers(),
            error_prefix="X tweet failed",
            session=await self._http_session(),
        )

        tweet_id = str((data.get("data") or {}).get("id") or "")
        if not tweet_id:
            raise SocialHttpError("X post succeeded but no tweet id was returned.")

        return f"https://x.com/i/status/{tweet_id}"

