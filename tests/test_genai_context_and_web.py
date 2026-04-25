from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from sigrok import genai
from sigrok.config import settings


@dataclass
class FakeAuthor:
    id: int
    name: str
    display_name: str
    bot: bool = False


class FakeChannel:
    def __init__(self, history_messages: list["FakeMessage"]):
        self._history_messages = history_messages

    async def history(self, **_: object):
        for message in self._history_messages:
            yield message

    async def fetch_message(self, message_id: int):
        for message in self._history_messages:
            if message.id == message_id:
                return message
        raise LookupError(message_id)


class FakeMessage:
    def __init__(
        self,
        *,
        mid: int,
        author: FakeAuthor,
        content: str,
        created_at: datetime,
        channel: FakeChannel,
        guild: object,
        reference: object | None = None,
    ):
        self.id = mid
        self.author = author
        self.content = content
        self.created_at = created_at
        self.channel = channel
        self.guild = guild
        self.reference = reference
        self.attachments: list[object] = []
        self.embeds: list[object] = []


class DummyLocalTools(genai._GenAILocalWithWebTools):
    pass


def _new_tools() -> DummyLocalTools:
    return DummyLocalTools(settings)


def test_plain_context_format_includes_reply_parent_label() -> None:
    tools = _new_tools()
    now = datetime.now(timezone.utc)
    guild = SimpleNamespace(me=SimpleNamespace(id=777))
    channel = FakeChannel([])
    parent = FakeMessage(
        mid=1,
        author=FakeAuthor(id=11, name="alice", display_name="Alice"),
        content="hello there",
        created_at=now - timedelta(minutes=2),
        channel=channel,
        guild=guild,
    )
    child = FakeMessage(
        mid=2,
        author=FakeAuthor(id=22, name="bob", display_name="Bobby"),
        content="replying now",
        created_at=now - timedelta(minutes=1),
        channel=channel,
        guild=guild,
        reference=SimpleNamespace(message_id=1, resolved=parent),
    )
    lines = tools._render_plain_context_messages([parent, child])
    assert "Alice: hello there" in lines
    assert "Bobby (re:Alice): replying now" in lines


def test_collect_recent_context_dedupes_reply_chain_ids() -> None:
    tools = _new_tools()
    now = datetime.now(timezone.utc)
    guild = SimpleNamespace(me=SimpleNamespace(id=777))
    channel = FakeChannel([])
    parent = FakeMessage(
        mid=10,
        author=FakeAuthor(id=11, name="alice", display_name="Alice"),
        content="parent",
        created_at=now - timedelta(minutes=5),
        channel=channel,
        guild=guild,
    )
    peer = FakeMessage(
        mid=12,
        author=FakeAuthor(id=44, name="dave", display_name="Dave"),
        content="other context",
        created_at=now - timedelta(minutes=4),
        channel=channel,
        guild=guild,
    )
    source = FakeMessage(
        mid=15,
        author=FakeAuthor(id=22, name="bob", display_name="Bobby"),
        content="child",
        created_at=now - timedelta(minutes=1),
        channel=channel,
        guild=guild,
        reference=SimpleNamespace(message_id=10, resolved=parent),
    )
    channel._history_messages = [source, peer, parent]
    merged = asyncio.run(tools._collect_recent_context_messages(source, limit=2))
    ids = [m.id for m in merged]
    assert ids.count(10) == 1
    assert ids.count(12) == 1


def test_collect_recent_context_respects_merge_reply_chain_flag() -> None:
    tools = _new_tools()
    now = datetime.now(timezone.utc)
    guild = SimpleNamespace(me=SimpleNamespace(id=777))
    channel = FakeChannel([])
    parent = FakeMessage(
        mid=21,
        author=FakeAuthor(id=11, name="alice", display_name="Alice"),
        content="only in reference chain",
        created_at=now - timedelta(minutes=8),
        channel=channel,
        guild=guild,
    )
    recent = FakeMessage(
        mid=22,
        author=FakeAuthor(id=44, name="dave", display_name="Dave"),
        content="recent message",
        created_at=now - timedelta(minutes=4),
        channel=channel,
        guild=guild,
    )
    source = FakeMessage(
        mid=23,
        author=FakeAuthor(id=55, name="eve", display_name="Eve"),
        content="reply turn",
        created_at=now - timedelta(minutes=1),
        channel=channel,
        guild=guild,
        reference=SimpleNamespace(message_id=21, resolved=parent),
    )
    channel._history_messages = [source, recent]
    with_chain = asyncio.run(
        tools._collect_recent_context_messages(source, limit=1, merge_reply_chain=True)
    )
    without_chain = asyncio.run(
        tools._collect_recent_context_messages(source, limit=1, merge_reply_chain=False)
    )
    assert any(msg.id == 21 for msg in with_chain)
    assert all(msg.id != 21 for msg in without_chain)


def test_url_safety_helpers_block_internal_and_dedupe_sources() -> None:
    tools = _new_tools()
    assert tools._normalize_public_url("https://example.com/path#frag") == "https://example.com/path"
    assert tools._normalize_public_url("javascript:alert(1)") == ""
    assert tools._is_blocked_ip("127.0.0.1") is True
    assert tools._is_blocked_ip("10.0.0.5") is True
    deduped = tools._dedupe_sources(
        [
            {"title": "One", "url": "https://example.com/a#x", "snippet": "a"},
            {"title": "Duplicate", "url": "https://example.com/a", "snippet": "b"},
            {"title": "Two", "url": "https://example.com/b", "snippet": ""},
        ]
    )
    assert [row["url"] for row in deduped] == [
        "https://example.com/a",
        "https://example.com/b",
    ]
