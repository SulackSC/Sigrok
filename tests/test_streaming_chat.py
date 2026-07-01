from __future__ import annotations

import time

from sigrok.genai import GenAIBase
from sigrok.streaming.buffer import RecentMessageBuffer
from sigrok.streaming.mentions import message_mentions_bot, strip_bot_mention
from sigrok.streaming.messages import StreamingChatMessage
from sigrok.streaming.ratelimit import StreamingRateLimiter
from sigrok.streaming.response import (
    normalize_bot_response,
    should_skip_response,
    truncate_for_platform,
)


def test_message_mentions_bot_case_insensitive() -> None:
    assert message_mentions_bot("@Sigrok what is this", "sigrok")
    assert message_mentions_bot("hey @sigrok", "sigrok")
    assert not message_mentions_bot("sigrok hello", "sigrok")


def test_strip_bot_mention() -> None:
    assert strip_bot_mention("@Sigrok what game is this", "sigrok") == "what game is this"
    assert strip_bot_mention("  @sigrok   hi  ", "sigrok") == "hi"


def test_buffer_ring_eviction() -> None:
    buffer = RecentMessageBuffer(2)
    for idx in range(3):
        buffer.append(
            StreamingChatMessage(
                platform="twitch",
                channel_key="chan",
                message_id=str(idx),
                author_id="1",
                author_name="alice",
                author_display_name="Alice",
                content=f"msg {idx}",
                created_at=StreamingChatMessage.now_iso(),
            )
        )
    history = buffer.history("twitch", "chan")
    assert len(history) == 2
    assert history[0].message_id == "1"
    assert history[1].message_id == "2"


def test_to_genai_message_shape() -> None:
    message = StreamingChatMessage(
        platform="youtube",
        channel_key="UC123",
        message_id="abc",
        author_id="user1",
        author_name="viewer",
        author_display_name="Viewer",
        content="hello @Sigrok",
        created_at="2026-01-01T00:00:00Z",
        reply_to_message_id="parent",
    )
    payload = message.to_genai_message()
    assert payload["id"] == "abc"
    assert payload["author_handle"] == "viewer"
    assert payload["reply_to_message_id"] == "parent"
    assert payload["content"] == "hello @Sigrok"


def test_rate_limiter_global_and_per_user() -> None:
    limiter = StreamingRateLimiter(
        global_cooldown_seconds=10.0,
        per_user_cooldown_seconds=5.0,
    )
    assert limiter.allows_reply("twitch", "chan", "u1")
    limiter.record_reply("twitch", "chan", "u1")
    assert not limiter.allows_reply("twitch", "chan", "u1")
    assert not limiter.allows_reply("twitch", "chan", "u2")
    limiter._last_global_reply = time.monotonic() - 11.0
    assert limiter.allows_reply("twitch", "chan", "u2")


def test_truncate_for_platform() -> None:
    assert truncate_for_platform("hello", 10) == "hello"
    assert truncate_for_platform("abcdefghij", 8) == "abcde..."


def test_normalize_and_skip_response() -> None:
    assert normalize_bot_response('"hello"', bot_handle="sigrok") == "hello"
    assert should_skip_response("hi", "")
    assert not should_skip_response("hi", "still here, what's up")


def test_streaming_platform_prompts() -> None:
    twitch = GenAIBase._platform_context_system_prompt("twitch")
    youtube = GenAIBase._platform_context_system_prompt("youtube")
    kick = GenAIBase._platform_context_system_prompt("kick")
    assert "Twitch live chat" in twitch
    assert "YouTube live chat" in youtube
    assert "Kick live chat" in kick
    assert GenAIBase._is_streaming_platform("twitch")
    assert not GenAIBase._is_streaming_platform("discord")
