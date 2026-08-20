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


def test_duckduckgo_html_parser_removed() -> None:
    tools = _new_tools()
    assert not hasattr(tools, "_parse_duckduckgo_lite_results")
    assert not hasattr(tools, "_extract_duckduckgo_result_url")
    assert hasattr(tools, "_parse_searxng_results")
    assert hasattr(tools, "_request_searxng_json")


def test_parse_searxng_results_maps_and_caps() -> None:
    tools = _new_tools()
    payload = {
        "results": [
            {
                "title": "Alpha",
                "url": "https://example.com/a#frag",
                "content": "first hit",
            },
            {
                "title": "Alpha dup",
                "url": "https://example.com/a",
                "content": "duplicate url",
            },
            {
                "title": "Beta",
                "url": "https://example.com/b",
                "content": "second",
            },
            {
                "title": "Gamma",
                "url": "https://example.com/c",
                "content": "third",
            },
        ]
    }
    rows = tools._parse_searxng_results(payload, max_results=2)
    assert len(rows) == 2
    assert rows[0] == {
        "title": "Alpha",
        "url": "https://example.com/a",
        "snippet": "first hit",
    }
    assert rows[1]["title"] == "Beta"
    assert rows[1]["snippet"] == "second"


def test_searxng_origin_guard_rejects_mismatch() -> None:
    tools = _new_tools()
    saved = settings.genai.web_search.base_url
    settings.genai.web_search.base_url = "http://192.168.0.241:8080"
    try:
        tools._assert_searxng_origin(
            "http://192.168.0.241:8080/search?q=test&format=json"
        )
        try:
            tools._assert_searxng_origin(
                "http://127.0.0.1:8080/search?q=test&format=json"
            )
            raise AssertionError("expected origin mismatch")
        except ValueError as exc:
            assert "origin mismatch" in str(exc)
        try:
            tools._assert_searxng_origin(
                "http://192.168.0.241:9999/search?q=test&format=json"
            )
            raise AssertionError("expected origin mismatch")
        except ValueError as exc:
            assert "origin mismatch" in str(exc)
    finally:
        settings.genai.web_search.base_url = saved


def test_search_web_searxng_happy_path(monkeypatch) -> None:
    tools = _new_tools()
    saved_url = settings.genai.web_search.base_url
    saved_max = settings.genai.web_search.max_results
    settings.genai.web_search.base_url = "http://192.168.0.241:8080"
    settings.genai.web_search.max_results = 5

    async def fake_request(url: str) -> dict:
        assert url.startswith("http://192.168.0.241:8080/search?")
        assert "format=json" in url
        assert "q=openclaw" in url
        return {
            "results": [
                {
                    "title": "OpenClaw",
                    "url": "https://example.com/openclaw",
                    "content": "metasearch notes",
                },
                {
                    "title": "Extra",
                    "url": "https://example.com/extra",
                    "content": "more",
                },
            ]
        }

    monkeypatch.setattr(tools, "_request_searxng_json", fake_request)
    try:
        raw = asyncio.run(tools._search_web("openclaw", requested_results=1))
        payload = __import__("json").loads(raw)
        assert payload["query"] == "openclaw"
        assert len(payload["results"]) == 1
        assert payload["results"][0]["title"] == "OpenClaw"
        assert payload["results"][0]["snippet"] == "metasearch notes"
    finally:
        settings.genai.web_search.base_url = saved_url
        settings.genai.web_search.max_results = saved_max


def test_search_web_searxng_empty_and_error(monkeypatch) -> None:
    tools = _new_tools()
    saved_url = settings.genai.web_search.base_url
    settings.genai.web_search.base_url = "http://192.168.0.241:8080"

    async def empty_request(_url: str) -> dict:
        return {"results": []}

    monkeypatch.setattr(tools, "_request_searxng_json", empty_request)
    try:
        raw = asyncio.run(tools._search_web("no hits please"))
        payload = __import__("json").loads(raw)
        assert payload["query"] == "no hits please"
        assert payload["results"] == []
        assert "note" in payload
    finally:
        settings.genai.web_search.base_url = saved_url

    async def boom(_url: str) -> dict:
        raise ValueError("searxng request origin mismatch: expected 'x', got 'y'")

    monkeypatch.setattr(tools, "_request_searxng_json", boom)
    name, raw = asyncio.run(
        tools._execute_parsed_tool("search_web", {"query": "anything"})
    )
    assert name == "search_web"
    payload = __import__("json").loads(raw)
    assert payload["query"] == "anything"
    assert "error" in payload
    assert "origin mismatch" in payload["error"]


def test_dsml_tool_markup_strip_and_parse() -> None:
    tools = _new_tools()
    dsml = (
        '<｜｜DSML｜｜tool_calls>\n'
        '<｜｜DSML｜｜invoke name="search_web">\n'
        '<｜｜DSML｜｜parameter name="query" string="true">destiny lawsuit</｜｜DSML｜｜parameter>\n'
        '</｜｜DSML｜｜invoke>\n'
        '</｜｜DSML｜｜tool_calls>'
    )
    parsed = tools._parse_dsml_tool_calls(dsml)
    assert len(parsed) == 1
    assert parsed[0]["name"] == "search_web"
    assert parsed[0]["arguments"]["query"] == "destiny lawsuit"
    assert tools._strip_tool_call_blocks(dsml) == ""
    assert tools._looks_like_leaked_tool_markup(dsml)


def test_catch_up_questions_detected() -> None:
    tools = _new_tools()
    assert tools._is_catch_up_question("what's going on in here")
    assert tools._is_catch_up_question("catch me up")
    assert tools._is_catch_up_question("can you summarize the last day")
    assert tools._is_catch_up_question("whats the oldest message you can see")
    assert tools._is_catch_up_question("tldr")
    assert not tools._is_catch_up_question("is this true")
    assert not tools._is_catch_up_question("lol")


def test_reply_chain_catch_up_escalates_to_deep_window() -> None:
    tools = _new_tools()
    # A catch-up question sent as a reply (has_reference=True -> reply_chain mode)
    # must still pull the deep time-bounded window instead of the 2-turn cap.
    reply_mode, ctx_limit, history_minutes = tools._select_context_plan(
        "catch me up on what's going on",
        has_reference=True,
        retry_hint=None,
        recent_context_human_turns=None,
    )
    assert reply_mode == "reply_chain"
    assert ctx_limit == settings.genai.question.recent_messages
    assert history_minutes == settings.genai.history.minutes


def test_reply_chain_banter_stays_tight() -> None:
    tools = _new_tools()
    reply_mode, ctx_limit, history_minutes = tools._select_context_plan(
        "lol same",
        has_reference=True,
        retry_hint=None,
        recent_context_human_turns=None,
    )
    assert reply_mode == "reply_chain"
    assert ctx_limit <= 2
    assert history_minutes is None


def test_pinned_turns_disable_time_window() -> None:
    tools = _new_tools()
    # Deferred @schedule jobs pin the window explicitly; no time-bounded fetch.
    reply_mode, ctx_limit, history_minutes = tools._select_context_plan(
        "what's going on",
        has_reference=False,
        retry_hint=None,
        recent_context_human_turns=5,
    )
    assert ctx_limit == 5
    assert history_minutes is None


def test_deep_modes_get_time_window() -> None:
    tools = _new_tools()
    _, _, history_minutes = tools._select_context_plan(
        "what do you all think about the new caste discrimination ruling and its effects",
        has_reference=False,
        retry_hint=None,
        recent_context_human_turns=None,
    )
    assert history_minutes == settings.genai.history.minutes


def test_system_prompt_prefix_is_cache_stable() -> None:
    tools = _new_tools()
    # The static prefix (personality + platform + tool-use) must be byte-identical
    # across requests; only the dynamic datetime/reply-mode tail may differ.
    a = tools._build_system_prompt("discord", tools_enabled=True, reply_mode="discussion")
    b = tools._build_system_prompt("discord", tools_enabled=True, reply_mode="reply_chain")
    prefix = tools._static_system_prefix("discord")
    assert a.startswith(prefix)
    assert b.startswith(prefix)
    assert "[Tool Use]" in a.split("[Current date and time]")[0]


def test_datetime_prompt_has_no_subminute_churn() -> None:
    tools = _new_tools()
    block = tools._current_datetime_system_prompt()
    # Minute precision: seconds component of local_datetime must be :00.
    for line in block.splitlines():
        if line.strip().startswith("- local_datetime:"):
            iso = line.split("local_datetime:", 1)[1].strip()
            assert ":00" in iso[11:19], iso


def test_log_completion_usage_handles_cache_fields() -> None:
    from loguru import logger

    tools = _new_tools()
    captured: list[str] = []
    sink_id = logger.add(captured.append, level="INFO", format="{message}")
    try:
        usage = SimpleNamespace(
            prompt_tokens=1000,
            completion_tokens=50,
            total_tokens=1050,
            prompt_cache_hit_tokens=900,
            prompt_cache_miss_tokens=100,
            model_extra=None,
        )
        tools._log_completion_usage("llamacpp_chat", 0, SimpleNamespace(usage=usage))
        joined = " ".join(captured)
        assert "prompt_cache_hit_tokens=900" in joined
        assert "prompt_cache_miss_tokens=100" in joined

        # Missing usage must be a no-op, not a crash.
        captured.clear()
        tools._log_completion_usage("llamacpp_chat", 0, SimpleNamespace(usage=None))
        assert "completion_usage" not in " ".join(captured)
    finally:
        logger.remove(sink_id)


def test_opencode_go_client_requires_api_key() -> None:
    import pytest

    saved = settings.tokens.opencode_go
    settings.tokens.opencode_go = ""
    try:
        with pytest.raises(ValueError, match="opencode_go"):
            genai.GenAIOpenCodeGo(settings)
    finally:
        settings.tokens.opencode_go = saved


def test_opencode_go_uses_hosted_openai_defaults() -> None:
    saved = settings.tokens.opencode_go
    settings.tokens.opencode_go = "test-key"
    try:
        client = genai.GenAIOpenCodeGo(settings)
        assert client._openai_chat_extra_body(use_tools=True) == {
            "thinking": {"type": "disabled"}
        }
        assert client._openai_reasoning_effort() is None
        assert client._use_inline_tool_call_fallback() is True
        assert str(client._openai_local.base_url).rstrip("/").endswith("/v1")

        token = genai._max_thinking.set(True)
        try:
            assert client._openai_chat_extra_body(use_tools=True) == {
                "thinking": {"type": "enabled"}
            }
            assert client._openai_reasoning_effort() == "max"
            assert client._openai_completion_max_tokens() >= settings.bot.supersigrok.output_max
        finally:
            genai._max_thinking.reset(token)
    finally:
        settings.tokens.opencode_go = saved


def test_supersigrok_entitlement_user_and_role() -> None:
    from datetime import datetime, timedelta, timezone

    from sigrok import supersigrok

    saved_users = list(settings.bot.supersigrok.user_ids)
    saved_roles = list(settings.bot.supersigrok.role_ids)
    saved_everyone = settings.bot.supersigrok.everyone_until
    supersigrok._runtime_user_ids.clear()
    supersigrok._logged_everyone_until = None
    try:
        settings.bot.supersigrok.everyone_until = None
        settings.bot.supersigrok.user_ids = [111]
        settings.bot.supersigrok.role_ids = [222]
        assert supersigrok.is_supersigrok(user_id=111) is True
        assert supersigrok.is_supersigrok(user_id=999) is False
        assert supersigrok.is_supersigrok(user_id=999, role_ids=[222]) is True
        assert supersigrok.is_supersigrok(user_id=999, role_ids=[333]) is False

        supersigrok.grant_user(444)
        assert supersigrok.is_supersigrok(user_id=444) is True
        supersigrok.revoke_user(444)
        assert supersigrok.is_supersigrok(user_id=444) is False

        settings.bot.supersigrok.everyone_until = datetime.now(timezone.utc) + timedelta(
            hours=1
        )
        assert supersigrok.everyone_max_thinking_active() is True
        assert supersigrok.is_supersigrok(user_id=999) is True

        settings.bot.supersigrok.everyone_until = datetime.now(timezone.utc) - timedelta(
            minutes=1
        )
        assert supersigrok.everyone_max_thinking_active() is False
        assert supersigrok.is_supersigrok(user_id=999) is False
    finally:
        settings.bot.supersigrok.user_ids = saved_users
        settings.bot.supersigrok.role_ids = saved_roles
        settings.bot.supersigrok.everyone_until = saved_everyone
        supersigrok._runtime_user_ids.clear()
        supersigrok._logged_everyone_until = None
