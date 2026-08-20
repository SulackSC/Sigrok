from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from sigrok.channel_rules import (
    clear_channel_rules_cache,
    extract_channel_rule_body,
    fetch_channel_rules,
    is_channel_rule_pin,
)
from sigrok.genai import GenAIBase


class _FakeBot:
    id = 123456789012345678
    name = "Sigrok"
    display_name = "Sigrok"
    global_name = "Sigrok"


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    clear_channel_rules_cache()
    yield
    clear_channel_rules_cache()


def test_extract_body_with_snowflake_mentions() -> None:
    bot = _FakeBot()
    content = f"<@{bot.id}>\nno spoilers\nkeep it short\n<@{bot.id}>"
    assert extract_channel_rule_body(content, bot) == "no spoilers\nkeep it short"
    assert is_channel_rule_pin(content, bot)


def test_extract_body_with_at_name() -> None:
    bot = _FakeBot()
    content = "@Sigrok\nno spoilers for the current season\n@Sigrok"
    assert extract_channel_rule_body(content, bot) == "no spoilers for the current season"
    assert is_channel_rule_pin(content, bot)


def test_extract_body_case_insensitive_name() -> None:
    bot = _FakeBot()
    content = "@sigrok\nbe brief\n@SIGROK"
    assert extract_channel_rule_body(content, bot) == "be brief"


def test_missing_start_or_end_not_a_rule() -> None:
    bot = _FakeBot()
    assert extract_channel_rule_body("no spoilers\n@Sigrok", bot) == ""
    assert extract_channel_rule_body("@Sigrok\nno spoilers", bot) == ""
    assert not is_channel_rule_pin("just a normal pin", bot)


def test_empty_body_ignored() -> None:
    bot = _FakeBot()
    content = f"<@{bot.id}>\n\n<@{bot.id}>"
    assert extract_channel_rule_body(content, bot) == ""
    assert not is_channel_rule_pin(content, bot)


def test_nickname_mention_form() -> None:
    bot = _FakeBot()
    content = f"<@!{bot.id}>\nrule one\n<@!{bot.id}>"
    assert extract_channel_rule_body(content, bot) == "rule one"


@pytest.mark.asyncio
async def test_fetch_concatenates_multiple_pins_oldest_first() -> None:
    bot = _FakeBot()
    # Discord returns newest first; fetch should reverse to oldest first
    pins = [
        SimpleNamespace(content="@Sigrok\nnewer rule\n@Sigrok"),
        SimpleNamespace(content="@Sigrok\nolder rule\n@Sigrok"),
        SimpleNamespace(content="not a rule pin"),
        SimpleNamespace(content="@Sigrok\n\n@Sigrok"),  # empty body
    ]
    channel = SimpleNamespace(id=42, pins=AsyncMock(return_value=pins))
    rules = await fetch_channel_rules(channel, bot)
    assert rules == "older rule\n\nnewer rule"


@pytest.mark.asyncio
async def test_fetch_uses_cache() -> None:
    bot = _FakeBot()
    pins_fn = AsyncMock(
        return_value=[SimpleNamespace(content="@Sigrok\ncached rule\n@Sigrok")]
    )
    channel = SimpleNamespace(id=99, pins=pins_fn)
    first = await fetch_channel_rules(channel, bot)
    second = await fetch_channel_rules(channel, bot)
    assert first == second == "cached rule"
    assert pins_fn.await_count == 1


def test_dynamic_suffix_includes_channel_rules() -> None:
    suffix = GenAIBase._dynamic_system_suffix(
        reply_mode="discussion",
        channel_rules="no spoilers\nbe brief",
    )
    assert "[Channel rules]" in suffix
    assert "no spoilers" in suffix
    assert "be brief" in suffix
    assert "do not override your core identity" in suffix


def test_static_prefix_unchanged_by_channel_rules() -> None:
    without = GenAIBase._static_system_prefix("discord")
    with_rules = GenAIBase._build_personality_system_prompt(
        "discord", channel_rules="no spoilers"
    )
    assert with_rules.startswith(without)
    assert "[Channel rules]" in with_rules
    assert "[Channel rules]" not in without
