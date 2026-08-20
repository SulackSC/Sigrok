from __future__ import annotations

from pathlib import Path

import pytest

from sigrok.streaming import response as response_module
from sigrok.streaming.response import (
    NSFW_FILTER_REPLACEMENT,
    _load_nsfw_blocklist,
    _normalize_for_nsfw_scan,
    _normalize_leet_for_nsfw_scan,
    _term_matches,
    apply_nsfw_filter,
)


def test_clean_text_passes_through() -> None:
    assert apply_nsfw_filter("still here, what's up") == "still here, what's up"
    assert apply_nsfw_filter("check the github repo for docs") == "check the github repo for docs"


def test_blocked_word_replaces_entire_response() -> None:
    assert apply_nsfw_filter("here is some porn for you") == NSFW_FILTER_REPLACEMENT
    assert apply_nsfw_filter("that is nsfw content") == NSFW_FILTER_REPLACEMENT


def test_multi_word_phrase_match() -> None:
    assert apply_nsfw_filter("try rule 34 instead") == NSFW_FILTER_REPLACEMENT
    assert apply_nsfw_filter("do not send nudes ever") == NSFW_FILTER_REPLACEMENT


def test_word_boundary_avoids_false_positives() -> None:
    assert apply_nsfw_filter("that class was hard") == "that class was hard"
    assert apply_nsfw_filter("pass the test please") == "pass the test please"
    assert apply_nsfw_filter("classic album tbh") == "classic album tbh"


def test_leet_obfuscation_is_caught() -> None:
    assert apply_nsfw_filter("check this p0rn link") == NSFW_FILTER_REPLACEMENT
    assert apply_nsfw_filter("totally n$fw stuff") == NSFW_FILTER_REPLACEMENT


def test_load_nsfw_blocklist_ignores_comments_and_blanks(tmp_path: Path) -> None:
    blocklist = tmp_path / "nsfw_blocklist.txt"
    blocklist.write_text(
        "# header comment\n\nbadword\n# inline ignored\n\n  another term  \n",
        encoding="utf-8",
    )
    assert _load_nsfw_blocklist(blocklist) == ("badword", "another term")


def test_normalize_for_nsfw_scan_collapses_whitespace_and_leet() -> None:
    assert _normalize_for_nsfw_scan("  PORN   LINK  ") == "porn link"
    assert _normalize_leet_for_nsfw_scan("  P0RN   LINK  ") == "porn link"
    assert _normalize_leet_for_nsfw_scan("n$fw") == "nsfw"


def test_term_matches_single_and_multi_word() -> None:
    text = "here is some porn and rule 34"
    assert _term_matches(text, "porn")
    assert _term_matches(text, "rule 34")
    assert not _term_matches("classic album", "ass")


def test_apply_nsfw_filter_noop_when_blocklist_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(response_module, "_NSFW_BLOCKLIST", tuple())
    assert apply_nsfw_filter("anything goes") == "anything goes"
