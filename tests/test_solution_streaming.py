"""Streaming only the `solution` field out of a half-arrived JSON reply.

A Shape C brain generates `{"solution": "...", "confidence": 95, "unknown": ""}`
token by token. The UI must see the answer appear as it is written -- and must
*never* see the scaffolding: not the brace, not the quoted key, and above all
not `confidence`/`unknown`, which are routing metadata rather than an answer.

The `(hollowbyte)-feat/chat_app` branch declined to stream the local half for
exactly this reason ("streaming it would show the user the scaffolding"), so
this is the piece that makes token-level streaming usable on the local tier.
"""
from __future__ import annotations

import json

import pytest

from two_brain_router.signals.structured import SolutionStreamer, parse_structured


def _stream(chunks: list[str]) -> tuple[str, SolutionStreamer]:
    s = SolutionStreamer()
    out = [s.feed(c) for c in chunks]
    out.append(s.finish())
    return "".join(out), s


def test_only_the_solution_text_is_streamed():
    """The headline requirement: metadata fields never reach the UI."""
    streamed, s = _stream(
        ['{"solu', 'tion": "Par', "is is the ca", 'pital.", "confid',
         'ence": 95, "unknown": "the population"}']
    )
    assert streamed == "Paris is the capital."
    assert "confidence" not in streamed
    assert "unknown" not in streamed
    assert "95" not in streamed
    assert "{" not in streamed
    # ...and the raw buffer is still intact for the real parse afterwards.
    assert parse_structured(s.raw).unknown == "the population"


def test_a_key_split_across_chunks_is_still_found():
    """Chunk boundaries fall anywhere, so nothing can be matched per-chunk."""
    streamed, _ = _stream(['{"', "so", "lu", "ti", 'on"', ":", ' "', "hi", '"}'])
    assert streamed == "hi"


def test_one_character_at_a_time():
    """The degenerate case a token stream actually approaches."""
    raw = '{"solution": "abc def", "confidence": 90, "unknown": ""}'
    streamed, _ = _stream(list(raw))
    assert streamed == "abc def"


def test_escapes_are_decoded_not_shown_raw():
    raw = json.dumps({"solution": 'He said "hi".\nLine two.', "confidence": 90, "unknown": ""})
    streamed, _ = _stream(list(raw))
    assert streamed == 'He said "hi".\nLine two.'
    # Decoded, so the *literal* two-character sequences must not survive into
    # what the user reads. (The real newline is expected and asserted above.)
    assert r"\n" not in streamed
    assert r"\"" not in streamed


def test_a_unicode_escape_split_mid_sequence_is_never_half_emitted():
    r"""A `\uXXXX` cut in half must be withheld, not guessed at.

    Emitting a broken character puts something on screen that cannot be taken
    back -- there is no edit, only append.
    """
    raw = '{"solution": "caf\u00e9 open", "confidence": 90, "unknown": ""}'
    # Split precisely inside the escape sequence.
    cut = raw.index("\u00e9") + 3
    streamed, _ = _stream([raw[:cut], raw[cut:]])
    assert streamed == "café open"


def test_the_closing_quote_ends_the_stream_even_with_more_json_after():
    streamed, _ = _stream(['{"solution": "done", "confidence": 100, "unknown": "later"}'])
    assert streamed == "done"


def test_plain_prose_with_no_json_is_streamed_verbatim():
    """A model that ignores the format entirely still shows its words.

    `parse_structured` already degrades to prose; the stream has to degrade the
    same way, or a badly-behaved model produces a blank screen rather than a
    badly-formatted answer.
    """
    text = "Tokyo is in Japan Standard Time, and there is no JSON here at all."
    streamed, _ = _stream([text[:20], text[20:]])
    assert streamed == text


def test_a_short_prose_reply_is_not_swallowed_by_the_sniff():
    """Shorter than the JSON sniff window, so only `finish()` can release it."""
    streamed, _ = _stream(["JST."])
    assert streamed == "JST."


def test_a_code_fence_is_treated_as_structured_not_prose():
    """Models add ```json fences despite being told not to; that is still JSON,
    so the fence itself must not be streamed as if it were the answer."""
    streamed, _ = _stream(['```json\n{"solution": "fenced", "confidence": 90, "unknown": ""}\n```'])
    assert streamed == "fenced"
    assert "```" not in streamed


@pytest.mark.parametrize("alias", ["solution", "answer", "response"])
def test_the_field_aliases_models_actually_use_are_honoured(alias):
    streamed, _ = _stream(['{"%s": "aliased", "confidence": 90}' % alias])
    assert streamed == "aliased"


def test_nothing_is_emitted_twice():
    """Every feed returns only what is *new*; a UI appends blindly."""
    s = SolutionStreamer()
    pieces = [s.feed(c) for c in ['{"solution": "abc', "def", 'ghi", "confidence": 9}']]
    assert "".join(pieces) == "abcdefghi"
    assert s.finish() == ""


def test_an_empty_solution_streams_nothing_but_does_not_error():
    streamed, s = _stream(['{"solution": "", "confidence": 10, "unknown": "all of it"}'])
    assert streamed == ""
    assert parse_structured(s.raw).unknown == "all of it"
