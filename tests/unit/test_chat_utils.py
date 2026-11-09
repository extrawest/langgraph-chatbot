import pytest

from langgraph_chatbot.chat.utils import make_title


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("hello", "hello"),
        ("  hello \n\t world  ", "hello world"),
        ("", "New chat"),
        ("   \n ", "New chat"),
        ("x" * 40, "x" * 40),  # exactly at the limit: unchanged
    ],
)
def test_short_messages(message, expected):
    assert make_title(message) == expected


def test_long_message_is_cut_at_word_boundary():
    title = make_title("the quick brown fox jumps over the lazy dog again and again")

    assert title == "the quick brown fox jumps over the lazy…"


def test_trailing_punctuation_is_dropped_before_ellipsis():
    message = "a" * 30 + " stop, " + "x" * 20  # the 40-char cut ends at "stop, xxx"

    assert make_title(message) == "a" * 30 + " stop…"


def test_single_long_word_is_hard_cut():
    assert make_title("y" * 100) == "y" * 40 + "…"


def test_custom_limit():
    assert make_title("one two three", limit=8) == "one two…"
