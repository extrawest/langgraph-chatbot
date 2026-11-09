import pytest
from pydantic import ValidationError

from langgraph_chatbot.schemas.chat import ChatCreate, ChatUpdate


@pytest.mark.parametrize("model", [ChatCreate, ChatUpdate])
@pytest.mark.parametrize("title", ["", "   ", "\t\n", "x" * 256])
def test_invalid_titles(model, title):
    with pytest.raises(ValidationError):
        model(title=title)


@pytest.mark.parametrize("model", [ChatCreate, ChatUpdate])
def test_title_is_stripped(model):
    assert model(title="  Trip  ").title == "Trip"


def test_title_length_counts_after_stripping():
    assert ChatUpdate(title=" " + "x" * 255 + " ").title == "x" * 255


def test_create_default_title():
    assert ChatCreate().title == "New chat"


def test_update_requires_title():
    with pytest.raises(ValidationError):
        ChatUpdate()
