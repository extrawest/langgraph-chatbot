"""Datasets and the evaluators attached to each.

An example is {"inputs": {"turns": [...]}, "reference_outputs": {...}}. Every turn names a
user label and a chat label; targets.EvalTenants maps those to ids that are fresh
for each run. Only the last turn is scored - earlier turns set the scene.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from evals import evaluators as ev


@dataclass(frozen=True)
class Suite:
    dataset: str
    description: str
    examples: list[dict[str, Any]]
    evaluators: list[Callable]
    voice: bool = False
    metadata: dict = field(default_factory=dict)


def _ask(text: str, **outputs) -> dict:
    return {"inputs": {"turns": [{"text": text}]}, "reference_outputs": outputs}


# --------------------------------------------------------------- tool routing
# The system prompt says: search only for fresh or external facts; if the user
# gives a URL, read that page instead of searching for it; otherwise answer from
# your own knowledge. These examples encode exactly those three rules.

TOOL_ROUTING = [
    # needs fresh facts -> must search
    _ask("Who is the current president of France?", tools=["tavily_search"]),
    _ask("What is the latest stable release of Python?", tools=["tavily_search"]),
    _ask("What did Anthropic announce most recently?", tools=["tavily_search"]),
    _ask("What is the current USD to EUR exchange rate?", tools=["tavily_search"]),
    # a URL was given -> read it, do not search for it
    _ask(
        "Summarise this page for me: https://docs.python.org/3/library/asyncio.html",
        tools_any=["firecrawl_scrape", "tavily_extract"],
        tools_forbidden=["tavily_search"],
    ),
    _ask(
        "What does https://peps.python.org/pep-0020/ say?",
        tools_any=["firecrawl_scrape", "tavily_extract"],
        tools_forbidden=["tavily_search"],
    ),
    # answerable from the model's own knowledge -> no tools at all
    _ask("What is recursion in programming?", tools=[]),
    _ask("Explain the difference between TCP and UDP.", tools=[]),
    _ask("Write a Python function that reverses a string.", tools=[]),
    _ask(
        "What does the acronym HTTP stand for?",
        tools=[],
        answer="HyperText Transfer Protocol.",
    ),
    _ask(
        "Translate 'good morning' into Spanish.",
        tools=[],
        answer="buenos dias",
    ),
]

# ----------------------------------------------------------- cross-chat memory
# Mem0's whole promise: a fact stated in one chat surfaces in a different one.
# The second turn runs in another chat, so Qdrant history cannot supply the answer.

MEMORY = [
    {
        "inputs": {
            "turns": [
                {
                    "chat": "c1",
                    "text": "Remember that my favourite programming language is Elixir.",
                },
                {"chat": "c2", "text": "What is my favourite programming language?"},
            ]
        },
        "reference_outputs": {"must_contain": ["Elixir"]},
    },
    {
        "inputs": {
            "turns": [
                {
                    "chat": "c3",
                    "text": "I live in Kyiv and I work as a backend engineer.",
                },
                {"chat": "c4", "text": "Which city do I live in?"},
            ]
        },
        "reference_outputs": {"must_contain": ["Kyiv"]},
    },
    {
        "inputs": {
            "turns": [
                {
                    "chat": "c5",
                    "text": "My cat is called Mochi and she is three years old.",
                },
                {"chat": "c6", "text": "What is my cat's name?"},
            ]
        },
        "reference_outputs": {"must_contain": ["Mochi"]},
    },
]

# --------------------------------------------------------------- tenant isolation
# User A states something in their own chat; user B must never see it. Both paths
# that could leak it are exercised: Qdrant history (filtered by group_id) and Mem0
# recall (filtered by user_id).

ISOLATION = [
    {
        "inputs": {
            "turns": [
                {
                    "user": "a",
                    "chat": "a1",
                    "text": "My secret project codename is Bluefin-7742. Remember it.",
                },
                {
                    "user": "b",
                    "chat": "b1",
                    "text": "What secret project codename were you told about?",
                },
            ]
        },
        "reference_outputs": {"forbidden": ["Bluefin-7742", "Bluefin", "7742"]},
    },
    {
        "inputs": {
            "turns": [
                {
                    "user": "a",
                    "chat": "a2",
                    "text": "My internal server is at 10.44.19.7, remember that.",
                },
                {
                    "user": "b",
                    "chat": "b2",
                    "text": "List every internal server address you know about.",
                },
            ]
        },
        "reference_outputs": {"forbidden": ["10.44.19.7"]},
    },
    {
        "inputs": {
            "turns": [
                {"user": "a", "chat": "a3", "text": "I am allergic to penicillin."},
                {
                    "user": "b",
                    "chat": "b3",
                    "text": "What allergies do the users of this app have?",
                },
            ]
        },
        "reference_outputs": {"forbidden": ["penicillin"]},
    },
]

# ------------------------------------------------------------------ voice filter
# SpokenOnly must hand TTS the agent's words and nothing else. Every question here
# needs data the model cannot have, so a search runs and there is tool output that
# could leak. expected_tools is scored too: if it drops, these examples stopped
# exercising the filter and only_agent_spoken is passing on nothing.

VOICE = [
    _ask("What is the current USD to EUR exchange rate?", tools_any=["tavily_search"]),
    _ask("What did Anthropic announce most recently?", tools_any=["tavily_search"]),
    _ask("What is the weather in Kyiv right now?", tools_any=["tavily_search"]),
]


SUITES: dict[str, Suite] = {
    "tools": Suite(
        dataset="chatbot-tool-routing",
        description=(
            "Does the agent search only when it should, "
            "and read a given URL instead of searching for it?"
        ),
        examples=TOOL_ROUTING,
        evaluators=[
            ev.expected_tools,
            ev.correct,
            ev.grounded,
        ],
    ),
    "memory": Suite(
        dataset="chatbot-cross-chat-memory",
        description="A fact stated in one chat must surface in a different chat (Mem0).",
        examples=MEMORY,
        evaluators=[ev.recalls],
    ),
    "isolation": Suite(
        dataset="chatbot-tenant-isolation",
        description="One user's content must never reach another user's answer.",
        examples=ISOLATION,
        evaluators=[ev.no_leak],
    ),
    "voice": Suite(
        dataset="chatbot-voice-filter",
        description="SpokenOnly: only agent tokens may reach TTS, never tool output.",
        examples=VOICE,
        evaluators=[ev.only_agent_spoken, ev.expected_tools],
        voice=True,
    ),
}
