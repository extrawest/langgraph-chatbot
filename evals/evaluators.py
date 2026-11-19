"""Scoring functions. LangSmith names each metric after the function.

Most of these are deterministic on purpose: the properties this project claims -
correct tool routing, tenant isolation, cross-chat recall, no tool text in speech -
are all checkable without a judge model, which makes them cheap and repeatable.
"""

from pydantic import BaseModel, Field

from langgraph_chatbot.ai.llm import build_llm
from langgraph_chatbot.config import get_settings

# --------------------------------------------------------------- deterministic


def expected_tools(outputs: dict, reference_outputs: dict) -> bool:
    """Did the agent reach for the right tools, and only those?

    Encodes the system prompt's two rules: search only for fresh external facts,
    and read a URL the user already gave instead of searching for it.

    Reference keys: tools (all required), tools_any (at least one),
    tools_forbidden (none of these). All empty means "answer from your own
    knowledge, call nothing".
    """
    called = set(outputs.get("tools") or [])
    required = set(reference_outputs.get("tools") or [])
    any_of = set(reference_outputs.get("tools_any") or [])
    forbidden = set(reference_outputs.get("tools_forbidden") or [])

    if forbidden & called:
        return False
    if any_of and not (any_of & called):
        return False
    if required and not required.issubset(called):
        return False
    if not required and not any_of:
        return not called
    return True


def no_leak(outputs: dict, reference_outputs: dict) -> bool:
    """Tenant isolation, as a regression test.

    Another user said something in their own chat; none of it may appear here.
    """
    answer = (outputs.get("answer") or "").lower()
    return not any(s.lower() in answer for s in reference_outputs["forbidden"])


def recalls(outputs: dict, reference_outputs: dict) -> bool:
    """Cross-chat memory: a fact stated in an earlier chat came back in a new one."""
    answer = (outputs.get("answer") or "").lower()
    return all(s.lower() in answer for s in reference_outputs["must_contain"])


def only_agent_spoken(outputs: dict) -> bool:
    """SpokenOnly: TTS must receive agent tokens and nothing else.

    Vacuously true when no tool ran - there was no tool output to leak. The
    expected_tools metric on the same suite is what shows that happened.
    """
    return set(outputs.get("nodes") or []) == {"agent"}


# ------------------------------------------------------------------ llm judges


class _Verdict(BaseModel):
    passed: bool = Field(description="true if the answer meets the criterion")
    reason: str = Field(description="one short sentence")


_judge = None


def _get_judge():
    global _judge
    if _judge is None:
        _judge = build_llm(get_settings()).with_structured_output(_Verdict)
    return _judge


async def grounded(outputs: dict) -> bool:
    """When web tools were used, is the answer supported by what they returned?

    Vacuously true when nothing was retrieved - read this metric on the subset of
    examples where a search was expected.
    """
    source = outputs.get("tool_output") or ""
    if not source:
        return True

    verdict = await _get_judge().ainvoke(
        "Grade an AI assistant's answer for groundedness.\n\n"
        f"SOURCE (what the web tools returned):\n{source[:6000]}\n\n"
        f"ANSWER:\n{outputs.get('answer') or ''}\n\n"
        "Pass if everything ANSWER claims is supported by SOURCE or follows "
        "from it. Summarising several sources, giving a range, rounding, and "
        "converting units or rates all count as following from it. "
        "Fail only if ANSWER contradicts SOURCE, or states something that "
        "cannot be derived from it at all. "
        "Wording, style and length do not matter."
    )
    return verdict.passed


async def correct(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    """Does the answer carry the same key information as the reference?"""
    reference = reference_outputs.get("answer")
    if not reference:
        return True

    question = inputs["turns"][-1]["text"]
    verdict = await _get_judge().ainvoke(
        "Grade an AI assistant's answer against a reference.\n\n"
        f"QUESTION:\n{question}\n\n"
        f"REFERENCE ANSWER:\n{reference}\n\n"
        f"ANSWER:\n{outputs.get('answer') or ''}\n\n"
        "Pass if ANSWER conveys the same key information as REFERENCE. Extra "
        "detail, different wording and different length are fine. Fail only if "
        "ANSWER contradicts REFERENCE or omits what was asked for."
    )
    return verdict.passed
