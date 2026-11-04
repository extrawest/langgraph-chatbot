from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph_chatbot.config import Settings


def build_llm(settings: Settings) -> BaseChatModel:
    # Not ChatOpenAI over Gemini's OpenAI-compatible endpoint: Gemini 3 requires the
    # `thought_signature` of every tool call to be sent back with it, and the OpenAI
    # client drops that field, so any tool loop fails with a 400.
    # (Embeddings and Mem0 keep using the compatible endpoint: llm_base_url.)
    return ChatGoogleGenerativeAI(
        model=settings.llm_model,
        api_key=settings.llm_api_key.get_secret_value(),
    )
