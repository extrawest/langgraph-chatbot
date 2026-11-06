import logging

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    TurnHandlingOptions,
    inference,
)
from livekit.plugins import elevenlabs, langchain, silero

from langgraph_chatbot.ai.bootstrap import build_runtime
from langgraph_chatbot.ai.context import ContextSchema, Mode
from langgraph_chatbot.config import get_settings
from langgraph_chatbot.logger import log_context, setup_logging
from langgraph_chatbot.voice.constants import VOICE_AGENT_NAME
from langgraph_chatbot.voice.spoken_graph import SpokenOnly

# .env first: setup_logging reads LOG_LEVEL / LOG_FORMAT from the environment
load_dotenv()
setup_logging()

logger = logging.getLogger(__name__)


class Assistant(Agent):
    def __init__(self) -> None:
        # The real system prompt lives in the graph (VOICE_SYSTEM_PROMPT): the graph
        # only looks at the current turn, so these instructions never reach the model.
        # No chat_ctx either - history comes from Qdrant inside the graph.
        super().__init__(instructions="")


server = AgentServer()


@server.rtc_session(agent_name=VOICE_AGENT_NAME)
async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()
    try:
        participant = await ctx.wait_for_participant()
    except RuntimeError:
        logger.info("Room closed before a participant joined")
        return

    user_id = participant.identity
    chat_id = participant.attributes.get("chat_id")
    if not chat_id:
        logger.error("Participant %s joined without a chat_id", user_id)
        ctx.shutdown(reason="missing chat_id")
        return
    log_context.set({"chat_id": chat_id, "user_id": user_id, "mode": "voice"})

    settings = get_settings()
    try:
        runtime = await build_runtime(settings, ensure_collection=False)
    except Exception:
        logger.exception("Voice runtime failed to start")
        ctx.shutdown(reason="runtime failed to start")
        return

    session = AgentSession(
        vad=silero.VAD.load(),
        stt=elevenlabs.STT(model="scribe_v2_realtime", language_code="ru"),
        llm=langchain.LLMAdapter(
            graph=SpokenOnly(runtime.graph),
            context=ContextSchema(
                user_id=user_id,
                chat_id=chat_id,
                mode=Mode.VOICE,
            ),
        ),
        tts=elevenlabs.TTS(voice_id=settings.eleven_voice_id, language="ru"),
        turn_handling=TurnHandlingOptions(turn_detection=inference.TurnDetector()),
    )

    @session.on("user_input_transcribed")
    def _on_text(ev): ...

    @session.on("error")
    def _on_error(ev):
        # ev.source is the STT / LLM / TTS component that failed
        logger.error(
            "Voice session error from %s: %s", type(ev.source).__name__, ev.error
        )

    ctx.add_shutdown_callback(runtime.aclose)
    await session.start(room=ctx.room, agent=Assistant())


if __name__ == "__main__":
    agents.cli.run_app(server)
