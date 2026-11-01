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

load_dotenv()


class Assistant(Agent):
    def __init__(self) -> None:
        # The real system prompt lives in the graph (VOICE_SYSTEM_PROMPT): the graph
        # only looks at the current turn, so these instructions never reach the model.
        # No chat_ctx either - history comes from Qdrant inside the graph.
        super().__init__(instructions="")


server = AgentServer()


@server.rtc_session(agent_name="chatbot-voice")
async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()
    participant = await ctx.wait_for_participant()

    user_id = participant.identity
    chat_id = participant.attributes["chat_id"]

    settings = get_settings()
    runtime = await build_runtime(settings, ensure_collection=False)

    session = AgentSession(
        vad=silero.VAD.load(),
        stt=elevenlabs.STT(model="assemblyai/universal-3-6-pro", language="en"),
        llm=langchain.LLMAdapter(
            graph=runtime.graph,
            context=ContextSchema(
                user_id=user_id,
                chat_id=chat_id,
                mode=Mode.VOICE,
            ),
        ),
        tts=elevenlabs.STT(model="assemblyai/universal-3-6-pro", language="en"),
        turn_handling=TurnHandlingOptions(turn_detection=inference.TurnDetector()),
    )

    @session.on("user_input_transcribed")
    def _on_text(ev): ...

    @session.on("error")
    def _on_error(ev): ...

    ctx.add_shutdown_callback(runtime.aclose)
    await session.start(room=ctx.room, agent=Assistant())


if __name__ == "__main__":
    agents.cli.run_app(server)
