from datetime import timedelta
from uuid import uuid4

from livekit.api import AccessToken, VideoGrants
from livekit.protocol.agent_dispatch import RoomAgentDispatch
from livekit.protocol.room import RoomConfiguration

from langgraph_chatbot.config import Settings
from langgraph_chatbot.voice.constants import VOICE_AGENT_NAME


def build_voice_token(settings: Settings, *, user_id, chat_id) -> str:
    room = f"voice-{chat_id}-{uuid4().hex[:8]}"
    return (
        AccessToken(
            settings.livekit_api_key.get_secret_value(),
            settings.livekit_api_secret.get_secret_value(),
        )
        .with_identity(str(user_id))
        .with_attributes({"chat_id": str(chat_id)})
        .with_ttl(timedelta(minutes=10))
        .with_grants(VideoGrants(room_join=True, room=room))
        .with_room_config(
            RoomConfiguration(agents=[RoomAgentDispatch(agent_name=VOICE_AGENT_NAME)])
        )
        .to_jwt()
    )
