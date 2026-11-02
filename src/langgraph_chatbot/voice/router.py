from fastapi import APIRouter, HTTPException
from fastapi.staticfiles import StaticFiles

from langgraph_chatbot.chat.deps import ChatServiceDep
from langgraph_chatbot.deps import CurrentUser, SettingsDep
from langgraph_chatbot.schemas.voice import VoiceTokenIn, VoiceTokenOut
from langgraph_chatbot.voice.utils import build_voice_token

router = APIRouter(prefix="/voice")


@router.post("/token", response_model=VoiceTokenOut)
async def voice_token(
    data: VoiceTokenIn,
    current_user: CurrentUser,
    service: ChatServiceDep,
    settings: SettingsDep,
):
    if not settings.livekit_api_key or not settings.livekit_api_secret:
        raise HTTPException(status_code=503)

    chat = await service.get_or_create_chat(
        data.chat_id,
        current_user.id,
        title="Voice chat",
    )

    if chat is None:
        raise HTTPException(status_code=404, detail="Chat not found")
    token = build_voice_token(settings, user_id=current_user.id, chat_id=chat.id)
    return VoiceTokenOut(token=token, url=settings.livekit_url, chat_id=chat.id)
