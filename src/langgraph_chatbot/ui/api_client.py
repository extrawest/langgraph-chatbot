"""Thin async client for our FastAPI backend.

The UI knows nothing about the database or JWT internals - only about the HTTP
contract. Swapping the UI (or testing it) means swapping this one class.
"""

import json

import httpx


class ApiError(Exception):
    """Backend answered with an error (or could not be reached: status_code=0)."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def _extract_detail(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text or response.reason_phrase

    detail = body.get("detail") if isinstance(body, dict) else None
    if isinstance(detail, str):
        return detail
    if isinstance(detail, list):  # pydantic validation errors (422)
        return "; ".join(
            str(item.get("msg", item)) if isinstance(item, dict) else str(item)
            for item in detail
        )
    return response.reason_phrase


class ApiClient:
    def __init__(self, base_url: str, timeout: float = 10.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(self._timeout, read=60.0),
                follow_redirects=True,
            )
        return self._client

    async def _request(
        self, method: str, path: str, *, token: str | None = None, **kwargs
    ):
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        try:
            client = self._get_client()
            response = await client.request(method, path, headers=headers, **kwargs)
        except httpx.RequestError as exc:
            raise ApiError(0, f"Backend is unreachable ({self._base_url})") from exc

        if response.is_error:
            raise ApiError(response.status_code, _extract_detail(response))
        return response.json() if response.content else None

    # --- auth ---------------------------------------------------------------
    async def register(self, email: str, password: str) -> None:
        await self._request(
            "POST", "/auth/register", json={"email": email, "password": password}
        )

    async def login(self, email: str, password: str) -> str:
        # OAuth2 Password Flow: form-data, and the field is called "username"
        data = await self._request(
            "POST", "/auth/token", data={"username": email, "password": password}
        )
        return data["access_token"]

    async def me(self, token: str) -> dict:
        return await self._request("GET", "/auth/users/me", token=token)

    # --- chats --------------------------------------------------------------
    async def list_chats(self, token: str) -> list[dict]:
        return await self._request("GET", "/chats", token=token)

    async def get_messages(self, token: str, chat_id: str) -> list[dict]:
        return await self._request("GET", f"/chats/{chat_id}/messages", token=token)

    async def create_chat(self, token: str, title: str) -> dict:
        return await self._request("POST", "/chats", token=token, json={"title": title})

    async def rename_chat(self, token: str, chat_id: str, title: str) -> dict:
        return await self._request(
            "PATCH", f"/chats/{chat_id}", token=token, json={"title": title}
        )

    async def delete_chat(self, token: str, chat_id: str) -> None:
        await self._request("DELETE", f"/chats/{chat_id}", token=token)

    async def stream_message(
        self, message: str, chat_id: str | None, *, token: str | None = None
    ):
        """Yield (event_name, data) pairs from the SSE stream of POST /chats/messages."""
        client = self._get_client()
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        body = {"message": message, "chat_id": chat_id}

        try:
            async with client.stream(
                "POST", "/chats/messages", headers=headers, json=body
            ) as response:
                if response.is_error:
                    await response.aread()  # in stream mode the body is not read yet
                    raise ApiError(response.status_code, _extract_detail(response))

                event_name = None
                async for line in response.aiter_lines():
                    if line.startswith("event:"):
                        event_name = line.removeprefix("event:").strip()
                    elif line.startswith("data:"):
                        yield event_name, json.loads(line.removeprefix("data:").strip())
                    elif not line:
                        event_name = None  # blank line = end of the event
        except httpx.RequestError as exc:
            raise ApiError(0, f"Backend is unreachable ({self._base_url})") from exc

    async def start_voice_session(self, *, chat_id: str, token: str, **kwargs):
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        client = self._get_client()
        response = await client.request("GET", "/voice-ui", headers=headers, **kwargs)

        return response
