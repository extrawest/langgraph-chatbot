"""Thin async client for our FastAPI backend.

The UI knows nothing about the database or JWT internals - only about the HTTP
contract. Swapping the UI (or testing it) means swapping this one class.
"""

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

    async def _request(
        self, method: str, path: str, *, token: str | None = None, **kwargs
    ):
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        try:
            async with httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout,
                follow_redirects=True,
            ) as client:
                response = await client.request(method, path, headers=headers, **kwargs)
        except httpx.RequestError as exc:
            raise ApiError(0, f"Backend is unreachable ({self._base_url})") from exc

        if response.is_error:
            raise ApiError(response.status_code, _extract_detail(response))
        return response.json() if response.content else None

    # --- auth ---------------------------------------------------------------
    async def register(self, email: str, password: str) -> None:
        await self._request(
            "POST", "/auth/register", json={"username": email, "password": password}
        )

    async def login(self, email: str, password: str) -> str:
        # OAuth2 Password Flow: form-data, and the field is called "username"
        data = await self._request(
            "POST", "/auth/token", data={"email": email, "password": password}
        )
        return data["access_token"]

    async def me(self, token: str) -> dict:
        return await self._request("GET", "/auth/users/me", token=token)

    # --- chats --------------------------------------------------------------
    async def list_chats(self, token: str) -> list[dict]:
        return await self._request("GET", "/chats", token=token)

    async def create_chat(self, token: str, title: str) -> dict:
        return await self._request("POST", "/chats", token=token, json={"title": title})

    async def rename_chat(self, token: str, chat_id: str, title: str) -> dict:
        return await self._request(
            "PATCH", f"/chats/{chat_id}", token=token, json={"title": title}
        )

    async def delete_chat(self, token: str, chat_id: str) -> None:
        await self._request("DELETE", f"/chats/{chat_id}", token=token)
