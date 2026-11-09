import time
import uuid
from datetime import timedelta

import jwt
import pytest

from tests.helpers import DEFAULT_PASSWORD


async def login(client, email, password=DEFAULT_PASSWORD):
    return await client.post("/auth/token", data={"username": email, "password": password})


# ---------------------------------------------------------------- register
async def test_register_returns_a_working_token(client):
    response = await client.post(
        "/auth/register", json={"email": "New.User@Example.com", "password": "pw-123456"}
    )

    assert response.status_code == 201
    body = response.json()
    assert body["token_type"] == "bearer"
    me = await client.get("/auth/users/me", headers={"Authorization": f"Bearer {body['access_token']}"})
    assert me.status_code == 200
    assert me.json()["email"] == "new.user@example.com"


async def test_register_duplicate_email(client, user_factory):
    await user_factory(email="taken@example.com")

    response = await client.post("/auth/register", json={"email": "taken@example.com", "password": "x"})

    assert response.status_code == 409
    assert response.json() == {"detail": "Email already registered"}


@pytest.mark.parametrize(
    "body",
    [
        {"email": "not-an-email", "password": "pw"},
        {"email": "a@example.com"},
        {"password": "pw"},
    ],
)
async def test_register_validation(client, body):
    assert (await client.post("/auth/register", json=body)).status_code == 422


# ------------------------------------------------------------------- login
async def test_login(client, user_factory, settings):
    user = await user_factory(email="me@example.com")

    response = await login(client, "me@example.com")

    assert response.status_code == 200
    token = response.json()["access_token"]
    claims = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    assert claims["sub"] == str(user.id)
    ttl = claims["exp"] - time.time()
    assert settings.access_token_ttl_minutes * 60 - 5 < ttl <= settings.access_token_ttl_minutes * 60


@pytest.mark.parametrize(
    ("email", "password"),
    [("me@example.com", "wrong-password"), ("ghost@example.com", DEFAULT_PASSWORD)],
)
async def test_login_failure_does_not_reveal_which_part_was_wrong(client, user_factory, email, password):
    await user_factory(email="me@example.com")

    response = await login(client, email, password)

    assert response.status_code == 401
    assert response.json() == {"detail": "Incorrect username or password"}
    assert response.headers["WWW-Authenticate"] == "Bearer"


async def test_login_requires_form_fields(client):
    assert (await client.post("/auth/token", data={"username": "x"})).status_code == 422


# ---------------------------------------------------------------- users/me
async def test_me(client, user_factory, auth_headers):
    user = await user_factory(email="me@example.com")

    response = await client.get("/auth/users/me", headers=auth_headers(user))

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == str(user.id)
    assert body["email"] == "me@example.com"
    assert body["is_active"] is True
    assert "hashed_password" not in body
    assert set(body) == {"id", "email", "is_active", "created_at"}


async def test_me_without_token(client):
    response = await client.get("/auth/users/me")
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"


def _bearer(token):
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.parametrize(
    "make_bad_token",
    [
        pytest.param(lambda s, u: "garbage", id="malformed"),
        pytest.param(
            lambda s, u: jwt.encode({"sub": str(u.id)}, "other-secret-that-is-long-enough!!", "HS256"),
            id="wrong-secret",
        ),
        pytest.param(
            lambda s, u: jwt.encode({"foo": "bar"}, s.jwt_secret, "HS256"), id="no-sub"
        ),
        pytest.param(
            lambda s, u: jwt.encode({"sub": "not-a-uuid"}, s.jwt_secret, "HS256"), id="sub-not-uuid"
        ),
        pytest.param(
            lambda s, u: jwt.encode({"sub": str(uuid.uuid4())}, s.jwt_secret, "HS256"), id="unknown-user"
        ),
        pytest.param(
            lambda s, u: jwt.encode({"sub": str(u.id), "exp": 1}, s.jwt_secret, "HS256"), id="expired"
        ),
        pytest.param(
            lambda s, u: jwt.encode({"sub": str(u.id)}, "", "none"), id="alg-none"
        ),
    ],
)
async def test_me_rejects_bad_tokens(client, user_factory, settings, make_bad_token):
    user = await user_factory()

    response = await client.get("/auth/users/me", headers=_bearer(make_bad_token(settings, user)))

    assert response.status_code == 401
    assert response.json() == {"detail": "Could not validate credentials"}


async def test_me_rejects_inactive_user(client, user_factory, auth_headers):
    user = await user_factory(is_active=False)
    assert (await client.get("/auth/users/me", headers=auth_headers(user))).status_code == 401


async def test_expired_token_from_make_token(client, user_factory, make_token):
    user = await user_factory()
    token = make_token(user.id, expires_delta=timedelta(seconds=-5))
    assert (await client.get("/auth/users/me", headers=_bearer(token))).status_code == 401
