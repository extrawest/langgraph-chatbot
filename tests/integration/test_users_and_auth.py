"""UserRepository + AuthService against a real (SQLite) database."""

import uuid

import pytest

from langgraph_chatbot.auth.security import verify_password
from langgraph_chatbot.auth.service import AuthService
from langgraph_chatbot.models.user import User
from langgraph_chatbot.users.repository import EmailAlreadyExists, UserRepository
from tests.helpers import DEFAULT_PASSWORD


@pytest.fixture
def repo(session):
    return UserRepository(session)


@pytest.fixture
def auth(session, repo):
    return AuthService(session, repo)


# -------------------------------------------------------------- repository
async def test_add_and_get(repo, session):
    user = await repo.add(User(email="a@example.com", hashed_password="h"))
    await session.commit()

    assert isinstance(user.id, uuid.UUID)
    assert user.is_active is True
    assert (await repo.get_by_id(user.id)).email == "a@example.com"
    assert (await repo.get_by_email("a@example.com")).id == user.id


async def test_missing_user(repo):
    assert await repo.get_by_id(uuid.uuid4()) is None
    assert await repo.get_by_email("nobody@example.com") is None


async def test_duplicate_email(repo, user_factory):
    await user_factory(email="dup@example.com")

    with pytest.raises(EmailAlreadyExists, match="dup@example.com"):
        await repo.add(User(email="dup@example.com", hashed_password="h"))


# ------------------------------------------------------------- AuthService
async def test_register_normalizes_email_and_hashes_password(auth, repo):
    user = await auth.register("  Mixed.Case@Example.COM ", "pw-123456")

    stored = await repo.get_by_email("mixed.case@example.com")
    assert stored.id == user.id
    assert stored.hashed_password != "pw-123456"
    assert verify_password("pw-123456", stored.hashed_password)


async def test_register_commits(auth, sessionmaker):
    user = await auth.register("commit@example.com", "pw")

    async with sessionmaker() as other_session:  # visible outside the request session
        assert await UserRepository(other_session).get_by_id(user.id) is not None


async def test_register_duplicate(auth, user_factory):
    await user_factory(email="taken@example.com")

    with pytest.raises(EmailAlreadyExists):
        await auth.register("TAKEN@example.com", "pw")


async def test_authenticate(auth, user_factory):
    user = await user_factory(email="login@example.com")

    assert (await auth.authenticate_user("login@example.com", DEFAULT_PASSWORD)).id == user.id
    assert await auth.authenticate_user("login@example.com", "wrong") is None
    assert await auth.authenticate_user("ghost@example.com", DEFAULT_PASSWORD) is None


async def test_authenticate_uses_same_normalization_as_register(auth):
    await auth.register("Me@Example.com", "pw-123456")

    assert await auth.authenticate_user("Me@Example.com", "pw-123456") is not None
    assert await auth.authenticate_user("  ME@example.COM ", "pw-123456") is not None


async def test_get_user_normalizes(auth, user_factory):
    user = await user_factory(email="find@example.com")
    assert (await auth.get_user("  FIND@example.com ")).id == user.id
