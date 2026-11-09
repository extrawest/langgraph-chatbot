from datetime import UTC, datetime, timedelta

import jwt
import pytest

from langgraph_chatbot.auth.security import (
    create_access_token,
    decode_access_token,
    get_password_hash,
    verify_password,
)

SECRET = "unit-test-secret-that-is-long-enough-for-hs256"


def test_password_hash_roundtrip():
    hashed = get_password_hash("s3cret")

    assert hashed != "s3cret"
    assert verify_password("s3cret", hashed)
    assert not verify_password("wrong", hashed)


def test_password_hash_is_salted():
    assert get_password_hash("same") != get_password_hash("same")


def test_token_roundtrip_keeps_claims():
    token = create_access_token({"sub": "user-1", "role": "x"}, SECRET, "HS256")

    payload = decode_access_token(token, SECRET, "HS256")

    assert payload["sub"] == "user-1"
    assert payload["role"] == "x"


def test_token_does_not_mutate_input():
    data = {"sub": "user-1"}
    create_access_token(data, SECRET, "HS256")
    assert data == {"sub": "user-1"}


@pytest.mark.parametrize(
    ("delta", "expected"),
    [(None, timedelta(minutes=15)), (timedelta(hours=2), timedelta(hours=2))],
)
def test_token_expiry(delta, expected):
    before = datetime.now(UTC)
    token = create_access_token({"sub": "u"}, SECRET, "HS256", expires_delta=delta)

    exp = datetime.fromtimestamp(
        decode_access_token(token, SECRET, "HS256")["exp"], UTC
    )

    # exp is stored with second precision
    assert before + expected - timedelta(seconds=2) <= exp
    assert exp <= datetime.now(UTC) + expected + timedelta(seconds=1)


def test_expired_token_is_rejected():
    token = create_access_token(
        {"sub": "u"}, SECRET, "HS256", expires_delta=timedelta(seconds=-1)
    )
    with pytest.raises(jwt.ExpiredSignatureError):
        decode_access_token(token, SECRET, "HS256")


def test_token_signed_with_other_secret_is_rejected():
    token = create_access_token({"sub": "u"}, "another-secret-long-enough-for-hs256!!", "HS256")
    with pytest.raises(jwt.InvalidSignatureError):
        decode_access_token(token, SECRET, "HS256")


def test_algorithm_must_match():
    token = create_access_token({"sub": "u"}, SECRET * 2, "HS512")
    with pytest.raises(jwt.InvalidAlgorithmError):
        decode_access_token(token, SECRET, "HS256")
