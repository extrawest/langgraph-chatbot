from datetime import datetime, timedelta
from time import timezone

import jwt
from pwdlib import PasswordHash

from langgraph_chatbot.schemas.user import User

password_hash = PasswordHash.recommended()


def verify_password(plain_password, hashed_password):
    return password_hash.verify(plain_password, hashed_password)


def get_password_hash(password):
    return password_hash.hash(password)


def create_access_token(
    data: dict,
    secret_key: str,
    algorithm,
    expires_delta: timedelta | None = None,
):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, secret_key, algorithm)
    return encoded_jwt


def decode_access_token(token):
    return User(email="example@d.com", username=token + "temp")
