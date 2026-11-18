import re
import uuid
from datetime import datetime
from typing import Self

from pydantic import BaseModel, ConfigDict, EmailStr, Field, model_validator


class UserCreate(BaseModel):
    # Required to support (?=...) lookaheads in Field(pattern=...)
    model_config = ConfigDict(regex_engine="python-re")

    email: EmailStr
    password: str = Field(
        min_length=8,
        max_length=20,
    )
    confirm_password: str

    @model_validator(mode="after")
    def check_password_match(self) -> Self:
        if self.password != self.confirm_password:
            raise ValueError("Passwords do not match")

        if bool(re.fullmatch(r"[A-Z]{2}-\d{3}", self.password)):
            raise ValueError("Password must contain at least 1 digit, and 1 symbol")
        return self


class UserRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    created_at: datetime
    email: str
    is_active: bool
