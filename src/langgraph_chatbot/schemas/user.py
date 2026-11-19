import re
import uuid
from datetime import datetime
from typing import Self

from pydantic import BaseModel, ConfigDict, EmailStr, Field, model_validator


class UserCreate(BaseModel):
    # Required to support (?=...) lookaheads in Field(pattern=...)
    model_config = ConfigDict(regex_engine="python-re")

    email: EmailStr
    password: str
    confirm_password: str

    @model_validator(mode="after")
    def validate_password(self) -> Self:
        if len(self.password) < 8 or len(self.password) > 20:
            raise ValueError("Password must contain 8-20 characters")

        if not bool(re.fullmatch(r"^(?=.*\d)(?=.*[^a-zA-Z0-9\s]).+$", self.password)):
            raise ValueError("Password must contain at least 1 digit, and 1 symbol")

        if self.password != self.confirm_password:
            raise ValueError("Passwords do not match")
        return self


class UserRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    created_at: datetime
    email: str
    is_active: bool
