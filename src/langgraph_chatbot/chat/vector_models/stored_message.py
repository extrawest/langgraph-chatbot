from pydantic import BaseModel


class StoredMessage(BaseModel):
    role: str
    content: str
