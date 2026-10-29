def make_title(message: str, limit: int = 40) -> str:
    """Chat title from the first user message: collapse whitespace, cut at a word boundary."""
    text = " ".join(message.split())
    if len(text) <= limit:
        return text or "New chat"
    return text[:limit].rsplit(" ", 1)[0].rstrip(".,;:!?") + "…"
