"""The single prompt shared by every model, and SQL extraction from model output."""

from __future__ import annotations

import re

SYSTEM = (
    "You translate questions into SQLite queries. "
    "Reply with exactly one SQL query and nothing else."
)


def user_message(schema: str, question: str) -> str:
    return f"Database schema:\n{schema}\n\nQuestion: {question}"


def build_messages(schema: str, question: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_message(schema, question)},
    ]


_FENCE = re.compile(r"```(?:sql|sqlite)?\s*(.*?)```", flags=re.IGNORECASE | re.DOTALL)
_START = re.compile(r"\b(select|with)\b", flags=re.IGNORECASE)


def extract_sql(text: str) -> str:
    """Pull one SQL statement out of a model reply (handles code fences and chatter)."""
    if not text:
        return ""
    m = _FENCE.search(text)
    if m:
        text = m.group(1)
    m = _START.search(text)
    if m:
        text = text[m.start():]
    # keep the first statement only
    text = text.split(";")[0]
    return " ".join(text.split()).strip()
