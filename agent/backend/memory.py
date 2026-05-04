from collections import deque
from typing import Any


class RollingBuffer:

    def __init__(self, max_len: int = 12) -> None:
        self.buf: deque[dict[str, Any]] = deque(maxlen=max_len)
        self.max_len = max_len

    def append(self, message: dict[str, Any]) -> None:
        self.buf.append(message)

    def extend(self, messages: list[dict[str, Any]]) -> None:
        for message in messages:
            self.buf.append(message)

    def as_messages(self) -> list[dict[str, Any]]:
        return list(self.buf)

    def clear(self) -> None:
        self.buf.clear()

    def __len__(self) -> int:
        return len(self.buf)

    def __repr__(self) -> str:
        return f"RollingBuffer(size={len(self.buf)}, max={self.max_len})"