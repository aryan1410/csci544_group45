import sys
from typing import Any


class Logger:

    def __init__(self, prefix: str = ""):
        self.prefix = prefix

    def _format_message(self, component: str, message: str) -> str:
        if self.prefix:
            return f"[{self.prefix}] [{component}] {message}"
        return f"[{component}] {message}"

    def info(self, component: str, message: str, force_flush: bool = False) -> None:
        print(self._format_message(component, message))
        if force_flush:
            sys.stdout.flush()

    def warning(self, component: str, message: str, force_flush: bool = False) -> None:
        formatted = f"{self._format_message(component, message)}"
        print(formatted)
        if force_flush:
            sys.stdout.flush()

    def error(self, component: str, message: str, force_flush: bool = False) -> None:
        formatted = f"{self._format_message(component, message)}"
        print(formatted)
        if force_flush:
            sys.stdout.flush()

    def success(self, component: str, message: str, force_flush: bool = False) -> None:
        formatted = f"{self._format_message(component, message)}"
        print(formatted)
        if force_flush:
            sys.stdout.flush()

    def debug(self, component: str, message: str, data: Any = None, force_flush: bool = False) -> None:
        formatted = self._format_message(component, message)
        if data is not None:
            formatted += f" | Data: {data}"
        print(formatted)
        if force_flush:
            sys.stdout.flush()


logger = Logger()


def log(message: str, force_flush: bool = False) -> None:
    print(message)
    if force_flush:
        sys.stdout.flush()