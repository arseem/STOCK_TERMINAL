from __future__ import annotations

from typing import Callable, Any


# Simple indirection layer so non-UI code can log without depending on Textual widgets.
_writer: Callable[[Any], None] = print


def set_writer(writer: Callable[[Any], None]) -> None:
    global _writer
    _writer = writer


def write(message: Any) -> None:
    try:
        _writer(message)
    except Exception:
        # Never let logging crash the app.
        pass
