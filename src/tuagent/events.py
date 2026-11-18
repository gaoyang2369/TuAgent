from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, TypeAlias

from .messages import RunStatus
from .model import ModelMessage

RunEventKind: TypeAlias = Literal[
    "run_start",
    "run_end",
    "turn_start",
    "turn_end",
    "message",
    "text_delta",
    "tool_start",
    "tool_end",
]


@dataclass(frozen=True, slots=True)
class RunEvent:
    """A read-only observation with sequence numbers starting at one per run."""

    kind: RunEventKind
    run_id: str
    sequence: int
    turn: int | None = None
    tool_call_id: str | None = None
    message: ModelMessage | None = None
    delta: str | None = None
    status: RunStatus | None = None
    error_code: str | None = None


EventHandler: TypeAlias = Callable[[RunEvent], None]
