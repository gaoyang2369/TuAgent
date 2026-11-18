from dataclasses import dataclass
from typing import Any, Literal, TypeAlias


@dataclass(frozen=True, slots=True)
class UserMessage:
    content: str


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A model's tool call with its original JSON arguments."""

    id: str
    name: str
    arguments: str


@dataclass(frozen=True, slots=True)
class AssistantMessage:
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "tool_calls", tuple(self.tool_calls))


@dataclass(frozen=True, slots=True)
class ToolResultMessage:
    tool_call_id: str
    content: str
    is_error: bool = False
    error_code: str | None = None
    details: Any = None


Message: TypeAlias = UserMessage | AssistantMessage | ToolResultMessage
RunStatus: TypeAlias = Literal[
    "completed", "failed", "turn_limit", "cancelled", "incomplete"
]


@dataclass(frozen=True, slots=True)
class RunResult:
    status: RunStatus
    messages: tuple[Message, ...]
    turns: int
    text: str | None = None
    error_code: str | None = None
