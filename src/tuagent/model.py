from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias

from .messages import AssistantMessage, Message, ToolResultMessage, UserMessage
from .tools.registry import ToolDeclaration


@dataclass(frozen=True, slots=True)
class ModelToolResult:
    """The tool result fields visible to a model provider."""

    tool_call_id: str
    content: str
    is_error: bool


ModelMessage: TypeAlias = UserMessage | AssistantMessage | ModelToolResult


@dataclass(frozen=True, slots=True)
class ModelRequest:
    system_prompt: str
    messages: tuple[ModelMessage, ...]
    tools: tuple[ToolDeclaration, ...]


@dataclass(frozen=True, slots=True)
class ModelReply:
    """The terminal event containing one complete assistant message."""

    message: AssistantMessage
    stop_reason: Literal["stop", "tool_calls", "length"] = "stop"


@dataclass(frozen=True, slots=True)
class ModelTextDelta:
    """A text fragment emitted before the complete reply."""

    text: str


ModelEvent: TypeAlias = ModelReply | ModelTextDelta


class ModelProvider(Protocol):
    def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        """Yield optional text fragments, then one terminal reply, and finish."""
        ...


def _project_message(message: Message) -> ModelMessage:
    if isinstance(message, ToolResultMessage):
        return ModelToolResult(message.tool_call_id, message.content, message.is_error)
    return message


def _build_request(
    system_prompt: str, history: Sequence[Message], tools: tuple[ToolDeclaration, ...]
) -> ModelRequest:
    messages = tuple(_project_message(message) for message in history)
    return ModelRequest(system_prompt, messages, tools)
