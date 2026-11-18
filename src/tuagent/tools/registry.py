from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from pydantic import BaseModel


@dataclass(frozen=True, slots=True)
class ToolContext:
    tool_call_id: str
    context: Any = None


@dataclass(frozen=True, slots=True)
class ToolOutput:
    content: str
    details: Any = None
    is_error: bool = False


@dataclass(frozen=True, slots=True)
class ToolDeclaration:
    name: str
    description: str
    parameters: dict[str, Any]


Args = TypeVar("Args", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class _RegisteredTool(Generic[Args]):
    declaration: ToolDeclaration
    args_model: type[Args]
    handler: Callable[[Args, ToolContext], Awaitable[ToolOutput]]


class ToolRegistry:
    """Register async handlers and their Pydantic argument models."""

    def __init__(self) -> None:
        self._tools: dict[str, _RegisteredTool[Any]] = {}

    def register(
        self,
        name: str,
        description: str,
        args_model: type[Args],
        handler: Callable[[Args, ToolContext], Awaitable[ToolOutput]],
    ) -> None:
        if name in self._tools:
            raise ValueError(f"Tool {name!r} is already registered")

        declaration = ToolDeclaration(name, description, args_model.model_json_schema())
        self._tools[name] = _RegisteredTool(declaration, args_model, handler)

    def _snapshot(self) -> dict[str, _RegisteredTool[Any]]:
        return self._tools.copy()
