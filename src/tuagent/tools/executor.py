from copy import deepcopy
from typing import Any

from pydantic import ValidationError

from ..messages import ToolCall, ToolResultMessage
from .registry import ToolContext, ToolDeclaration, ToolOutput, ToolRegistry


class ToolExecutor:
    """Execute calls against a snapshot of a tool registry."""

    def __init__(self, registry: ToolRegistry, *, context: Any = None) -> None:
        self._tools = registry._snapshot()
        self._context = context

    def declarations(self) -> tuple[ToolDeclaration, ...]:
        return tuple(deepcopy(tool.declaration) for tool in self._tools.values())

    async def execute(self, call: ToolCall) -> ToolResultMessage:
        tool = self._tools.get(call.name)
        if tool is None:
            return _tool_error(call, "unknown_tool", f"Unknown tool {call.name!r}.")

        try:
            args = tool.args_model.model_validate_json(call.arguments)
        except ValidationError as error:
            issues = error.errors(
                include_url=False, include_context=False, include_input=False
            )
            if issues[0]["type"] in {"json_invalid", "json_type"}:
                return _tool_error(
                    call, "invalid_json", "Tool arguments must be valid JSON."
                )

            messages = []
            for issue in issues:
                field = ".".join(map(str, issue["loc"])) or "arguments"
                messages.append(f"{field}: {issue['msg']}")
            return _tool_error(call, "invalid_arguments", "\n".join(messages))
        except Exception:
            return _tool_error(
                call, "tool_validation_error", "Tool argument validation failed."
            )

        try:
            output = await tool.handler(args, ToolContext(call.id, self._context))
        except Exception:
            return _tool_error(
                call, "tool_exception", f"Tool {call.name!r} raised an exception."
            )

        if not isinstance(output, ToolOutput) or not isinstance(output.content, str):
            return _tool_error(
                call,
                "invalid_tool_output",
                "Tool must return ToolOutput with text content.",
            )

        return ToolResultMessage(
            tool_call_id=call.id,
            content=output.content,
            is_error=output.is_error,
            error_code="tool_reported_error" if output.is_error else None,
            details=output.details,
        )


def _tool_error(call: ToolCall, code: str, content: str) -> ToolResultMessage:
    return ToolResultMessage(call.id, content, is_error=True, error_code=code)
