from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, TypedDict

from ..messages import AssistantMessage, ToolCall, UserMessage
from ..model import ModelEvent, ModelMessage, ModelReply, ModelRequest, ModelTextDelta
from ..tools import ToolDeclaration

if TYPE_CHECKING:
    from openai import AsyncOpenAI
    from openai.types.chat import (
        ChatCompletionAssistantMessageParam,
        ChatCompletionFunctionToolParam,
        ChatCompletionMessageParam,
    )
    from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall


class ChatCompletionsProtocolError(ValueError):
    """The endpoint sent a stream this adapter cannot represent."""


class _RequestOptions(TypedDict, total=False):
    tools: list[ChatCompletionFunctionToolParam]


class ChatCompletionsProvider:
    """Stream text and function calls using a caller-managed async OpenAI client.

    Install the ``chat-completions`` extra to use the OpenAI SDK. Configure the
    endpoint and authentication on the client and keep it open while runs consume
    streams. Each response stream is closed by the provider.
    """

    def __init__(
        self,
        model: str,
        *,
        client: AsyncOpenAI,
        extra_body: Mapping[str, object] | None = None,
    ) -> None:
        if extra_body is not None and {
            "model",
            "messages",
            "tools",
            "stream",
        }.intersection(extra_body):
            raise ValueError(
                "extra_body cannot override the model, messages, tools, or stream"
            )
        self._model = model
        self._client = client
        self._extra_body = dict(extra_body) if extra_body is not None else None

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        messages: list[ChatCompletionMessageParam] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.extend(_serialize_message(message) for message in request.messages)
        options: _RequestOptions = {}
        if request.tools:
            options["tools"] = [_serialize_tool(tool) for tool in request.tools]

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            stream=True,
            extra_body=self._extra_body,
            **options,
        )
        text: list[str] = []
        calls: dict[int, _ToolCallParts] = {}
        finish_reason: Literal["stop", "tool_calls", "length"] | None = None

        async with response:
            async for chunk in response:
                if not chunk.choices:
                    continue
                if len(chunk.choices) != 1 or chunk.choices[0].index != 0:
                    raise ChatCompletionsProtocolError(
                        "Expected a single streamed choice at index 0"
                    )
                if finish_reason is not None:
                    raise ChatCompletionsProtocolError(
                        "A choice followed the finish reason"
                    )

                choice = chunk.choices[0]
                delta = choice.delta
                if delta.function_call is not None:
                    raise ChatCompletionsProtocolError(
                        "Expected tool_calls instead of function_call"
                    )
                for call in delta.tool_calls or ():
                    if type(call.index) is not int or call.index < 0:
                        raise ChatCompletionsProtocolError("Invalid tool call index")
                    if call.index not in calls:
                        calls[call.index] = _ToolCallParts()
                    calls[call.index].append(call)
                for fragment in (delta.content, delta.refusal):
                    if fragment:
                        text.append(fragment)
                        yield ModelTextDelta(fragment)

                reason = choice.finish_reason
                if reason is not None:
                    if reason not in ("stop", "tool_calls", "length"):
                        raise ChatCompletionsProtocolError(
                            f"Unsupported finish reason {reason!r}"
                        )
                    finish_reason = reason

        if finish_reason is None:
            return
        tool_calls = tuple(calls[index].to_call() for index in sorted(calls))
        if finish_reason == "tool_calls" and not tool_calls:
            raise ChatCompletionsProtocolError(
                "The tool_calls finish reason requires a tool call"
            )
        yield ModelReply(AssistantMessage("".join(text), tool_calls), finish_reason)


def _serialize_message(message: ModelMessage) -> ChatCompletionMessageParam:
    if isinstance(message, UserMessage):
        return {"role": "user", "content": message.content}
    if isinstance(message, AssistantMessage):
        payload: ChatCompletionAssistantMessageParam = {
            "role": "assistant",
            "content": message.content,
        }
        if message.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                for call in message.tool_calls
            ]
        return payload
    content = f"Tool error: {message.content}" if message.is_error else message.content
    return {"role": "tool", "tool_call_id": message.tool_call_id, "content": content}


def _serialize_tool(tool: ToolDeclaration) -> ChatCompletionFunctionToolParam:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


@dataclass(slots=True)
class _ToolCallParts:
    id: str | None = None
    name: str | None = None
    arguments: list[str] = field(default_factory=list)

    def append(self, call: ChoiceDeltaToolCall) -> None:
        if call.type not in (None, "function"):
            raise ChatCompletionsProtocolError("Expected a function tool call")
        self.id = _merge_metadata("ID", self.id, call.id)
        if call.function is not None:
            self.name = _merge_metadata("function name", self.name, call.function.name)
            if call.function.arguments is not None:
                self.arguments.append(call.function.arguments)

    def to_call(self) -> ToolCall:
        if (
            self.id is None
            or not self.id.strip()
            or self.name is None
            or not self.name.strip()
        ):
            raise ChatCompletionsProtocolError(
                "Tool call is missing its ID or function name"
            )
        return ToolCall(self.id, self.name, "".join(self.arguments))


def _merge_metadata(
    field_name: str, prior: str | None, value: str | None
) -> str | None:
    if not value:
        return prior
    if prior is not None and prior != value:
        raise ChatCompletionsProtocolError(f"Conflicting tool call {field_name}")
    return value
