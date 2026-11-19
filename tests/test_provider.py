import json
import unittest
from collections.abc import AsyncIterator, Callable
from importlib.util import find_spec
from typing import Any

if find_spec("openai") is None:
    raise unittest.SkipTest(
        "Install the chat-completions extra to run the provider tests"
    )

import httpx
from openai import AsyncOpenAI
from pydantic import BaseModel

from tuagent import (
    Agent,
    AssistantMessage,
    ModelReply,
    ModelRequest,
    ModelTextDelta,
    RunEvent,
    ToolCall,
    ToolContext,
    ToolOutput,
    ToolResultMessage,
    UserMessage,
)
from tuagent.providers import ChatCompletionsProtocolError, ChatCompletionsProvider


def _chunk(delta: dict[str, Any], finish: str | None = None) -> bytes:
    return _frame([{"index": 0, "delta": delta, "finish_reason": finish}])


def _frame(choices: list[dict[str, Any]]) -> bytes:
    payload = json.dumps(
        {
            "id": "completion-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "test-model",
            "choices": choices,
        }
    )
    return f"data: {payload}\n\n".encode()


def _sse(*frames: bytes, done: bool = True) -> httpx.Response:
    content = b"".join(frames)
    if done:
        content += b"data: [DONE]\n\n"
    return httpx.Response(
        200, headers={"content-type": "text/event-stream"}, content=content
    )


class _BrokenStream(httpx.AsyncByteStream):
    def __init__(self, *frames: bytes) -> None:
        self.frames = frames
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for frame in self.frames:
            yield frame
        raise httpx.ReadError("fixture connection reset")

    async def aclose(self) -> None:
        self.closed = True


class ChatCompletionsTests(unittest.IsolatedAsyncioTestCase):
    def client(self, handle: Callable[[httpx.Request], httpx.Response]) -> AsyncOpenAI:
        return AsyncOpenAI(
            api_key="test-key",
            base_url="https://chat.example.invalid/v1",
            max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
        )

    async def test_text_deltas_and_terminal_reply(self) -> None:
        requests: list[httpx.Request] = []

        def handle(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return _sse(
                _chunk({"role": "assistant", "content": ""}),
                _chunk({"content": "Hello"}),
                _chunk({"content": " there"}),
                _chunk({}, "stop"),
                _frame([]),
            )

        async with self.client(handle) as client:
            provider = ChatCompletionsProvider(
                "test-model",
                client=client,
                extra_body={"thinking": {"type": "disabled"}},
            )
            request = ModelRequest("", (UserMessage("Hello"),), ())
            events = [event async for event in provider.stream(request)]

        self.assertEqual(
            events,
            [
                ModelTextDelta("Hello"),
                ModelTextDelta(" there"),
                ModelReply(AssistantMessage("Hello there")),
            ],
        )
        (sent,) = requests
        self.assertEqual((sent.method, sent.url.path), ("POST", "/v1/chat/completions"))
        self.assertEqual(sent.headers["Authorization"], "Bearer test-key")
        self.assertEqual(
            json.loads(sent.content),
            {
                "model": "test-model",
                "stream": True,
                "thinking": {"type": "disabled"},
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

    async def test_agent_sends_back_interleaved_calls_with_text_and_original_ids(
        self,
    ) -> None:
        class Args(BaseModel):
            key: str

        executions: list[tuple[str, str]] = []
        requests: list[dict[str, Any]] = []
        events: list[RunEvent] = []

        async def lookup(args: Args, ctx: ToolContext) -> ToolOutput:
            executions.append((ctx.tool_call_id, args.key))
            return ToolOutput(f"value:{args.key}", details={"local": "private"})

        def handle(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            requests.append(payload)
            if len(requests) == 1:
                return _sse(
                    _chunk(
                        {
                            "content": "Looking ",
                            "tool_calls": [
                                {
                                    "index": 1,
                                    "id": "call-B",
                                    "type": "function",
                                    "function": {
                                        "name": "lookup",
                                        "arguments": '{"key":',
                                    },
                                },
                            ],
                        }
                    ),
                    _chunk(
                        {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call-A",
                                    "type": "function",
                                    "function": {
                                        "name": "lookup",
                                        "arguments": '{ "key": ',
                                    },
                                },
                                {"index": 1, "function": {"arguments": ' "B"'}},
                            ]
                        }
                    ),
                    _chunk(
                        {
                            "content": "up.",
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": '"A" }'}},
                                {"index": 1, "function": {"arguments": "}"}},
                            ],
                        }
                    ),
                    _chunk({}, "tool_calls"),
                )
            returned_values = [
                message["content"]
                for message in payload["messages"]
                if message["role"] == "tool"
            ]
            return _sse(_chunk({"content": " and ".join(returned_values)}, "stop"))

        async with self.client(handle) as client:
            agent = Agent(
                ChatCompletionsProvider("test-model", client=client),
                system_prompt="Use the lookup tool.",
            )
            agent.tools.register("lookup", "Look up a value", Args, lookup)
            result = await agent.run("Look up A and B", on_event=events.append)

        self.assertEqual(
            (result.status, result.turns, result.text),
            ("completed", 2, "value:A and value:B"),
        )
        self.assertEqual(executions, [("call-A", "A"), ("call-B", "B")])
        self.assertEqual(
            [
                event.delta
                for event in events
                if event.kind == "text_delta" and event.turn == 1
            ],
            ["Looking ", "up."],
        )
        first, second = requests
        prefix = [
            {"role": "system", "content": "Use the lookup tool."},
            {"role": "user", "content": "Look up A and B"},
        ]
        self.assertEqual(first["messages"], prefix)
        self.assertEqual(
            first["tools"],
            [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "description": "Look up a value",
                        "parameters": Args.model_json_schema(),
                    },
                }
            ],
        )
        self.assertEqual(second["tools"], first["tools"])
        self.assertEqual(
            second["messages"],
            [
                *prefix,
                {
                    "role": "assistant",
                    "content": "Looking up.",
                    "tool_calls": [
                        {
                            "id": "call-A",
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "arguments": '{ "key": "A" }',
                            },
                        },
                        {
                            "id": "call-B",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": '{"key": "B"}'},
                        },
                    ],
                },
                {"role": "tool", "tool_call_id": "call-A", "content": "value:A"},
                {"role": "tool", "tool_call_id": "call-B", "content": "value:B"},
            ],
        )

    async def test_length_preserves_tool_id_and_skips_execution(self) -> None:
        class Args(BaseModel):
            pass

        calls: list[str] = []
        requests: list[dict[str, Any]] = []
        chunk = _chunk(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "cut-1",
                        "type": "function",
                        "function": {"name": "work", "arguments": "{}"},
                    },
                ]
            }
        )

        async def work(args: Args, ctx: ToolContext) -> ToolOutput:
            calls.append(ctx.tool_call_id)
            return ToolOutput("executed")

        def handle(request: httpx.Request) -> httpx.Response:
            requests.append(json.loads(request.content))
            if len(requests) == 1:
                return _sse(chunk, _chunk({}, "length"))
            return _sse(_chunk({"content": "Recovered"}, "stop"))

        async with self.client(handle) as client:
            agent = Agent(ChatCompletionsProvider("test-model", client=client))
            agent.tools.register("work", "Record work", Args, work)
            result = await agent.run("Do the work")

        self.assertEqual((result.status, result.text), ("completed", "Recovered"))
        self.assertEqual(calls, [])
        self.assertEqual(
            result.messages[1],
            AssistantMessage(tool_calls=(ToolCall("cut-1", "work", "{}"),)),
        )
        receipt = result.messages[2]
        assert isinstance(receipt, ToolResultMessage)
        self.assertEqual(
            (receipt.tool_call_id, receipt.error_code), ("cut-1", "truncated")
        )
        self.assertEqual(
            requests[1]["messages"][-1],
            {
                "role": "tool",
                "tool_call_id": "cut-1",
                "content": f"Tool error: {receipt.content}",
            },
        )

    async def test_disconnect_leaves_complete_looking_drafts_unexecuted(self) -> None:
        class Args(BaseModel):
            pass

        chunk = _chunk(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "draft-1",
                        "type": "function",
                        "function": {"name": "work", "arguments": "{}"},
                    },
                ]
            }
        )
        broken = _BrokenStream(chunk, _chunk({}, "tool_calls"))
        for response, error in (
            (_sse(chunk, done=False), "model_stream_incomplete"),
            (
                httpx.Response(
                    200, stream=broken, headers={"content-type": "text/event-stream"}
                ),
                "model_error",
            ),
        ):
            with self.subTest(error=error):
                calls: list[str] = []

                async def work(args: Args, ctx: ToolContext) -> ToolOutput:
                    calls.append(ctx.tool_call_id)
                    return ToolOutput("executed")

                async with self.client(lambda _: response) as client:
                    agent = Agent(ChatCompletionsProvider("test-model", client=client))
                    agent.tools.register("work", "Record work", Args, work)
                    result = await agent.run("Do the work")

                self.assertEqual((result.status, result.error_code), ("failed", error))
                self.assertEqual(result.messages, (UserMessage("Do the work"),))
                self.assertEqual(calls, [])
        self.assertTrue(broken.closed)

    async def test_missing_call_id_is_a_protocol_error(self) -> None:
        chunk = _chunk(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "type": "function",
                        "function": {"name": "work", "arguments": "{}"},
                    },
                ]
            },
            "tool_calls",
        )
        async with self.client(lambda _: _sse(chunk)) as client:
            provider = ChatCompletionsProvider("test-model", client=client)
            with self.assertRaisesRegex(ChatCompletionsProtocolError, "missing its ID"):
                _ = [
                    event
                    async for event in provider.stream(
                        ModelRequest("", (UserMessage("work"),), ())
                    )
                ]
