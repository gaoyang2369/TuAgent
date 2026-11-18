import unittest

from pydantic import BaseModel

from fakes import ScriptedModel
from tuagent import (
    Agent,
    AssistantMessage,
    ModelReply,
    ModelRequest,
    ModelToolResult,
    ToolCall,
    ToolContext,
    ToolOutput,
    ToolRegistry,
    ToolResultMessage,
    UserMessage,
)


class ToolTests(unittest.IsolatedAsyncioTestCase):
    def test_duplicate_name_is_rejected(self) -> None:
        class Args(BaseModel):
            pass

        async def handler(args: Args, ctx: ToolContext) -> ToolOutput:
            return ToolOutput("result")

        registry = ToolRegistry()
        registry.register("example", "First registration", Args, handler)

        with self.assertRaisesRegex(ValueError, "already registered"):
            registry.register("example", "Duplicate", Args, handler)

    async def test_invalid_arguments_and_unknown_tools_do_not_execute(self) -> None:
        class Args(BaseModel):
            quantity: int

        executions = []

        async def handler(args: Args, ctx: ToolContext) -> ToolOutput:
            executions.append(args.quantity)
            return ToolOutput("result")

        registry = ToolRegistry()
        registry.register("example", "An example tool", Args, handler)

        for call, code, expected_content in (
            (
                ToolCall("bad-json", "example", '{"quantity":'),
                "invalid_json",
                "valid JSON",
            ),
            (
                ToolCall("bad-value", "example", '{"quantity":"many"}'),
                "invalid_arguments",
                "quantity",
            ),
            (
                ToolCall("missing-tool", "other", '{"quantity":1}'),
                "unknown_tool",
                "other",
            ),
        ):
            with self.subTest(code=code):
                assistant = AssistantMessage(tool_calls=(call,))
                model = ScriptedModel(
                    ModelReply(assistant), ModelReply(AssistantMessage("Recovered"))
                )
                result = await Agent(model, tools=registry).run("Try a tool")

                self.assertEqual(result.status, "completed")
                self.assertEqual(len(model.requests), 2)
                self.assertEqual(executions, [])
                receipt = result.messages[2]
                assert isinstance(receipt, ToolResultMessage)
                self.assertEqual(receipt.tool_call_id, call.id)
                self.assertTrue(receipt.is_error)
                self.assertEqual(receipt.error_code, code)
                self.assertIn(expected_content, receipt.content)
                self.assertEqual(
                    model.requests[1].messages,
                    (
                        UserMessage("Try a tool"),
                        assistant,
                        ModelToolResult(call.id, receipt.content, True),
                    ),
                )

    async def test_tool_batch_returns_ordered_results_after_a_handler_error(
        self,
    ) -> None:
        class Args(BaseModel):
            value: str

        calls: list[str] = []

        async def handler(args: Args, ctx: ToolContext) -> ToolOutput:
            calls.append(ctx.tool_call_id)
            if args.value == "raise":
                raise RuntimeError("internal error detail")
            if args.value == "report":
                return ToolOutput("Try another value", is_error=True)
            return ToolOutput(args.value)

        registry = ToolRegistry()
        registry.register("example", "An example tool", Args, handler)
        requests = (
            ToolCall("t1", "example", '{"value":"raise"}'),
            ToolCall("t2", "example", '{"value":"report"}'),
            ToolCall("t3", "example", '{"value":"ok"}'),
        )
        assistant = AssistantMessage("I'll try these values.", requests)
        model = ScriptedModel(
            ModelReply(assistant), ModelReply(AssistantMessage("Done"))
        )

        result = await Agent(model, tools=registry).run("Try each value")

        self.assertEqual(result.status, "completed")
        self.assertEqual(calls, ["t1", "t2", "t3"])
        self.assertEqual(model.requests[1].messages[1], assistant)
        self.assertEqual(
            model.requests[1].messages[2:],
            (
                ModelToolResult("t1", "Tool 'example' raised an exception.", True),
                ModelToolResult("t2", "Try another value", True),
                ModelToolResult("t3", "ok", False),
            ),
        )

    async def test_invalid_call_ids_reject_the_whole_batch(self) -> None:
        class Args(BaseModel):
            pass

        calls = []

        async def handler(args: Args, ctx: ToolContext) -> ToolOutput:
            calls.append(ctx.tool_call_id)
            return ToolOutput("result")

        registry = ToolRegistry()
        registry.register("example", "An example tool", Args, handler)

        for bad_id in ("", "call-1"):
            with self.subTest(bad_id=bad_id):
                assistant = AssistantMessage(
                    tool_calls=(
                        ToolCall("call-1", "example", "{}"),
                        ToolCall(bad_id, "example", "{}"),
                    )
                )
                model = ScriptedModel(ModelReply(assistant))
                agent = Agent(model, tools=registry)

                result = await agent.run("Try multiple calls")

                self.assertEqual(result.status, "failed")
                self.assertEqual(result.error_code, "invalid_tool_call_id")
                self.assertEqual(calls, [])
                self.assertEqual(agent.history, (UserMessage("Try multiple calls"),))

    async def test_registration_during_a_run_only_applies_to_later_runs(self) -> None:
        class Args(BaseModel):
            pass

        registry = ToolRegistry()
        executions = []
        call = ToolCall("added-1", "added", "{}")
        tool_reply = ModelReply(AssistantMessage(tool_calls=(call,)))

        async def handler(args: Args, ctx: ToolContext) -> ToolOutput:
            executions.append(ctx.tool_call_id)
            return ToolOutput("added tool result")

        def add_tool(request: ModelRequest) -> ModelReply:
            registry.register("added", "A newly registered tool", Args, handler)
            return tool_reply

        done = ModelReply(AssistantMessage("Done"))
        model = ScriptedModel(add_tool, done, tool_reply, done)
        agent = Agent(model, tools=registry)

        first = await agent.run("Try the new tool")
        self.assertEqual(first.status, "completed")
        self.assertEqual(executions, [])
        self.assertEqual([request.tools for request in model.requests], [(), ()])
        self.assertEqual(
            model.requests[1].messages[-1],
            ModelToolResult(call.id, "Unknown tool 'added'.", True),
        )

        second = await agent.run("Try again")
        self.assertEqual(second.status, "completed")
        self.assertEqual(executions, [call.id])
        self.assertEqual([tool.name for tool in model.requests[2].tools], ["added"])
        self.assertEqual(
            model.requests[3].messages[-1],
            ModelToolResult(call.id, "added tool result", False),
        )
