import asyncio
import unittest
from collections.abc import AsyncIterator

from pydantic import BaseModel

from fakes import ScriptedModel
from tuagent import (
    Agent,
    AssistantMessage,
    ModelEvent,
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


class AgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_direct_reply_with_only_a_provider(self) -> None:
        reply = AssistantMessage("Hello")
        provider = ScriptedModel(ModelReply(reply))
        agent = Agent(provider)

        result = await agent.run("Hi")

        self.assertEqual(
            provider.requests, [ModelRequest("", (UserMessage("Hi"),), ())]
        )
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.turns, 1)
        self.assertEqual(result.text, "Hello")
        self.assertIsNone(result.error_code)
        self.assertEqual(result.messages, (UserMessage("Hi"), reply))
        self.assertEqual(agent.history, result.messages)

    async def test_final_reply_uses_tool_output_under_either_registered_name(
        self,
    ) -> None:
        class LookupArgs(BaseModel):
            key: str

        for name, return_value in (
            ("lookup", "first value"),
            ("lookup", "changed value"),
            ("renamed_lookup", "first value"),
        ):
            with self.subTest(name=name, return_value=return_value):
                calls: list[tuple[LookupArgs, ToolContext]] = []
                app_context = object()
                private_details = object()

                async def lookup(args: LookupArgs, ctx: ToolContext) -> ToolOutput:
                    calls.append((args, ctx))
                    return ToolOutput(return_value, details=private_details)

                def answer_from_result(request: ModelRequest) -> ModelReply:
                    result = request.messages[-1]
                    assert isinstance(result, ModelToolResult)
                    return ModelReply(AssistantMessage(f"Found: {result.content}"))

                registry = ToolRegistry()
                registry.register(name, "Look up a value", LookupArgs, lookup)
                user = UserMessage("Look up item")
                call = ToolCall("lookup-1", name, '{"key": "item"}')
                assistant = AssistantMessage("I'll look it up.", (call,))
                model = ScriptedModel(
                    ModelReply(assistant, "tool_calls"), answer_from_result
                )
                agent = Agent(
                    model,
                    system_prompt="Use the lookup tool.",
                    tools=registry,
                    context=app_context,
                )

                result = await agent.run(user.content)

                self.assertEqual(result.status, "completed")
                self.assertEqual(result.turns, 2)
                self.assertEqual(len(model.requests), 2)
                self.assertEqual(len(calls), 1)
                args, ctx = calls[0]
                self.assertEqual(args, LookupArgs(key="item"))
                self.assertEqual(ctx.tool_call_id, call.id)
                self.assertIs(ctx.context, app_context)
                first, second = model.requests
                self.assertEqual(first.messages, (user,))
                self.assertEqual(first.system_prompt, "Use the lookup tool.")
                self.assertEqual(second.system_prompt, first.system_prompt)
                self.assertEqual(second.tools, first.tools)
                (declaration,) = first.tools
                self.assertEqual(declaration.name, name)
                self.assertEqual(declaration.description, "Look up a value")
                self.assertEqual(declaration.parameters, LookupArgs.model_json_schema())
                self.assertEqual(
                    second.messages,
                    (user, assistant, ModelToolResult(call.id, return_value, False)),
                )
                self.assertEqual(result.text, f"Found: {return_value}")
                self.assertEqual(
                    result.messages,
                    (
                        user,
                        assistant,
                        ToolResultMessage(
                            call.id, return_value, details=private_details
                        ),
                        AssistantMessage(f"Found: {return_value}"),
                    ),
                )

    async def test_sequential_runs_share_session_history(self) -> None:
        first_reply = AssistantMessage("Your name is Tu.")
        second_reply = AssistantMessage("Tu")
        model = ScriptedModel(ModelReply(first_reply), ModelReply(second_reply))
        agent = Agent(model)

        first = await agent.run("My name is Tu.")
        second = await agent.run("What is my name?")

        self.assertEqual(first.messages, (UserMessage("My name is Tu."), first_reply))
        self.assertEqual(
            second.messages, (UserMessage("What is my name?"), second_reply)
        )
        self.assertEqual(
            model.requests[1].messages, first.messages + (second.messages[0],)
        )
        self.assertEqual(agent.history, first.messages + second.messages)

    async def test_turn_limit_preserves_the_last_tool_result(self) -> None:
        class CountArgs(BaseModel):
            count: int

        executions: list[int] = []

        async def count(args: CountArgs, ctx: ToolContext) -> ToolOutput:
            self.assertIsNone(ctx.context)
            executions.append(args.count)
            return ToolOutput(str(args.count))

        def request_next_call(request: ModelRequest) -> ModelReply:
            value = len(request.messages) // 2 + 1
            call = ToolCall(f"count-{value}", "count", f'{{"count": {value}}}')
            return ModelReply(AssistantMessage(tool_calls=(call,)), "tool_calls")

        model = ScriptedModel(*([request_next_call] * 3))
        agent = Agent(model, max_turns=2)
        agent.tools.register("count", "Record a count", CountArgs, count)

        result = await agent.run("Keep counting")

        self.assertEqual(result.status, "turn_limit")
        self.assertEqual(result.error_code, "turn_limit")
        self.assertEqual(result.turns, 2)
        self.assertIsNone(result.text)
        self.assertEqual(len(model.requests), 2)
        self.assertEqual(executions, [1, 2])
        self.assertEqual(
            model.requests[1].messages[-1], ModelToolResult("count-1", "1", False)
        )
        self.assertEqual(result.messages[-1], ToolResultMessage("count-2", "2"))
        self.assertEqual(agent.history, result.messages)

    async def test_text_reply_completes_on_the_last_allowed_turn(self) -> None:
        provider = ScriptedModel(ModelReply(AssistantMessage("done")))
        result = await Agent(provider, max_turns=1).run("start")

        self.assertEqual(
            (result.status, result.turns, result.text), ("completed", 1, "done")
        )

    def test_turn_limits_must_be_positive_integers(self) -> None:
        for limit in (0, -1, True, 1.5):
            with (
                self.subTest(limit=limit),
                self.assertRaisesRegex(ValueError, "positive integer"),
            ):
                Agent(ScriptedModel(), max_turns=limit)

    async def test_incomplete_model_stream_keeps_the_user_message(self) -> None:
        model = ScriptedModel(None, ModelReply(AssistantMessage("Recovered")))
        agent = Agent(model)

        failed = await agent.run("First")
        recovered = await agent.run("Second")

        self.assertEqual(
            (failed.status, failed.error_code), ("failed", "model_stream_incomplete")
        )
        self.assertEqual(failed.messages, (UserMessage("First"),))
        self.assertEqual(recovered.status, "completed")
        self.assertEqual(
            model.requests[1].messages, (UserMessage("First"), UserMessage("Second"))
        )

    async def test_concurrent_run_leaves_the_active_run_in_charge(self) -> None:
        started = asyncio.Event()
        finish = asyncio.Event()

        class WaitingModel:
            async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
                started.set()
                await finish.wait()
                yield ModelReply(AssistantMessage("Finished"))

        agent = Agent(WaitingModel())
        task = asyncio.create_task(agent.run("First"))
        try:
            await started.wait()
            with self.assertRaisesRegex(RuntimeError, "run in progress"):
                await agent.run("Second")
        finally:
            finish.set()
            result = await task

        self.assertEqual(
            agent.history, (UserMessage("First"), AssistantMessage("Finished"))
        )
        self.assertEqual(result.status, "completed")
