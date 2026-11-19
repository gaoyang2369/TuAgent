import asyncio
import unittest
from collections.abc import AsyncIterator
from dataclasses import FrozenInstanceError
from typing import cast

from pydantic import BaseModel

from fakes import ScriptedModel
from tuagent import (
    Agent,
    AssistantMessage,
    ModelEvent,
    ModelReply,
    ModelRequest,
    ModelTextDelta,
    ModelToolResult,
    RunEvent,
    ToolCall,
    ToolContext,
    ToolOutput,
    ToolResultMessage,
    UserMessage,
)


class LifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_caller_cancel_after_tool_is_released_prevents_next_dispatch(
        self,
    ) -> None:
        class Args(BaseModel):
            pass

        first_call = ToolCall("first-1", "first", "{}")
        second_call = ToolCall("second-1", "second", "{}")
        assistant = AssistantMessage(tool_calls=(first_call, second_call))
        model = ScriptedModel(ModelReply(assistant, "tool_calls"))
        agent = Agent(model)
        waiting = asyncio.Event()
        ready = asyncio.Event()
        executions: list[str] = []
        events: list[RunEvent] = []

        async def first(args: Args, ctx: ToolContext) -> ToolOutput:
            executions.append(ctx.tool_call_id)
            waiting.set()
            await ready.wait()
            return ToolOutput("confirmed")

        async def second(args: Args, ctx: ToolContext) -> ToolOutput:
            executions.append(ctx.tool_call_id)
            return ToolOutput("second result")

        agent.tools.register("first", "Return a result after release", Args, first)
        agent.tools.register("second", "Record a call", Args, second)
        task = asyncio.create_task(agent.run("Run both", on_event=events.append))
        try:
            await asyncio.wait_for(waiting.wait(), timeout=5)
        finally:
            ready.set()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        self.assertEqual(executions, [first_call.id])
        self.assertEqual(len(model.requests), 1)
        self.assertEqual(
            agent.history[:-1],
            (
                UserMessage("Run both"),
                assistant,
                ToolResultMessage(first_call.id, "confirmed"),
            ),
        )
        skipped = agent.history[-1]
        assert isinstance(skipped, ToolResultMessage)
        self.assertEqual(
            (skipped.tool_call_id, skipped.is_error, skipped.error_code),
            (second_call.id, True, "not_started"),
        )
        self.assertEqual(
            [event.tool_call_id for event in events if event.kind == "tool_start"],
            [first_call.id],
        )
        self.assertEqual(
            [event.status for event in events if event.kind == "run_end"], ["cancelled"]
        )

    async def test_task_cancellation_keeps_the_first_result_and_pairs_the_second(
        self,
    ) -> None:
        class Args(BaseModel):
            pass

        first_call = ToolCall("first-1", "first", "{}")
        second_call = ToolCall("second-1", "second", "{}")
        assistant = AssistantMessage("Running both tools.", (first_call, second_call))
        provider = ScriptedModel(ModelReply(assistant, "tool_calls"))
        agent = Agent(provider)
        waiting = asyncio.Event()
        release = asyncio.Event()
        cleaning_up = asyncio.Event()
        finish_cleanup = asyncio.Event()
        cleanup_finished = asyncio.Event()
        calls: list[str] = []
        events: list[RunEvent] = []
        first_seen = []

        async def first(args: Args, ctx: ToolContext) -> ToolOutput:
            calls.append(ctx.tool_call_id)
            first_seen.extend(agent.history)
            return ToolOutput("saved")

        async def second(args: Args, ctx: ToolContext) -> ToolOutput:
            calls.append(ctx.tool_call_id)
            waiting.set()
            try:
                await release.wait()
            finally:
                cleaning_up.set()
                await finish_cleanup.wait()
                cleanup_finished.set()
            return ToolOutput("late result")

        agent.tools.register("first", "Finish immediately", Args, first)
        agent.tools.register("second", "Wait for a signal", Args, second)
        task = asyncio.create_task(agent.run("Run both", on_event=events.append))
        try:
            await asyncio.wait_for(waiting.wait(), timeout=5)
            during_second = agent.history
        finally:
            task.cancel()
            try:
                await asyncio.wait_for(cleaning_up.wait(), timeout=5)
                task.cancel()
            finally:
                finish_cleanup.set()
            with self.assertRaises(asyncio.CancelledError):
                await task

        self.assertEqual(calls, [first_call.id, second_call.id])
        self.assertTrue(cleanup_finished.is_set())
        self.assertEqual(len(provider.requests), 1)
        prefix = (UserMessage("Run both"), assistant)
        self.assertEqual(tuple(first_seen), prefix)
        self.assertEqual(
            during_second, (*prefix, ToolResultMessage(first_call.id, "saved"))
        )
        self.assertEqual(agent.history[:-1], during_second)
        interrupted = agent.history[-1]
        assert isinstance(interrupted, ToolResultMessage)
        self.assertEqual(interrupted.tool_call_id, second_call.id)
        self.assertTrue(interrupted.is_error)
        self.assertEqual(interrupted.error_code, "outcome_unknown")
        self.assertEqual(
            [event.status for event in events if event.kind == "run_end"], ["cancelled"]
        )

    async def test_abort_settles_unstarted_calls_and_continues_without_reexecution(
        self,
    ) -> None:
        class Args(BaseModel):
            pass

        calls = tuple(
            ToolCall(call_id, "work", "{}") for call_id in ("first", "waiting", "last")
        )
        assistant = AssistantMessage("Running the batch.", calls)
        answer = AssistantMessage("Read the existing results.")
        model = ScriptedModel(ModelReply(assistant), ModelReply(answer))
        agent = Agent(model)
        waiting = asyncio.Event()
        executions: list[str] = []
        events: list[RunEvent] = []

        async def work(args: Args, ctx: ToolContext) -> ToolOutput:
            executions.append(ctx.tool_call_id)
            if ctx.tool_call_id == "waiting":
                waiting.set()
                await asyncio.Event().wait()
            return ToolOutput("first result")

        agent.tools.register("work", "Run one part of the batch", Args, work)
        task = asyncio.create_task(agent.run("Start the batch", on_event=events.append))
        try:
            await asyncio.wait_for(waiting.wait(), timeout=5)
        finally:
            agent.abort()
            agent.abort()
        cancelled = await asyncio.wait_for(task, timeout=5)

        self.assertEqual(
            (cancelled.status, cancelled.turns, cancelled.error_code),
            ("cancelled", 1, "cancelled"),
        )
        self.assertEqual(len(model.requests), 1)
        self.assertEqual(cancelled.messages, agent.history)
        receipts = cancelled.messages[2:]
        self.assertEqual(receipts[0], ToolResultMessage("first", "first result"))
        self.assertEqual(
            [
                (receipt.tool_call_id, receipt.is_error, receipt.error_code)
                for receipt in receipts
                if isinstance(receipt, ToolResultMessage)
            ],
            [
                ("first", False, None),
                ("waiting", True, "outcome_unknown"),
                ("last", True, "not_started"),
            ],
        )
        self.assertEqual(
            [event.tool_call_id for event in events if event.kind == "tool_start"],
            ["first", "waiting"],
        )
        self.assertEqual(
            [
                (event.tool_call_id, event.error_code)
                for event in events
                if event.kind == "tool_end"
            ],
            [("first", None), ("waiting", "outcome_unknown"), ("last", "not_started")],
        )

        resumed_events: list[RunEvent] = []
        resumed = await agent.continue_run(on_event=resumed_events.append)

        self.assertEqual(
            (resumed.status, resumed.turns, resumed.messages),
            ("completed", 1, (answer,)),
        )
        self.assertEqual(executions, ["first", "waiting"])
        self.assertEqual(len(model.requests), 2)
        self.assertEqual(
            model.requests[1].messages,
            (
                UserMessage("Start the batch"),
                assistant,
                *(
                    ModelToolResult(
                        receipt.tool_call_id, receipt.content, receipt.is_error
                    )
                    for receipt in receipts
                    if isinstance(receipt, ToolResultMessage)
                ),
            ),
        )
        self.assertEqual(
            [event.status for event in events if event.kind == "run_end"], ["cancelled"]
        )
        self.assertEqual(
            [event.status for event in resumed_events if event.kind == "run_end"],
            ["completed"],
        )
        self.assertNotEqual(events[0].run_id, resumed_events[0].run_id)
        self.assertEqual(resumed_events[0].sequence, 1)

    async def test_abort_closes_the_model_stream_and_allows_user_message_continuation(
        self,
    ) -> None:
        waiting = asyncio.Event()
        closed = asyncio.Event()
        requests: list[ModelRequest] = []
        events: list[RunEvent] = []

        class PausedModel:
            async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
                requests.append(request)
                if len(requests) == 1:
                    try:
                        yield ModelTextDelta("draft text")
                        waiting.set()
                        await asyncio.Event().wait()
                    finally:
                        await asyncio.sleep(0)
                        closed.set()
                yield ModelReply(AssistantMessage("Finished"))

        agent = Agent(PausedModel())
        agent.abort()
        task = asyncio.create_task(agent.run("Start", on_event=events.append))
        try:
            await asyncio.wait_for(waiting.wait(), timeout=5)
            with self.assertRaisesRegex(RuntimeError, "run in progress"):
                await agent.continue_run(on_event=events.append)
        finally:
            agent.abort()

        result = await asyncio.wait_for(task, timeout=5)
        self.assertTrue(closed.is_set())
        self.assertEqual(
            (result.status, result.turns, result.messages),
            ("cancelled", 1, (UserMessage("Start"),)),
        )
        self.assertEqual(
            [event.delta for event in events if event.kind == "text_delta"],
            ["draft text"],
        )
        self.assertEqual([event.kind for event in events].count("run_start"), 1)
        self.assertEqual([event.kind for event in events].count("run_end"), 1)

        resumed = await agent.continue_run()
        self.assertEqual(resumed.text, "Finished")
        self.assertEqual(requests[1].messages, (UserMessage("Start"),))

    async def test_observer_can_abort_at_assistant_commit_before_any_tool_starts(
        self,
    ) -> None:
        class Args(BaseModel):
            pass

        call = ToolCall("pending-1", "work", "{}")
        assistant = AssistantMessage(tool_calls=(call,))
        agent = Agent(ScriptedModel(ModelReply(assistant)))
        executions: list[str] = []
        events: list[RunEvent] = []

        async def work(args: Args, ctx: ToolContext) -> ToolOutput:
            executions.append(ctx.tool_call_id)
            return ToolOutput("work done")

        def observe(event: RunEvent) -> None:
            events.append(event)
            if event.message == assistant:
                agent.abort()

        agent.tools.register("work", "Record a call", Args, work)
        result = await agent.run("Start work", on_event=observe)

        self.assertEqual(result.status, "cancelled")
        self.assertEqual(executions, [])
        self.assertEqual(result.messages[:2], (UserMessage("Start work"), assistant))
        receipt = result.messages[-1]
        assert isinstance(receipt, ToolResultMessage)
        self.assertEqual(
            (receipt.tool_call_id, receipt.error_code), (call.id, "not_started")
        )
        self.assertEqual(
            [event.status for event in events if event.kind == "run_end"], ["cancelled"]
        )

    async def test_continue_rejects_empty_or_completed_history_without_starting_a_run(
        self,
    ) -> None:
        model = ScriptedModel(ModelReply(AssistantMessage("Done")))
        agent = Agent(model)
        events: list[RunEvent] = []

        with self.assertRaisesRegex(RuntimeError, "Continuation requires"):
            await agent.continue_run(on_event=events.append)
        await agent.run("Start")
        with self.assertRaisesRegex(RuntimeError, "Continuation requires"):
            await agent.continue_run(on_event=events.append)

        self.assertEqual(len(model.requests), 1)
        self.assertEqual(events, [])

    async def test_continue_rejects_unpaired_tool_results_before_requesting_model(
        self,
    ) -> None:
        calls = (ToolCall("first", "save", "{}"), ToolCall("second", "save", "{}"))
        assistant = AssistantMessage(tool_calls=calls)
        first = ToolResultMessage("first", "saved first")
        second = ToolResultMessage("second", "saved second")
        user = UserMessage("Save both")

        for history in (
            (user, assistant, first),
            (user, assistant, first, UserMessage("Try again")),
            (user, assistant, first, ToolResultMessage("other", "unexpected")),
            (user, assistant, second, first),
            (user, assistant, first, second, second),
            (user, first),
        ):
            with self.subTest(history=history):
                model = ScriptedModel(ModelReply(AssistantMessage("Recovered")))
                agent = Agent(model)
                events: list[RunEvent] = []
                # Seed histories the public runtime normally prevents.
                agent._history.extend(history)

                with self.assertRaisesRegex(RuntimeError, "Continuation requires"):
                    await agent.continue_run(on_event=events.append)

                self.assertEqual(model.requests, [])
                self.assertEqual(events, [])
                self.assertEqual(agent.history, history)

    async def test_disconnect_preserves_settled_history_and_continue_only_requests_model(
        self,
    ) -> None:
        class Args(BaseModel):
            pass

        calls = (ToolCall("save-1", "save", "{}"), ToolCall("save-2", "save", "{}"))
        assistant = AssistantMessage(tool_calls=calls)

        def disconnect(request: ModelRequest) -> ModelReply:
            raise ConnectionError("connection lost")

        for failed_step, code in (
            ((ModelTextDelta("uncommitted"),), "model_stream_incomplete"),
            (disconnect, "model_error"),
        ):
            with self.subTest(code=code):
                executions: list[str] = []
                events: list[RunEvent] = []

                async def save(args: Args, ctx: ToolContext) -> ToolOutput:
                    executions.append(ctx.tool_call_id)
                    return ToolOutput("stored")

                model = ScriptedModel(
                    ModelReply(assistant),
                    failed_step,
                    ModelReply(AssistantMessage("Recovered")),
                )
                agent = Agent(model)
                agent.tools.register("save", "Save a value", Args, save)

                failed = await agent.run("Save it", on_event=events.append)
                self.assertEqual(
                    (failed.status, failed.turns, failed.error_code),
                    ("failed", 2, code),
                )
                self.assertEqual(
                    agent.history,
                    (
                        UserMessage("Save it"),
                        assistant,
                        ToolResultMessage(calls[0].id, "stored"),
                        ToolResultMessage(calls[1].id, "stored"),
                    ),
                )
                self.assertEqual(failed.messages, agent.history)
                self.assertEqual(
                    [event.status for event in events if event.kind == "run_end"],
                    ["failed"],
                )

                resumed = await agent.continue_run()
                self.assertEqual(
                    (resumed.status, resumed.turns, resumed.text),
                    ("completed", 1, "Recovered"),
                )
                self.assertEqual(resumed.messages, (AssistantMessage("Recovered"),))
                self.assertEqual(len(model.requests), 3)
                self.assertEqual(model.requests[2], model.requests[1])
                self.assertEqual(executions, [call.id for call in calls])

    async def test_malformed_second_call_rejects_entire_batch(self) -> None:
        class Args(BaseModel):
            pass

        first = ToolCall("first", "save", "{}")
        for malformed_call, code in (
            (ToolCall("second", cast(str, ["save"]), "{}"), "invalid_tool_call_name"),
            (ToolCall("second", "save", cast(str, {})), "invalid_tool_call_arguments"),
            (cast(ToolCall, {}), "invalid_tool_call"),
        ):
            with self.subTest(code=code):
                executions: list[str] = []
                events: list[RunEvent] = []

                async def save(args: Args, ctx: ToolContext) -> ToolOutput:
                    executions.append(ctx.tool_call_id)
                    return ToolOutput("saved")

                assistant = AssistantMessage(tool_calls=(first, malformed_call))
                model = ScriptedModel(ModelReply(assistant, "tool_calls"))
                agent = Agent(model)
                agent.tools.register("save", "Save a value", Args, save)

                result = await agent.run("Save two values", on_event=events.append)

                self.assertEqual((result.status, result.error_code), ("failed", code))
                self.assertEqual(agent.history, (UserMessage("Save two values"),))
                self.assertEqual(executions, [])
                self.assertFalse(any(event.kind == "tool_start" for event in events))
                self.assertEqual(
                    [event.error_code for event in events if event.kind == "run_end"],
                    [code],
                )

    async def test_tool_calls_stop_reason_requires_at_least_one_call(self) -> None:
        model = ScriptedModel(ModelReply(AssistantMessage("No calls"), "tool_calls"))
        agent = Agent(model)

        result = await agent.run("Save a value")

        self.assertEqual(
            (result.status, result.error_code), ("failed", "invalid_stop_reason")
        )
        self.assertEqual(result.messages, (UserMessage("Save a value"),))
        self.assertEqual(agent.history, result.messages)
        self.assertEqual(len(model.requests), 1)

    async def test_length_skips_the_entire_batch_even_with_valid_json(self) -> None:
        class Args(BaseModel):
            value: int

        calls = (
            ToolCall("truncated-1", "work", '{"value":1}'),
            ToolCall("truncated-2", "work", '{"value":2}'),
        )
        assistant = AssistantMessage("Partial reply", calls)
        for limit in (1, 2):
            with self.subTest(limit=limit):
                executions: list[str] = []
                events: list[RunEvent] = []

                async def work(args: Args, ctx: ToolContext) -> ToolOutput:
                    executions.append(ctx.tool_call_id)
                    return ToolOutput(str(args.value))

                provider = ScriptedModel(
                    ModelReply(assistant, "length"),
                    ModelReply(AssistantMessage("Recovered")),
                )
                agent = Agent(provider, max_turns=limit)
                agent.tools.register("work", "Record a call", Args, work)

                result = await agent.run("Try both", on_event=events.append)

                self.assertEqual(result.turns, limit)
                self.assertEqual(
                    result.status, "turn_limit" if limit == 1 else "completed"
                )
                self.assertEqual(executions, [])
                self.assertEqual(
                    result.messages[:2], (UserMessage("Try both"), assistant)
                )
                receipts = result.messages[2:4]
                self.assertEqual(
                    [
                        (receipt.tool_call_id, receipt.is_error, receipt.error_code)
                        for receipt in receipts
                        if isinstance(receipt, ToolResultMessage)
                    ],
                    [(call.id, True, "truncated") for call in calls],
                )
                self.assertFalse(any(event.kind == "tool_start" for event in events))
                self.assertEqual(
                    [
                        event.tool_call_id
                        for event in events
                        if event.kind == "tool_end"
                    ],
                    [call.id for call in calls],
                )
                self.assertEqual(
                    [event.status for event in events if event.kind == "run_end"],
                    [result.status],
                )
                if limit == 2:
                    self.assertEqual(
                        provider.requests[1].messages,
                        (
                            UserMessage("Try both"),
                            assistant,
                            *(
                                ModelToolResult(
                                    receipt.tool_call_id, receipt.content, True
                                )
                                for receipt in receipts
                                if isinstance(receipt, ToolResultMessage)
                            ),
                        ),
                    )

    async def test_text_only_length_returns_partial_text_as_incomplete(self) -> None:
        assistant = AssistantMessage("Partial")
        model = ScriptedModel((ModelTextDelta("Par"), ModelReply(assistant, "length")))
        events: list[RunEvent] = []

        result = await Agent(model).run("Start", on_event=events.append)

        self.assertEqual(
            (result.status, result.error_code, result.text),
            ("incomplete", "truncated", "Partial"),
        )
        self.assertEqual(result.messages, (UserMessage("Start"), assistant))
        self.assertEqual(
            [event.status for event in events if event.kind == "run_end"],
            ["incomplete"],
        )

    async def test_events_are_ordered_read_only_and_project_tool_results(self) -> None:
        class Args(BaseModel):
            pass

        call = ToolCall("read-1", "read", "{}")
        assistant = AssistantMessage("Draft", (call,))
        final = AssistantMessage("Complete")
        model = ScriptedModel(
            (ModelTextDelta("Dr"), ModelTextDelta("aft"), ModelReply(assistant)),
            (ModelTextDelta("Com"), ModelReply(final)),
        )
        agent = Agent(model)
        events: list[RunEvent] = []

        async def read(args: Args, ctx: ToolContext) -> ToolOutput:
            return ToolOutput("visible", details={"secret": ["private"]})

        agent.tools.register("read", "Read a value", Args, read)
        result = await agent.run("Read", on_event=events.append)

        self.assertEqual(result.messages[-1], final)
        self.assertEqual(result.text, "Complete")
        self.assertEqual(
            [event.sequence for event in events], list(range(1, len(events) + 1))
        )
        self.assertEqual({event.run_id for event in events}, {events[0].run_id})
        self.assertTrue(events[0].run_id)
        self.assertEqual(
            [(event.kind, event.turn, event.tool_call_id) for event in events],
            [
                ("run_start", None, None),
                ("message", None, None),
                ("turn_start", 1, None),
                ("text_delta", 1, None),
                ("text_delta", 1, None),
                ("message", 1, None),
                ("tool_start", 1, call.id),
                ("message", 1, call.id),
                ("tool_end", 1, call.id),
                ("turn_end", 1, None),
                ("turn_start", 2, None),
                ("text_delta", 2, None),
                ("message", 2, None),
                ("turn_end", 2, None),
                ("run_end", None, None),
            ],
        )
        self.assertEqual(
            [event.delta for event in events if event.kind == "text_delta"],
            ["Dr", "aft", "Com"],
        )
        self.assertEqual(
            [event.message for event in events if event.kind == "message"],
            [
                UserMessage("Read"),
                assistant,
                ModelToolResult(call.id, "visible", False),
                final,
            ],
        )
        self.assertEqual(events[-1].status, "completed")
        with self.assertRaises(FrozenInstanceError):
            setattr(events[-1], "status", "failed")

    async def test_model_event_after_terminal_reply_cannot_trigger_tools(self) -> None:
        class Args(BaseModel):
            pass

        call = ToolCall("bad-stream-1", "work", "{}")
        events: list[RunEvent] = []
        executions: list[str] = []

        async def work(args: Args, ctx: ToolContext) -> ToolOutput:
            executions.append(ctx.tool_call_id)
            return ToolOutput("worked")

        model = ScriptedModel(
            (
                ModelReply(AssistantMessage(tool_calls=(call,))),
                ModelTextDelta("too late"),
            )
        )
        agent = Agent(model)
        agent.tools.register("work", "Attempt a call", Args, work)

        result = await agent.run("Start", on_event=events.append)

        self.assertEqual((result.status, result.error_code), ("failed", "model_error"))
        self.assertEqual(result.messages, (UserMessage("Start"),))
        self.assertEqual(executions, [])
        self.assertEqual(
            [event.status for event in events if event.kind == "run_end"], ["failed"]
        )
