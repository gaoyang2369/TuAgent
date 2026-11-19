import asyncio
import logging
from collections import deque
from collections.abc import Sequence
from typing import Any, NamedTuple
from uuid import uuid4

from .events import EventHandler, RunEvent, RunEventKind
from .messages import (
    AssistantMessage,
    Message,
    RunResult,
    RunStatus,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from .model import (
    ModelMessage,
    ModelProvider,
    ModelReply,
    ModelRequest,
    ModelTextDelta,
    _build_request,
    _project_message,
)
from .tools.executor import ToolExecutor, _tool_error

_logger = logging.getLogger(__name__)


class _Outcome(NamedTuple):
    status: RunStatus
    text: str | None = None
    error_code: str | None = None


class _RunLoop:
    def __init__(
        self,
        provider: ModelProvider,
        history: list[Message],
        *,
        system_prompt: str,
        executor: ToolExecutor,
        max_turns: int,
        on_event: EventHandler | None,
    ) -> None:
        self._provider = provider
        self._history = history
        self._system_prompt = system_prompt
        self._executor = executor
        self._max_turns = max_turns
        self._on_event = on_event
        self._run_id = uuid4().hex
        self._sequence = 0
        self._turns = 0
        self._caller = asyncio.current_task()
        self._caller_cancels = (
            self._caller.cancelling() if self._caller is not None else 0
        )
        self._task: asyncio.Task[Any] | None = None
        self._abort_requested = False
        self._interrupt: asyncio.Handle | None = None

    def abort(self) -> None:
        if self._abort_requested:
            return
        self._abort_requested = True
        if self._task is not None and not self._task.done():
            # A synchronous observer can request abort while the worker is running.
            self._interrupt = self._task.get_loop().call_soon(self._task.cancel)

    def _check_abort(self) -> None:
        # The worker may resume before the caller can relay its cancellation.
        if (
            self._caller is not None
            and self._caller.cancelling() > self._caller_cancels
        ):
            self._abort_requested = True
        if self._abort_requested:
            if self._interrupt is not None:
                self._interrupt.cancel()
            raise asyncio.CancelledError

    async def execute(self, user_message: UserMessage | None) -> RunResult:
        self._task = asyncio.current_task()
        start = len(self._history)
        outcome = _Outcome("failed", error_code="internal_error")
        self._emit("run_start")
        try:
            if user_message is not None:
                self._commit(user_message)
            outcome = await self._advance()
        except asyncio.CancelledError:
            outcome = _Outcome("cancelled", error_code="cancelled")
        finally:
            self._emit("run_end", status=outcome.status, error_code=outcome.error_code)
            if self._interrupt is not None:
                self._interrupt.cancel()
            self._task = None

        return RunResult(
            status=outcome.status,
            messages=tuple(self._history[start:]),
            turns=self._turns,
            text=outcome.text,
            error_code=outcome.error_code,
        )

    async def _advance(self) -> _Outcome:
        for turn in range(1, self._max_turns + 1):
            self._check_abort()
            self._emit("turn_start", turn=turn)
            try:
                self._check_abort()
                self._turns = turn
                outcome = await self._run_turn(turn)
            finally:
                self._emit("turn_end", turn=turn)
            if outcome is not None:
                return outcome
            self._check_abort()
        return _Outcome("turn_limit", error_code="turn_limit")

    async def _run_turn(self, turn: int) -> _Outcome | None:
        request = _build_request(
            self._system_prompt, self._history, self._executor.declarations()
        )
        try:
            reply = await self._read_reply(request, turn)
        except Exception:
            self._check_abort()
            return _Outcome("failed", error_code="model_error")

        self._check_abort()
        if reply is None:
            return _Outcome("failed", error_code="model_stream_incomplete")
        stop_reason = reply.stop_reason
        if not isinstance(stop_reason, str) or stop_reason not in (
            "stop",
            "tool_calls",
            "length",
        ):
            return _Outcome("failed", error_code="invalid_stop_reason")

        message = reply.message
        if not isinstance(message, AssistantMessage) or not isinstance(
            message.content, str
        ):
            return _Outcome("failed", error_code="invalid_assistant_message")
        calls = message.tool_calls
        if stop_reason == "tool_calls" and not calls:
            return _Outcome("failed", error_code="invalid_stop_reason")
        call_error = _tool_call_error(calls)
        if call_error is not None:
            return _Outcome("failed", error_code=call_error)

        self._commit(message, turn=turn)
        if stop_reason == "length":
            if not calls:
                return _Outcome("incomplete", message.content, "truncated")
            for call in calls:
                self._record_tool_result(
                    _tool_error(
                        call,
                        "truncated",
                        "Tool skipped because the model reply was truncated.",
                    ),
                    turn,
                )
        elif calls:
            await self._execute_tools(calls, turn)
        else:
            return _Outcome("completed", message.content)
        return None

    async def _read_reply(self, request: ModelRequest, turn: int) -> ModelReply | None:
        reply = None
        stream = self._provider.stream(request)
        try:
            async for event in stream:
                self._check_abort()
                if reply is not None:
                    raise ValueError("A model event followed the terminal reply")
                if isinstance(event, ModelTextDelta):
                    self._emit("text_delta", turn=turn, delta=event.text)
                    self._check_abort()
                elif isinstance(event, ModelReply):
                    reply = event
                else:
                    raise ValueError("Unexpected model event")
        finally:
            close = getattr(stream, "aclose", None)
            if close is not None:
                await close()
        return reply

    async def _execute_tools(self, calls: tuple[ToolCall, ...], turn: int) -> None:
        settled = 0
        started = False
        try:
            for call in calls:
                self._check_abort()
                self._emit("tool_start", turn=turn, tool_call_id=call.id)
                self._check_abort()
                started = True
                result = await self._executor.execute(call)
                started = False
                self._record_tool_result(result, turn)
                settled += 1
        except asyncio.CancelledError:
            for call in calls[settled:]:
                result = (
                    _tool_error(
                        call,
                        "outcome_unknown",
                        "Tool was interrupted and its outcome is unknown.",
                    )
                    if started
                    else _tool_error(
                        call,
                        "not_started",
                        "Tool did not start because the run was cancelled.",
                    )
                )
                self._record_tool_result(result, turn)
                started = False
            raise

    def _record_tool_result(self, result: ToolResultMessage, turn: int) -> None:
        self._commit(result, turn=turn)
        self._emit(
            "tool_end",
            turn=turn,
            tool_call_id=result.tool_call_id,
            error_code=result.error_code,
        )

    def _commit(self, message: Message, *, turn: int | None = None) -> None:
        self._history.append(message)
        tool_call_id = (
            message.tool_call_id if isinstance(message, ToolResultMessage) else None
        )
        self._emit(
            "message",
            turn=turn,
            tool_call_id=tool_call_id,
            message=_project_message(message),
        )

    def _emit(
        self,
        kind: RunEventKind,
        *,
        turn: int | None = None,
        tool_call_id: str | None = None,
        message: ModelMessage | None = None,
        delta: str | None = None,
        status: RunStatus | None = None,
        error_code: str | None = None,
    ) -> None:
        if self._on_event is None:
            return
        self._sequence += 1
        event = RunEvent(
            kind,
            self._run_id,
            self._sequence,
            turn,
            tool_call_id,
            message,
            delta,
            status,
            error_code,
        )
        try:
            self._on_event(event)
        except (Exception, asyncio.CancelledError):
            _logger.exception("Run event observer failed for %s", kind)


def _tool_call_error(calls: tuple[ToolCall, ...]) -> str | None:
    call_ids: set[str] = set()
    for call in calls:
        if not isinstance(call, ToolCall):
            return "invalid_tool_call"
        if not isinstance(call.id, str) or not call.id.strip() or call.id in call_ids:
            return "invalid_tool_call_id"
        if not isinstance(call.name, str):
            return "invalid_tool_call_name"
        if not isinstance(call.arguments, str):
            return "invalid_tool_call_arguments"
        call_ids.add(call.id)
    return None


def _can_continue(history: Sequence[Message]) -> bool:
    if not history or not isinstance(history[-1], (UserMessage, ToolResultMessage)):
        return False

    expected_ids: deque[str] = deque()
    for message in history:
        if isinstance(message, ToolResultMessage):
            if not expected_ids or message.tool_call_id != expected_ids.popleft():
                return False
        elif expected_ids:
            return False
        elif isinstance(message, AssistantMessage):
            if _tool_call_error(message.tool_calls) is not None:
                return False
            expected_ids.extend(call.id for call in message.tool_calls)
    return not expected_ids
