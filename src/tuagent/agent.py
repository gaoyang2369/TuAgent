import asyncio
from typing import Any

from .events import EventHandler
from .loop import _RunLoop, _can_continue
from .messages import Message, RunResult, UserMessage
from .model import ModelProvider
from .tools import ToolExecutor, ToolRegistry


class Agent:
    """Run asynchronous conversations with one in-memory history per agent."""

    def __init__(
        self,
        provider: ModelProvider,
        *,
        system_prompt: str = "",
        tools: ToolRegistry | None = None,
        context: Any = None,
        max_turns: int = 16,
    ) -> None:
        if type(max_turns) is not int or max_turns < 1:
            raise ValueError("max_turns must be a positive integer")

        self.tools = tools if tools is not None else ToolRegistry()
        self._provider = provider
        self._system_prompt = system_prompt
        self._context = context
        self._max_turns = max_turns
        self._history: list[Message] = []
        self._active_run: _RunLoop | None = None

    @property
    def history(self) -> tuple[Message, ...]:
        return tuple(self._history)

    async def run(
        self, text: str, *, on_event: EventHandler | None = None
    ) -> RunResult:
        """Append a user message and run, optionally notifying a synchronous observer."""
        return await self._run(UserMessage(text), on_event)

    def abort(self) -> None:
        """Request cancellation of the active run from the agent's event loop."""
        if self._active_run is not None:
            self._active_run.abort()

    async def continue_run(self, *, on_event: EventHandler | None = None) -> RunResult:
        """Resume after a user message or a settled batch of tool results."""
        return await self._run(None, on_event)

    async def _run(
        self, user_message: UserMessage | None, on_event: EventHandler | None
    ) -> RunResult:
        if self._active_run is not None:
            raise RuntimeError("This agent already has a run in progress")
        if user_message is None and not _can_continue(self._history):
            raise RuntimeError(
                "Continuation requires a user message or settled tool results"
            )

        run = _RunLoop(
            self._provider,
            self._history,
            system_prompt=self._system_prompt,
            executor=ToolExecutor(self.tools, context=self._context),
            max_turns=self._max_turns,
            on_event=on_event,
        )
        self._active_run = run
        try:
            worker = asyncio.create_task(run.execute(user_message))
            try:
                return await asyncio.shield(worker)
            except asyncio.CancelledError:
                run.abort()
                # Repeated cancels of the caller must still allow the worker to settle the batch.
                while not worker.done():
                    try:
                        await asyncio.shield(worker)
                    except asyncio.CancelledError:
                        continue
                raise
        finally:
            self._active_run = None
