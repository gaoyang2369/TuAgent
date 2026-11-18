from collections.abc import AsyncIterator, Callable
from typing import TypeAlias

from tuagent import ModelEvent, ModelRequest


Step: TypeAlias = (
    ModelEvent | tuple[ModelEvent, ...] | Callable[[ModelRequest], ModelEvent] | None
)


class ScriptedModel:
    def __init__(self, *steps: Step) -> None:
        self._steps = iter(steps)
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        self.requests.append(request)
        step = next(self._steps)
        if isinstance(step, tuple):
            for event in step:
                yield event
        elif step is not None:
            yield step(request) if callable(step) else step
