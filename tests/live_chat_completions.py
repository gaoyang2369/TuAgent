"""Opt-in smoke test against a real Chat Completions endpoint."""

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
from openai import AsyncOpenAI
from pydantic import BaseModel

from tuagent import Agent, ToolContext, ToolOutput
from tuagent.providers import ChatCompletionsProvider


def _read_key(env_file: Path) -> str:
    key = os.environ.get("TUAGENT_API_KEY", "").strip()
    if not key and env_file.is_file():
        for line in env_file.read_text(encoding="utf-8-sig").splitlines():
            name, separator, value = line.partition("=")
            if separator and name.strip() == "TUAGENT_API_KEY":
                key = value.strip().strip("\"'")
                break
    if not key:
        raise SystemExit(
            "Set TUAGENT_API_KEY in the environment or the selected env file"
        )
    return key


async def _verify(
    model: str, base_url: str, key: str, *, deepseek_non_thinking: bool
) -> bool:
    requests: list[dict[str, Any]] = []
    statuses: list[int] = []
    proofs: list[tuple[str, str, str]] = []

    async def record_request(request: httpx.Request) -> None:
        requests.append(json.loads(request.content))

    async def record_response(response: httpx.Response) -> None:
        statuses.append(response.status_code)

    class ProofArgs(BaseModel):
        label: str

    async def issue_proof(args: ProofArgs, context: ToolContext) -> ToolOutput:
        value = f"tuagent-proof-{uuid4().hex}"
        proofs.append((context.tool_call_id, args.label, value))
        return ToolOutput(value)

    http_client = httpx.AsyncClient(
        event_hooks={"request": [record_request], "response": [record_response]}
    )
    async with AsyncOpenAI(
        api_key=key,
        base_url=base_url,
        max_retries=0,
        timeout=30,
        http_client=http_client,
    ) as client:
        extra_body = (
            {"thinking": {"type": "disabled"}} if deepseek_non_thinking else None
        )
        provider = ChatCompletionsProvider(model, client=client, extra_body=extra_body)
        direct = await Agent(provider, max_turns=1).run(
            "Reply with exactly TUAGENT_DIRECT_OK."
        )
        direct_requests = len(requests)

        agent = Agent(provider, max_turns=2)
        agent.tools.register(
            "m3_acceptance_code",
            "Return a fresh verification code for the requested label.",
            ProofArgs,
            issue_proof,
        )
        round_trip = await agent.run(
            'Call m3_acceptance_code exactly once with label "m3". '
            "Reply with the exact code from its result."
        )

    tool_requests = requests[direct_requests:]
    paired = False
    proof = None
    if len(proofs) == 1 and len(tool_requests) == 2:
        call_id, label, proof = proofs[0]
        history = tool_requests[1]["messages"]
        paired = (
            label == "m3"
            and history[-2]["role"] == "assistant"
            and [call["id"] for call in history[-2]["tool_calls"]] == [call_id]
            and history[-1]
            == {"role": "tool", "tool_call_id": call_id, "content": proof}
        )

    direct_ok = (
        direct_requests == 1
        and direct.status == "completed"
        and (direct.text or "").strip() == "TUAGENT_DIRECT_OK"
    )
    tool_ok = (
        paired
        and round_trip.status == "completed"
        and proof is not None
        and proof in (round_trip.text or "")
    )
    report = {
        "model": model,
        "http_statuses": statuses,
        "direct": {
            "passed": direct_ok,
            "requests": direct_requests,
            "status": direct.status,
            "error": direct.error_code,
            "answer": direct.text,
        },
        "tool_round_trip": {
            "passed": tool_ok,
            "requests": len(tool_requests),
            "handler_executions": len(proofs),
            "paired_wire_messages": paired,
            "status": round_trip.status,
            "error": round_trip.error_code,
            "generated_value": proof,
            "answer": round_trip.text,
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2).replace(key, "[redacted]"))
    return direct_ok and tool_ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--env-file", type=Path, default=Path(".env.local"))
    parser.add_argument("--deepseek-non-thinking", action="store_true")
    args = parser.parse_args()
    passed = asyncio.run(
        _verify(
            args.model,
            args.base_url,
            _read_key(args.env_file),
            deepseek_non_thinking=args.deepseek_non_thinking,
        )
    )
    raise SystemExit(0 if passed else 1)
