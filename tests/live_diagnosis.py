"""Ask the configured generation model to diagnose a read-only simulator window."""

import argparse
import asyncio
import json
from pathlib import Path

from openai import AsyncOpenAI

from live_chat_completions import _read_key
from tuagent import AssistantMessage, ToolResultMessage
from tuagent.diagnosis import diagnosis_agent
from tuagent.providers import ChatCompletionsProvider
from tuagent.telemetry.contracts import WindowArgs
from tuagent.telemetry.mysql import TABLES


def review_window(history, window):
    """Derive review bounds only after the model has located an in-window trigger."""
    calls = {
        call.id: call
        for msg in history
        if isinstance(msg, AssistantMessage)
        for call in msg.tool_calls
    }
    triggers = []
    for msg in history:
        if not isinstance(msg, ToolResultMessage) or msg.is_error:
            continue
        call = calls[msg.tool_call_id]
        if call.name != "find_code_events":
            continue
        args = json.loads(call.arguments)
        if (
            args.get("table_name") != window.table_name
            or args.get("target_code")
            or args.get("code_field", "fault_code") != "fault_code"
            or any(
                args.get(key) != getattr(window, key)
                for key in ("start_ms", "end_ms", "last_seconds")
            )
        ):
            continue
        payload = json.loads(msg.content)
        if payload.get("events_truncated") or payload.get("content_truncated"):
            continue
        triggers.extend(e for e in payload.get("events", []) if e.get("trigger"))
    event = max(triggers, key=lambda e: e["trigger"]["timestamp_ms"], default=None)
    if event and event["code"] == "F30002":
        t = event["trigger"]["timestamp_ms"]
        return (
            f"请复核你刚定位的最近触发。精确细查范围为 [{t - 30000}, {t + 30001})。"
            "请用此窗口只查 dc_voltage 的原始点，检查实际覆盖、有效点数与首次峰值。"
            "复用已有功率、转速、代码和手册返回，在最终中文报告中只列三个关键样本、"
            "锁存计数、最吻合的手册原因原句及待核实项。查阅更多背景时区分两个范围。"
        )
    if event and event["code"] == "F07011":
        t = event["trigger"]["timestamp_ms"]
        return (
            f"复核热趋势。请 summarize_telemetry 查询 [{t - 10800000},{t + 1800000})，"
            "fields 仅 motor_load_rate 和 motor_temp，metrics 仅 mean，bucket_seconds=1800。"
            f"再查 [{t - 60000},{t + 60000}) 的一分钟桶。用上述实见的桶和先前两处报警、"
            "故障附近的原始点完成独立的中文报告，保留原请求范围、负载与温升、手册原因原句、版次和页码，注明 candidate_only。不得将观测时差认定为参数延时。"
        )
    return None


async def run(args, window):
    key = _read_key(args.env_file)
    async with AsyncOpenAI(
        api_key=key, base_url=args.base_url, timeout=90, max_retries=1
    ) as client:
        provider = ChatCompletionsProvider(
            args.model,
            client=client,
            extra_body={"thinking": {"type": "disabled"}}
            if args.model == "deepseek-flash"
            else None,
        )
        agent = diagnosis_agent(
            provider,
            window,
            args.index_dir,
            env_file=args.env_file,
            manual_pilot=args.manual_pilot,
        )
        review = None
        result = await agent.run(args.question)
        if result.status == "completed" and "跳闸" in args.question:
            review = review_window(agent.history, window)
            if review:
                result = await agent.run(review)
    calls = {
        call.id: call
        for msg in agent.history
        if isinstance(msg, AssistantMessage)
        for call in msg.tool_calls
    }
    receipts = []
    for msg in agent.history:
        if isinstance(msg, ToolResultMessage):
            call = calls[msg.tool_call_id]
            try:
                payload = json.loads(msg.content)
            except ValueError:
                payload = msg.content
            receipts.append(
                {
                    "id": call.id,
                    "tool": call.name,
                    "arguments": call.arguments,
                    "is_error": msg.is_error,
                    "result": payload,
                }
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "model": args.model,
        "question": args.question,
        "window": window.model_dump(),
        "status": result.status,
        "error_code": result.error_code,
        "receipts": receipts,
        "review_prompt": review,
        "report": result.text,
    }
    args.out.with_suffix(".json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf8"
    )
    print(
        result.status,
        result.error_code or "",
        [r["tool"] for r in receipts],
        flush=True,
    )
    if result.status != "completed" or not result.text:
        return False
    args.out.write_text(result.text.rstrip() + "\n", encoding="utf8")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=TABLES, required=True)
    parser.add_argument("--start-ms", type=int)
    parser.add_argument("--end-ms", type=int)
    parser.add_argument("--last-seconds", type=int)
    parser.add_argument("--question", default="这段时间最近一次跳闸是怎么回事")
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--env-file", type=Path, default=Path(".env.local"))
    parser.add_argument("--index-dir", type=Path, default=Path(".manual/index"))
    parser.add_argument("--manual-pilot", action="store_true")
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Markdown path; JSON receipts saved alongside",
    )
    args = parser.parse_args()
    if args.out.suffix.lower() != ".md":
        parser.error("--out must be a Markdown (.md) path")
    window = WindowArgs(
        table_name=args.source,
        start_ms=args.start_ms,
        end_ms=args.end_ms,
        last_seconds=args.last_seconds,
    )
    window.require_window()
    return asyncio.run(run(args, window))


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
