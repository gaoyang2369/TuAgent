"""Fault diagnosis through the six read-only business tools."""

from pathlib import Path

from .agent import Agent
from .manual import register_manual_tools
from .model import ModelProvider
from .telemetry import TELEMETRY_ANALYSIS_PROMPT, register_telemetry_tools
from .telemetry.contracts import WindowArgs
from .telemetry.data import beijing
from .tools import ToolRegistry

FAULT_DIAGNOSIS = """fault-diagnosis · G120_SIM
根据用户实际所问选工具。手册原因、最短时间等知识问题直接 search_manual；
当天实际故障次数用 find_code_events，按问题限定代码。收费需要服务报价，库存需要库存系统。
问题仅问“手册列了哪些原因”时，只在手册中取证，用户已描述的现象就是检索问题，
即使提到“今天”也不查询任何现场数据，不从表名猜报码。摘录被截断时，只列原文可见项，不推断总数。只问“报了几次”时，
仅用 find_code_events 取 new_trigger_count，成功后直接答次数、期间与覆盖。
以下流程只在用户要求解释一次实际跳闸、停机时执行。对停机诊断先遍历指定窗口的 fault_code 事件，不预设目标码。以最近一个非空 trigger 为 T；
若请求窗内没有 trigger，直接结束诊断并报告本窗计数、延续状态与覆盖。此时不得扩大窗口、
不得改用 last_seconds 或窗口外的故障回答。只有本窗已确认触发时才读取其前后背景。
异常细查只用所选源。围绕观测到的 T 可向前后读取背景，窗口外事件不计为本窗触发。
- F30002 读 T±30秒（右端加1毫秒以包含末点），同点对齐指令/实速、转矩、actual_power、
  feedback_power、dc_voltage、fault_code。覆盖整窗后才报电压峰值及其首次时刻和行ID。分别记录 T-30000、T+30000；若多读了背景，峰值只统计这 21 个采样点。
  摘出减速前、负轴功率、峰值和跳闸后的实测点。两种功率同源，只计一项证据。
- F07900 读 T 前60秒至后30秒，同点对齐指令、实速、转矩、电流和报码；
  区分机械卡滞假设与已观测的零速高转矩。
- F07011 先看 T 前3小时至后30分钟的电流、负载率、温度趋势（长窗截断时先用半小时
  分桶覆盖全程，再对关键窗口看分钟桶）；在该段定位 alarm_code=A07910，
  从两个事件返回的 trigger.timestamp_ms 分别读取触发点前后各3秒的原始电流、负载率、温度与两种代码；每次仅取这五个字段，报告各自的行ID与温度。缺前史就标明可证边界。
以实际代码 lookup_drive_code，并用同码 search_manual 查“原因”，设备系列传 G120_SIM。
手册是 candidate_only，控制单元和固件未知；标题与原因分别引用返回的版次、PDF 页和英文原句。
表名只用于选源。现场保护阈值需现场配置；不要把观测值称为设定值。
对具体停机诊断，只使用工具可见的有效采样和手册原文形成一份中文 Markdown 诊断记录。按“观测事实”
“吻合的候选原因”“待核实项”组织。注明源、字段版本、查询时刻、请求和实际覆盖、缺口。
数值带单位、北京时间和行ID或桶起止、字段、指标；逐字引用手册原因。简洁列出关键点。没有逐点细查的事件只报告计数。参数编号以 p 开头，页码取 pdf_page。
时间以工具的 time_beijing 字段为准，避免手算。"""


def diagnosis_agent(
    provider: ModelProvider,
    window: WindowArgs,
    index_dir: Path,
    *,
    env_file: Path | None = None,
    manual_pilot: bool = False,
) -> Agent:
    window.require_window()
    registry = ToolRegistry()
    register_telemetry_tools(registry, env_file=env_file)
    register_manual_tools(registry, index_dir=index_dir, chinese_pilot=manual_pilot)
    scope = f"本次数据源 {window.table_name}。"
    if window.start_ms is not None:
        scope += (
            f"请求窗口 [{window.start_ms}, {window.end_ms}) 毫秒，"
            f"北京时间 [{beijing(window.start_ms)}, {beijing(window.end_ms)})。"
        )
    else:
        scope += f"请求窗口为最近 {window.last_seconds} 秒。"
    return Agent(
        provider,
        tools=registry,
        max_turns=18,
        system_prompt="\n\n".join((TELEMETRY_ANALYSIS_PROMPT, FAULT_DIAGNOSIS, scope)),
    )
