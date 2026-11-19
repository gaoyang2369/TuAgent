# G120 遥测工具

从仓库根目录运行以下命令。实库验证读取进程环境变量 `HOST`、`MYSQL_USER`、`MYSQL_PW`、`DB_NAME`、`PORT`，缺项从 `.env.local` 读取。`DB_NAME` 使用 `dcma`。有可信模型哈希时可配置 `TELEMETRY_SOURCE_SHA256`，或在应用注册工具时传入 `expected_source_sha256`。

```powershell
python -m pip install -e ".[mysql,chat-completions]"
python -m unittest discover -s tests -v
python tests/live_telemetry.py
```

上述安装命令包含完整离线测试所需的 Chat Completions 可选依赖。单独运行实库示例只需安装 `.[mysql]`。`tests/live_telemetry.py` 用 `ScriptedModel` 发出四类工具调用，使用 MySQL 真实返回的数据完成 `Agent.run` 往返，输出北京时间、样本覆盖、工具返回量、三小时分页检查和锁存续窗检查。若窗口里找不到同时观测到触发与解除的事件，锁存复核会标为 `not_verified`，脚本以非零状态退出。

## 接入现有 Agent

```python
from tuagent import Agent, ToolRegistry
from tuagent.telemetry import TELEMETRY_ANALYSIS_PROMPT, register_telemetry_tools


async def ask_telemetry(provider):
    registry = ToolRegistry()
    register_telemetry_tools(registry)
    agent = Agent(provider, tools=registry, system_prompt=TELEMETRY_ANALYSIS_PROMPT)
    result = await agent.run("超温模拟表最近三小时的电机温度趋势如何？")
    return result.text
```

`provider` 使用应用已配置的 `ModelProvider`。已有系统提示词可拼接 `TELEMETRY_ANALYSIS_PROMPT`。省略 MySQL 扩展时，通用 Agent 和替身数据源仍可使用。固定样本的离线工具往返可单独运行。

```powershell
python -m unittest discover -s tests -p test_telemetry.py -k test_agent_round_trip_with_registered_tools_and_analysis_prompt -v
```

## 调用和结果

| 工具 | 参数示例 |
| --- | --- |
| `get_signal_metadata` | `{"table_name":"sim_f07011_overtemp","fields":["motor_temp","air_intake_temp"]}` |
| `query_telemetry` | `{"table_name":"sim_f30002_overvoltage","mode":"latest","fields":["dc_voltage"],"limit":100}` |
| `query_telemetry` | `{"table_name":"sim_f07011_overtemp","mode":"window","fields":["motor_temp"],"last_seconds":10800,"limit":1000}` |
| `summarize_telemetry` | `{"table_name":"sim_f07011_overtemp","fields":["motor_temp"],"last_seconds":10800,"bucket_seconds":60}` |
| `find_code_events` | `{"table_name":"sim_f07900_blocked","last_seconds":3600,"code_field":"fault_code","target_code":"F07900"}` |

- 默认版本为 `g120_sim_v1`。省略 `get_signal_metadata.fields` 时提供全部 32 个字段名，含保留列 `actual_solution` 的定义。原始采样投影排除该列。
- 历史窗口左闭右开，最长 7 天。游标固定窗口和字段，以数值毫秒与行 ID 续读。对模拟器的追加、补入与过期清理，工具用窗口行数和最大自增 ID 检查数据是否变化。每次调用使用独立只读事务，续页会复用已验证的覆盖统计。遇到 `stale_cursor` 时，丢弃已拼接的旧页，使用首个结果的绝对 `start_ms/end_ms` 从头读取。旧版缺少窗口版本信息的游标也走此恢复路径。
- 采样工具单页最多 1000 点。历史结果的 `coverage` 对应整个窗口，`page` 给出本页数量和续页游标。原始点的引用含表、版本、毫秒时刻和行 ID，配合该点的 `values` 字段名定位信号。
- 统计按窗口起点分桶，最多 2000 桶。变化率用桶内首末有效样本的差值除以经过秒数。阈值时长只累加两端均有效且满足阈值的连续 3 秒间隔，跨桶按时间分配。同一时刻有多条读数时，全部读数均满足条件才将该时刻用于确认时长。原始点仍逐行返回，非有限数用字符串表示并标记 `valid=false`。复核统计时使用桶引用、信号名、指标及结果中的 `calculation`。
- `new_trigger_count` 只统计连续有效采样中从字符串 `'0'` 变为非零码的确认触发。同一时刻的同码行合并为一次观测，冲突码计入 `conflicting_code_timestamps` 并打断连续性。窗口左端的清除保留在事件列表里，窗口内新增触发和确认锁存时长均为零。跨窗锁存和缺前样本的带码点也可出现在事件列表里。确认锁存时长按相邻同码间隔累加，另列观测到的触发至清除的时差。
- `ToolOutput.content` 最多 16 KiB，历史模式提供从模型实际可见末点继续的游标。较长统计或事件明细可依提示细化查询。应用可从 `ToolResultMessage.details` 读取工具页或统计桶的完整本地结果。字典无可信外部哈希时，校验状态为 `dictionary_consistency_only`。

## 2026-11-19 验收记录

安装 `.[mysql,chat-completions]` 后运行离线全量测试，40 项通过。固定样本覆盖 9 秒母线升压到 800 V 的峰值及时刻、无效进气温度剔除、有效零值保留、缺样及无效点打断超限时长，以及报警与故障序列的锁存边界。

实库只读验证于北京时间 16:46:10 开始，`Agent.run` 的 7 次调用均返回给脚本模型，进程退出码为 0。

| 检查 | 实际结果 |
| --- | --- |
| 四表最新采样 | 各 10 点，四表覆盖均为 16:45:39 至 16:46:06，每表同时返回所请求的 6 个字段定义。 |
| 超温表三小时分钟统计 | 请求区间为 13:46:12.868 至 16:46:12.868。返回 180 桶、3598/3600 点，尾部缺 2 点。电机温度峰值 145.004 °C，观测于 13:48:00。 |
| 超温表三小时原始分页 | 固定窗口为 13:46:06 至 16:46:06，四页各返回 1000、1000、1000、600 点。合计 3600 点，无漏行或重行。实查进气温度无效点为 0；`-200` 和 `NULL` 的剔除由离线样本验证。 |
| 堵转表一小时事件 | 返回 1198/1200 点，尾部缺 2 点。确认 2 次 `F07900` 触发，时刻为 16:10:15 和 16:40:15。抽查第一段锁存，其 35 条带码记录对应 1 次触发。从锁存中途重查得到 1 段延续，确认新触发 0 次。 |

四表字典返回同一模型哈希，实测核验状态为 `dictionary_consistency_only`。

## 2026-11-19 代码复核与复验

离线全量测试 46 项通过，遥测实现与验证脚本的 Ruff 检查通过。新增回归覆盖翻页期间补入或清理、同一时刻的重复与冲突读数、左端清除、默认字段清单、非有限数、连接重试边界和锁存验证缺少样本。

北京时间 17:08 完成实库只读复验，脚本退出码为 0。四张表各取到 10 个最新点及所需字段定义，7 次工具调用均通过 `Agent.run` 返回脚本模型。

| 检查 | 实际结果 |
|---|---|
| 三小时历史分页 | 固定窗口 `[14:08:00, 17:08:00)` 共 3600 点。四页分别为 1000、1000、1000、600 点，按时刻与行 ID 核对无漏行或重行。 |
| 三小时分钟统计 | 相对窗口截至 17:08:03.394，返回 180 桶，实有 3599 个采样点，期望 3600 个。 |
| 一小时堵转事件 | 确认 2 次 `F07900` 新触发。抽查一次锁存含 35 行，从中途重查得到 1 段延续、0 次新触发。 |

复验过程中两次出现 MySQL 建连错误 `2003`。实现在建立连接时对 `2002`、`2003` 最多重试一次。鉴权失败立即返回，已开始的事务遇到读取错误则结束该次查询并返回原因。
