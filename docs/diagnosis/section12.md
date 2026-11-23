# 第 12 节验收记录

2026 年 11 月 23 日，真实生成模型 `deepseek-flash` 经现有 Agent Loop 查询只读 MySQL 实库中的 G120 **模拟采样**和本地 Siemens 英文 PDF。首例 F30002 的采样峰值、触发时刻、负轴功率和原因页均已独立核对。中文无代码试点保留原版，结束本轮检索优化。[F30002 中文报告](F30002.md) · [模型可见的完整取证记录与生成原文](F30002.record.json)

## 带码诊断

三例都使用同一个 system prompt、同一套六个工具和原问“这段时间最近一次跳闸是怎么回事”。事件查询先遍历所选窗口，取最后一个可确认的触发。模型选择工具，程序从其已读到的事件计算精确复核窗口。过压复核完整采样峰值，超温复核长窗负载和温度。以下均为北京时间，字段版本为 `g120_sim_v1`。

| 请求源和窗口 | 观测事实 | 吻合的候选原因及待核实项 |
|---|---|---|
| `sim_f30002_overvoltage`，17:00–17:50 | 2 次新触发。最近一次在 17:40:09，行 432642。此前 9 秒出现负转矩与负轴功率，电压依次为 557、638、719、800 V。独立重读 T±30 秒的 21 行，时间戳、行 ID 和电压逐点一致。 | 标题在第 904 页，第 905 页的 “motor regenerates too much energy.” 与减速回馈吻合。进线电压、制动电阻状态和整定值待核实。 |
| `sim_f07900_blocked`，17:00–17:50 | 2 次新触发，最近一次为 17:40:15，行 432644。17:40:00 至 17:40:12 的五个连续点（432639–432643）指令为 1000 rpm，实速为 0 rpm，转矩为 3.5 N·m。触发行电流和转矩归零。 | 第 885 页原句 “Motor has been operating at the torque limit longer than the time specified in p2177 and below the speed threshold set in p2175.” 零速高转矩支持堵转现象。机械卡滞、实际限矩、p2175/p2177 与编码器反馈待核实。 |
| `sim_f07011_overtemp`，11:00–17:50 | 故障窗覆盖 8200 点。T 前 3 小时至后 30 分钟共 4200 点。12:18–13:48 的三个半小时桶电流负载率均值均约 119.19%，温度均值依次为 81.81、116.51、137.56 °C。13:19:48 报 A07910（130.007 °C，行 427435），13:48:00 触发 F07011（145.004 °C，行 427999），间隔 28 分 12 秒。 | 持续高电流负载伴随升温，跳闸后电流归零。A07910 在第 887 页，F07011 在第 870 页。后者列有 “motor is overloaded.” 传感器类型、环境通风、温度阈值与实际延时均待核实。 |

三份手册释义均取自 A5E33839529，02/2023，§4.2。G120_SIM 的控制单元与固件未知，适用性标为 `candidate_only`。模型能看到工具契约中的实际表名。因此额外用每张故障表自身的 17:00–17:05 作对照，三窗各 100/100 点，分别确认 **0 次新触发**。首次过压对照中模型曾越窗寻找故障，收紧指引后复验通过。归档的行引用、统计桶、事件返回和工具调用顺序可在滚动数据过期后核对。[其他场景的证据摘录](scenario-evidence.json)

**真实模型的意图区分。** “今天减速时母线过压，手册列了哪些原因”调用 `search_manual` 和 `lookup_drive_code`，仅查询手册。最终复验引用了再生能量过多、供电电压过高两条可见原因，并指出列表在摘录处截断。“今天这台设备报了几次过压”只调用 `find_code_events`，00:00–17:50 的 `new_trigger_count` 为 36。这里使用全局默认手册检索。另问上门费用与库房数量，模型分别指出缺服务报价、库存系统，均未给出金额或数量。[意图与工具实测](scenario-evidence.json)

## 删除与兼容性

| 位置 | 实际改动 |
|---|---|
| `src/tuagent/manual/evidence_services.py` | 删除整个 HTTP 客户端模块，包括 OllamaRewriter、HttpReranker 和专用服务异常处理。 |
| `evidence.py`、`tools.py` | 删除改写和重排协议、配置、固定问句生成、改写校验、附加英文向量、二次融合、成对打分、相应失败分支、旧摘录开关及重复的原候选记录。保留原问术语匹配、同页原文、范围过滤和显式向量降级。 |
| `tests/manual_eval/evaluate_v2.py` | 移除实验臂、专用超时与服务地址、重排二进制与权重参数、历史冻结写入逻辑和专用统计字段。现在从保留路径执行公开回归或新题，输出到新文件。 |
| `tests/test_manual_evidence.py`、使用说明 | 删除改写漂移断言和重排成功/失败替身。验证原问只编码一次、范围过滤不编码、失败和显式降级。历史九路统一从提交 `24503d3ef6b882b13ff39b0afa06d1506e0e4c6a` 的源码与脚本复现。 |

全局默认仍为 `hybrid_concepts`。开启 `chinese_pilot=True` 后，范围匹配的中文无代码检索使用 `evidence_rrf`。明确代码继续精确查询或按码走默认检索。保留 `semantic=False`、`vector`、`rrf` 和本地 `evidence_rules`。六个业务工具的模型参数保持原契约。[当前用法](../manual-usage.md) · [九路历史报告和命令](../manual-evidence-v2.md)

**离线与本地手册验证。** 63 项单元测试通过。使用相同 PDF 索引和输入，42 道 v1 题的默认模型可见内容与清理前逐字一致。确定性向量替身另比对试点命中、无答案、向量故障、显式降级四种返回，四项逐字一致。46 项快照的 SHA-256 均为 `b05ee41284acf7456c590bcc16340b2b184632158ba4e4fa1cd7b87f8377b360`。真实 PDF 验收通过 15 次脚本模型工具往返，覆盖四个目标代码及页码。

## 一轮选句实验

先留出 7 道新自然问法（4 可答、3 手册外），然后只在原选择器上做一次修改。固定召回和融合参数，候选选择有时间、单位、条件的原句以及与操作频次相符的列表项，现场事件意图由提问内容判断。开发使用公开题。语料、题集、逐题前五条可见引文与失败原因已分开保存。[新题和标注](../../tests/manual_eval/section12_questions.json) · [新旧逐题实测](../../tests/manual_eval/section12_results.json)

| 同条件试验 | 原试点 | 选句候选 |
|---|---:|---:|
| 新题，严格 `evidence_rrf`，完成 / 发起 | 0/7 | 0/7 |
| 可答题有可见支持 / 发起，成功返回分母 | 0/4，完成 0 | 0/4，完成 0 |
| 硬负例误返回 / 完成 | 0/0 | 0/0 |
| 超时 / 发起；连接失败 / 发起 | 0/7；7/7 | 0/7；7/7 |
| 同一预热协议，含失败的端到端 P95 | 5073.899 ms | 5055.505 ms |
| 同题仅本地规则取证，支持 / 可答 | 0/4 | 2/4 |
| 本地规则取证，硬负例误返回 / 完成；P95 | 0/3；27.482 ms | 0/3；29.002 ms |

每版先单独预热一次，随后按相同题序计时，P95 使用最近秩。两次严格运行的预热也都失败。内网 Ollama 的 TCP 连接建立后，`/api/tags` 无 HTTP 响应即断开，工具报告 `embedding_unavailable`。两版均未取得查询向量。表中严格模式的 P95 是失败返回延迟，成功请求的预热延迟不可估计。单用本地规则时，原版在 J2 返回了不支持所问的内容；候选在 J2、J3 有可见支持，J1、J4 仍误拒。

用原融合次序回放已公开的 22 道可答与 10 道硬负例，原版支持 7/22，候选 11/22，硬负例误返回均 0/10。B05 的第 871 页引文保留了 p0607 计时到期后故障仍存在、最早在 A07015 报警后 **50 ms** 才出 F07016 的条件。B20 的第 911 页原文选中第 5 项，单位时间预充次数太多导致 **precharging resistors** 过热。原始盲测成绩为 7/22，仅放宽词数的旧候选事后回放为 9/22，本轮 11/22 也是事后回放。

**决策为保留原试点。** 新题没有取得有效的融合结果，无法证实支持数净增、误返回持平和预热 P95 ≤ 2.5 秒。候选仅归档为复现实验的补丁，当前运行代码使用原选句与拒答规则。

## 从仓库根目录复现

已配置的环境直接运行以下 PowerShell 命令。新环境先安装 `python -m pip install -e ".[chat-completions,mysql,manual,manual-vector]"`，按手册使用说明准备原 PDF 和索引。数据库和模型密钥由现有环境或 `.env.local` 提供。

```powershell
python -m unittest discover -s tests -v
python tests/live_manual.py --pdf .manual/g120-cu240be-2-0223-en.pdf --index-dir .manual/index --evidence-v2
python tests/check_overvoltage_report.py
python tests/live_diagnosis.py --source sim_f30002_overvoltage --last-seconds 3600 --out .manual/diagnosis/recent.md
```

以下是验收当日的固定窗口。MySQL 滚动保留 7 天，归档报告的回查命令 `python tests/check_overvoltage_report.py` 使用保存的工具消息和原 PDF，可长期运行。

```powershell
python tests/live_diagnosis.py --source sim_f30002_overvoltage --start-ms 1795424400000 --end-ms 1795427400000 --out .manual/diagnosis/overvoltage.md
python tests/live_diagnosis.py --source sim_f07900_blocked --start-ms 1795424400000 --end-ms 1795427400000 --out .manual/diagnosis/blocked.md
python tests/live_diagnosis.py --source sim_f07011_overtemp --start-ms 1795402800000 --end-ms 1795427400000 --out .manual/diagnosis/thermal.md
foreach ($table in 'sim_f30002_overvoltage','sim_f07900_blocked','sim_f07011_overtemp') {
    python tests/live_diagnosis.py --source $table --start-ms 1795424400000 --end-ms 1795424700000 --out ".manual/diagnosis/control-$table.md"
}
python tests/live_diagnosis.py --source sim_f30002_overvoltage --start-ms 1795363200000 --end-ms 1795427400000 --question '今天减速时母线过压，手册列了哪些原因' --out .manual/diagnosis/manual-intent.md
python tests/live_diagnosis.py --source sim_f30002_overvoltage --start-ms 1795363200000 --end-ms 1795427400000 --question '今天这台设备报了几次过压' --out .manual/diagnosis/event-intent.md
```

清理前后通过同一个快照脚本比较。以下命令在新的 PowerShell 会话中执行。

```powershell
git archive -o .manual/section12-original.zip 24503d3 src
Expand-Archive .manual/section12-original.zip -DestinationPath .manual/section12-original -Force
python tests/manual_eval/cleanup_snapshot.py --out .manual/current-snapshot.json
$env:PYTHONPATH = (Resolve-Path .manual/section12-original/src).Path
python tests/manual_eval/cleanup_snapshot.py --out .manual/original-snapshot.json
(Get-FileHash .manual/current-snapshot.json).Hash -eq (Get-FileHash .manual/original-snapshot.json).Hash
```

新题比较只运行原试点和一次选句候选。另开 PowerShell，用新输出文件名。候选补丁只应用到临时源码副本。

```powershell
$env:OLLAMA_BASE_URL = 'http://10.108.13.254:11434'
$env:EMBEDDING_MODEL = 'qwen3-embedding:8b'
python tests/manual_eval/evaluate_v2.py --split new --arms fusion --out .manual/new-original.json
New-Item .manual/section12-selection -ItemType Directory -Force | Out-Null
Copy-Item src .manual/section12-selection/ -Recurse -Force
git apply --directory=.manual/section12-selection tests/manual_eval/section12-selection.patch
$env:PYTHONPATH = (Resolve-Path .manual/section12-selection/src).Path
python tests/manual_eval/evaluate_v2.py --split new --arms fusion --out .manual/new-candidate.json
```
