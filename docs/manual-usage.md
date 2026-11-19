# G120 手册索引与工具

从仓库根目录运行。当前工作区的源 PDF 与索引保存在 Git 忽略的 `.manual/` 下。下面的命令可在空工作区重建相同语料。只运行手册工具时，索引读取使用 Python 标准库 SQLite。导入 PDF 需要 `manual` 扩展，完整离线套件还需要仓库已有的 `chat-completions` 扩展。

```powershell
python -m pip install -e ".[manual,mysql,chat-completions]"

$root = '.manual'
New-Item -ItemType Directory -Force $root | Out-Null
$pdf = Join-Path $root 'g120-cu240be-2-0223-en.pdf'
$index = Join-Path $root 'index'
$source = 'https://cache.industry.siemens.com/dl/files/919/109817919/att_1133702/v1/G120_CU240BE-2_list_man_0223_en-US.pdf'
$sha256 = 'd6a1e4b9728de4d577b00dd5a81af761c7f57809afbe2557791d3870054912af'
if (-not (Test-Path $pdf)) {
    curl.exe --fail --location --output $pdf $source
    if ($LASTEXITCODE -ne 0) { throw 'Manual download failed' }
}
tuagent-import-manual $pdf --index-dir $index --document-number A5E33839529 `
    --edition '02/2023' --language en-US --device-series G120 `
    --control-unit CU240B-2 --control-unit CU240B-2_DP `
    --control-unit CU240E-2 --control-unit CU240E-2_DP --control-unit CU240E-2_DP_F `
    --control-unit CU240E-2_F --control-unit CU240E-2_PN --control-unit CU240E-2_PN_F `
    --firmware '4.7 SP14' --source-url $source --sha256 $sha256

python -m unittest discover -s tests -v
python tests/live_manual.py --pdf $pdf --index-dir $index
```

导入命令也可用 `python -m tuagent.manual.importer` 调用。它从第 4.2 节书签定位页区间。兼容同类列式布局但缺书签的 PDF 可用 `--pages FIRST:LAST` 指定 1 起算的两端包含页区间。相同 PDF、版次、声明和页区间重复导入返回 `unchanged`。各版次独立保留。导入在临时 SQLite 副本上提交后原子发布，工具在调用时打开已发布的只读索引。

## 接入 Agent

```python
from tuagent import Agent, ToolRegistry
from tuagent.manual import register_manual_tools


async def ask_manual(provider):
    registry = ToolRegistry()
    register_manual_tools(registry, index_dir=".manual/index")
    agent = Agent(provider, tools=registry)
    result = await agent.run("G120 上的 F30002 是什么意思？请注明适用范围和出处。")
    return result.text
```

`provider` 可使用应用现有的模型。可执行示例 `tests/live_manual.py` 使用 `ScriptedModel` 对真实手册索引验证工具消息。只查释义和段落时应用无需 MySQL。需要六个工具时，在同一个 `ToolRegistry` 上再调用 `register_telemetry_tools`。

| 工具 | 参数示例 |
| --- | --- |
| `lookup_drive_code` | `{"device_series":"G120","code":"F30002","control_unit":"CU240E-2","firmware":"4.7 SP14","edition":"02/2023"}` |
| `search_manual` | `{"device_series":"G120","query":"减速时母线电压升高，手册列了哪些可能原因","section":"4.2","limit":3}` |
| `search_manual` | `{"device_series":"G120_SIM","code":"F30002","query":"原因"}` |

- 精确查码保留 F/A 前缀和前导零。空索引、未知码、版次不适用和多版歧义分别返回 `empty_index`、`not_found`、`version_not_applicable` 和 `ambiguous_version`。使用 `G120_SIM` 查询时，真实手册的适用性标为 `candidate_only`，模拟场景释义另附设计文档来源。
- `hybrid_concepts` 先过滤适用手册，再融合关键词与本地中英领域概念。概念表把同义表达归到同一特征，并把减速与能量回馈作为相关特征参与排序。该路径可全程离线运行，覆盖概念由代码内的词表确定。用 `register_manual_tools(..., semantic=False)` 可关闭概念路径，结果标为 `keyword_only`。
- `search_manual` 每次最多返回 8 个短摘录，带条目、章节、实际 PDF 页、版次、源文件哈希和可打开链接。证据不足时返回 `no_results`。代码释义的起页可与原因和核查段落的页分开，跨页片段的链接各指向其所在页。

## 向量检索的可选 Ollama 后端

向量检索可以复用内网 Ollama 服务。默认使用本地规则，需要向量模式时配置以下环境变量，并按评测记录构建缓存。[建库与评测命令](manual-retrieval-eval.md) · [Ollama 模型说明](https://ollama.com/library/qwen3-embedding:8b)

```dotenv
OLLAMA_BASE_URL=http://10.108.13.254:11434
EMBEDDING_MODEL=qwen3-embedding:8b
```

开始真实评测前，用 `GET /api/tags` 确认当前运行环境能访问服务且模型已安装。再用 `POST /api/embed` 做一次短文本测试，记录实际向量维度。服务地址使用标准 URL 格式 `http://主机:11434`。建库和查询使用同一个模型版本和维度，缓存与当前手册语料关联。连通性、批量大小、超时和检索延迟以实际测试结果为准。[模型列表接口](https://docs.ollama.com/api/tags) · [向量生成接口](https://docs.ollama.com/api/embed)

可以把下面这段附在向量增强任务的提示词末尾。

> 已有一个部署在另一台内网服务器上的 Ollama Embedding 服务可用于真实评测，地址为 `http://10.108.13.254:11434`，模型为 `qwen3-embedding:8b`。请通过运行环境的 `OLLAMA_BASE_URL` 和 `EMBEDDING_MODEL` 提供连接配置，先核对模型列表并验证一次向量请求。继续使用 TuAgent 现有的 G120 手册段落、页码和 SQLite 索引。向量存储、批量大小和超时根据本项目的实测决定。服务不可达或模型不可用时，完成规则基线和替身测试，记录真实模型验收的阻塞原因。

## 2026-11-19 验收记录

完整离线套件 52 项通过。官方 PDF 哈希与命令中的预期值相符，第 4.2 节 PDF 第 819–936 页导入了 352 个条目和 2,436 个段落。重复导入返回 `unchanged`。离线样本验证了不同版次可分别查询，以及写入或发布失败后旧索引仍可读。

真实手册脚本退出码为 0。模型通过 `Agent.run` 收到 14 条结果，包括释义、控制单元、固件范围和原文引用。

| 代码 | 类型 | 条目起页 | 原因页 |
| --- | --- | --- | --- |
| `F07011` | 故障 | 870 | 870 |
| `F07900` | 故障 | 885 | 885 |
| `A07910` | 报警 | 887 | 887 |
| `F30002` | 故障 | 904 | 905 |

F30002 的原因和核查摘录均逐字落在原 PDF 第 905 页 F30003 标题之前。限定 F30002 搜索欠压返回 `no_results`。无代码的中文减速过压问题使用 `hybrid_concepts`，首条命中 F30002 第 905 页的原因段。番茄浓汤问题无命中。关闭概念路径后，英文回馈关键词仍命中第 905 页，模式为 `keyword_only`。

G120_SIM 对应的实际控制单元和固件仍待确认。真实手册验收覆盖 02/2023 英文版。多版逻辑使用合成版次复测，其他真实版次、语言和扫描件尚无验收样本。

## 向量增强前的代码复核

本次复核以 `79696befd298ba85af71371363c596472fb7e3ae` 为规则基线。全量离线测试 52 项通过，手册模块及测试的 Ruff 检查通过。真实手册脚本退出码为 0，`Agent.run` 经脚本模型完成 14 次工具调用。另用 Poppler 独立提取原 PDF 第 819–936 页，对照 SQLite，352 个 F/A 条目代码和起始页逐项相符。四个目标代码的原因页及 `F30002` 第 905 页的跨页引用通过原文核验。

### 经公共工具复现的行为

使用 `ToolExecutor.execute` 调用 `search_manual`，统一传入 `device_series="G120"`、`limit=5`，检索模式为当前默认的 `hybrid_concepts`。下列问法和近似改写已在开发中可见，后续归入开发集或回归用例。

| 输入 | 当前实测 | 对下一阶段的影响 |
|---|---|---|
| “减速时母线电压升高，手册列了哪些可能原因” | 第一条为 `F30002 / cause / 905` | 保留已命中的术语问法作为对照。 |
| “刹车时直流侧电压往上窜，手册列了哪些原因？” | `no_results`，没有候选证据 | 期望仍为 `F30002 / cause / 905`，可验证向量召回能否补回相同证据。 |
| “F99999 母线电压升高，是什么原因？” | 未传 `code` 时第一条为 `F30002 / cause / 905`。单独传 `code="F99999"` 查询相同现象则是 `no_results` | 自由文本中的未知码可能串到另一个代码，应统一显式代码的处理。 |
| `code="F30002"`，问“如何排查” | `no_results`。换成“核查”可找到第 905 页的 `remedy` | “如何排查”已列于短问白名单，但前一步的意图表达式未识别“排查”。 |
| “母线电压升高会影响食堂菜单的价格吗？” | 返回 `ok`，第一条为 `F30002 / title / 904` | 只匹配术语亦可误答。无答案测试应含领域词重合、但手册无法回答的问题。 |

### 检索实现影响评测的三处细节

- `index.candidates` 用词项倒排表生成候选，并将段落类型限定为 `title`、`cause`、`remedy`、`diagnostic_value`、`cross_reference`。索引共有 2,436 段，其中 1,380 段属于这五类。向量分支应从相同适用范围独立取候选，才能找回倒排表漏掉的段落。若新增可检索类型，应另行评估语料扩容的收益。[候选查询](../src/tuagent/manual/index.py)
- `SearchPlan` 的 `semantic=True` 启用的是手写双语概念、相关词与意图规则。`rank` 将问题中的代码作为加分项，结构化 `code` 参数才参与 SQL 范围过滤。[查询与排序](../src/tuagent/manual/retrieval.py)
- `_excerpt` 用词项和概念在原段内选取短摘录。同代码、同页也可有多个同类段落，例如 `A07910` 第 887 页有两个 `cross_reference`。评测应保存可接受的英文证据短语或稳定段落标识，核对模型实际收到的摘录。[摘录与模型可见结果](../src/tuagent/manual/tools.py)

### Embedding 服务预检

内网 Ollama 的 `/api/tags` 返回 `qwen3-embedding:8b`，模型 digest 为 `64b933495768fbd3b87c20583d379728a07471e0c66733a9df87cd1901b3c44b`。调用 `/api/embed` 批量传入一条中文问题和一条英文短句，收到两条 4096 维向量，元素均为有限数，二范数保留六位小数均为 1。该次预检验证了接口、批量和维度。全库召回质量及 P95 延迟由设计文档第 9 节的对照实验测量。查询侧任务指令和文档侧输入应分别记录。[Qwen 官方模型卡](https://huggingface.co/Qwen/Qwen3-Embedding-8B) · [Ollama 向量接口](https://docs.ollama.com/api/embed)
