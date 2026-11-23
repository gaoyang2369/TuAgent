# G120 手册检索证据选择与拒答

默认调用仍使用 `hybrid_concepts`。应用可在 `register_manual_tools` 选择 `mode="evidence_rules"` 或 `mode="evidence_rrf"`。精确查码、系列/控制单元/固件/版次/语言/章节过滤、PDF 页码与输出上限沿用现有契约。新模式的内部排名和决策记录位于工具的 `details.ranking_diagnostics`，模型收到的仍是 `ToolOutput.content` 中的原文和引用。[原 v1 评测](manual-retrieval-eval.md)

## 题集与上线预算

v2 保留 10 道开发题和 38 道盲测题。盲测有 22 个中文无代码可答意图、10 个领域硬负例、2 道精确查码和各 1 道参数、比较、版次冲突、代码冲突题。近似表达同组，同组只出现在一个分区。原 v1 的 42 题、9 道人工英语改写与 3 道追加复核题另作回归。标注从 Siemens A5E33839529 02/2023 英语版核对原页和同段原文。比较题每个对象分别计证据，可接受的替代段落归在相同证据组。[v2 题集](../tests/manual_eval/questions_v2.json) · [v1 回归译写](../tests/manual_eval/v1_translations.json)

题集验哈希后，先冻结上线预算。中文无代码可见证据净增至少 3 道，硬负例误返回至多 1 道且不高于现默认，预热端到端 P95 至多 2,500 ms，检索服务完成率至少 95%。P50/P95 用最近秩法，纳入超时，排除预热和精确查码。候选 Hit@20 在拒答前计算，Top-5 支持率读取实际模型消息。可答题误拒、错码、错页和服务错误单列。[冻结口径](../tests/manual_eval/protocol_v2.json)

## 实验路径

- `default` 和 `frozen_rrf` 复现上一阶段的规则与冻结融合。两条增加评测专用候选追踪。
- `rules_window` 和 `rules_items` 使用同一套确定性规则修复和证据门槛。前者保留旧摘录窗口，后者选择原文完整句或列表项，用于单列摘录变化。
- `fusion` 分别取作用域内的规则与完整向量排名，多路按 `passage_id` 去重后做 RRF，再逐条检验证据属性。可能原因仍来自英文原段。每条原文限 280 字符、24 个词，比较题每组均有证据才返回。
- `fixed_terms` 在原中文向量上增加英文词项。`fixed_english_vector` 在同一固定英语词项路径上额外编码这条英语问法，两路向量通过排名融合。不触发改写时仍使用原中文。对明确代码、范围冲突、否定、比较、数字、单位和参数等容易改变含义的输入保持原问。改写保留原有英文技术词，候选改写校验新增代码、数字、单位及词项。
- `rerank` 和 `fixed_and_rerank` 用原问与标题、段类型及原段成对，在已经融合的同一候选集上进行一次批量打分。拒答独立检查所问属性、段类型和原文证据，成对相关性分数用于改排序。上线评测关闭服务失败回退，实际模式与失败逐题记录。应用可用 `EvidenceOptions(fallback=True)` 显式开启降级。

内网自动改写器只接收原问题与命中的允许术语。它在开发集与固定译写对照，正式配置由开发数据选定。入口校验当前代码、题集、PDF、语料、向量矩阵和服务模型与冻结清单一致，正式盲测禁止覆盖排序参数。[查询改写论文](https://aclanthology.org/2023.emnlp-main.322/) · [Qwen3 成对评分模型卡](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)

## 基础复现

从仓库根目录运行，索引和向量步骤见 v1 记录。下面的单题均来自已公开回归集。[安装与旧索引复用](manual-retrieval-eval.md)

```powershell
$env:OLLAMA_BASE_URL = 'http://10.108.13.254:11434'
python -m unittest discover -s tests -v
python tests/live_manual.py --pdf .manual/g120-cu240be-2-0223-en.pdf --index-dir .manual/index --evidence-v2
python tests/manual_eval/evaluate_v2.py --split regression --min-anchors 2 --arms default rules_window rules_items --out .manual/v2/regression.json
python tests/manual_eval/evaluate_v2.py --split regression --min-anchors 2 --case H07 H24 H11 D02 R02 --arms frozen_rrf fusion fixed_terms --out .manual/v2/known-cases.json
```

回归中的 D02、H07、R02 在先融合再判定的路径中分别以 F30002 第 905 页原因、F30011 第 906 页处理和 F07900 第 885 页证据在首位通过。H24 仍为 `no_results`。H11 的 F30001 第 904 页段在原规则已命中，但旧摘录漏掉电缆长度。真实 PDF 的脚本模型在 `Agent.run` 消息里收到了完整原列表项，控制单元与固件适用性也一同返回。[Siemens 原手册第 904 页](https://cache.industry.siemens.com/dl/files/919/109817919/att_1133702/v1/G120_CU240BE-2_list_man_0223_en-US.pdf#page=904)

## 开发选型与正式配置

开发集对照每路取 20 与 40、要求 1 与 2 个主题锚点。选择每路 20、至少 2 个主题锚点。两种深度在完成的固定词项开发实验中可见证据支持均为 3/6，40 个候选没有观察到收益。未加改写的最终开发配置为 3/6，中文无代码 2/4，硬负例误返回 0/3。重排与重排加固定词项在开发集均为 3/6，各有一次服务超时。

内网 `glm-4.7-flash-46k:latest` 的自动改写样本为 D201、D202、D204。前两题发起改写，输出都超出保真校验允许的术语范围，回原问。D204 保留带 `U/f` 的原问。三次检索都完成，两个已改写尝试的端到端时长约 10.95 秒和 9.46 秒。正式盲测比较固定词项与增添英文向量，无自动改写臂。

本地成对打分采用 ggml-org 转换的 Qwen3-Reranker-0.6B Q8_0 和 llama.cpp b12540 Vulkan。以两条原段做对照，H07 的正确保险丝段和错误过压诊断值分数分别约为 0.992 和 0.0001；H24 询问上门费用，过压诊断值分数约 0.979，仍缺少费用证据。两组相关性分数用于排序，费用原问由证据门槛拒答。[固定 GGUF 版本](https://huggingface.co/ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/tree/a02f48bb4f057028298c21fa033da2b30d7742d5) · [llama.cpp b12540](https://github.com/ggml-org/llama.cpp/releases/tag/b12540)

以下命令复用索引和模型文件，缺失文件才下载。正式入口在运行前校验冻结清单。输出路径需为新文件。[运行清单与哈希](../tests/manual_eval/freeze_v2.json)

```powershell
New-Item -ItemType Directory -Force .manual/v2 | Out-Null
$zip = '.manual/v2/llama-vulkan.zip'
$weights = '.manual/v2/rerank-ggml.gguf'
$exe = '.manual/v2/llama-vulkan/llama-server.exe'
$modelHash = '22c9979ce4fbcdc5acdc310c6641c32797eff1aa980b8f7a2db8a8ea23429a48'
if (-not (Test-Path $exe)) {
    curl.exe -fL -o $zip https://github.com/ggml-org/llama.cpp/releases/download/b12540/llama-b12540-bin-win-vulkan-x64.zip
    if ($LASTEXITCODE -ne 0) { throw 'llama.cpp download failed' }
    Expand-Archive -LiteralPath $zip -DestinationPath .manual/v2/llama-vulkan -Force
}
if (-not (Test-Path $weights)) {
    curl.exe -fL -o $weights https://huggingface.co/ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/resolve/a02f48bb4f057028298c21fa033da2b30d7742d5/qwen3-reranker-0.6b-q8_0.gguf
    if ($LASTEXITCODE -ne 0) { throw 'Qwen3 reranker download failed' }
}
if ((Get-FileHash $weights).Hash.ToLower() -ne $modelHash) { throw 'Unexpected reranker model' }
$alias = 'qwen3-reranker-' + $modelHash
$serverArgs = @('--model',(Resolve-Path $weights).Path,'--rerank','--ctx-size','24576','--parallel','12',
    '--threads','6','--threads-batch','8','--batch-size','4096','--ubatch-size','2048',
    '--device','Vulkan0','--gpu-layers','99','--host','127.0.0.1','--port','18085','--alias',$alias)
$server = Start-Process -FilePath $exe -ArgumentList $serverArgs -WindowStyle Hidden -PassThru

# Wait for http://127.0.0.1:18085/health to return {"status":"ok"}.
python tests/manual_eval/evaluate_v2.py --split blind --reranker-binary $exe --reranker-weights $weights --out .manual/v2/blind-replay.json
```

## 正式盲测结果

同一次冻结盲测含 36 次检索和 2 次精确查码，每路均通过 2/2 次精确查码。检索可答题 24 道，其中中文无代码 22 道。候选是进入拒答前的每路 Top-20 并集。英文向量臂额外检索的 Top-20 也纳入并集。本次运行的全部原始排名与拒答轨迹在工作区的 `.manual/v2/blind.json`。[逐题结果、原问、改写、正确候选阶段排名和可见原文](../tests/manual_eval/blind_results_v2.json)

| 路径 | 中文候选 Hit@20 | 中文最终 Hit@1 / @5 | 中文摘录支持@5 | 全部可答 Hit@1 / @5 | 全部可答摘录支持@5 | 可答误拒 | 硬负例误返回 / 完成 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 默认规则 | 5/22 | 0/22 · 1/22 | 0/22 | 0/24 · 1/24 | 0/24 | 17/24 | 3/10 |
| 冻结 RRF | 22/22 | 3/22 · 3/22 | 0/22 | 3/24 · 3/24 | 0/24 | 19/24 | 1/10 |
| 修复规则＋旧窗口 | 5/22 | 0/22 · 2/22 | 1/22 | 0/24 · 4/24 | 3/24 | 16/24 | 0/10 |
| 修复规则＋完整项 | 5/22 | 0/22 · 2/22 | 1/22 | 0/24 · 4/24 | 3/24 | 16/24 | 0/10 |
| 先融合再判定 | 22/22 | 9/22 · 14/22 | 7/22 | 9/24 · 16/24 | 9/24 | 5/24 | 0/10 |
| 加固定术语 | 22/22 | 10/22 · 14/22 | 7/22 | 10/24 · 15/24 | 8/24 | 5/24 | 0/9 |
| 再加英文向量 | 21/22 | 9/22 · 13/22 | 6/22 | 9/24 · 14/24 | 7/24 | 5/24 | 0/9 |
| 仅加成对重排 | 22/22 | 13/22 · 14/22 | 7/22 | 14/24 · 16/24 | 9/24 | 5/24 | 0/10 |
| 固定术语＋重排 | 22/22 | 12/22 · 13/22 | 6/22 | 13/24 · 15/24 | 8/24 | 5/24 | 0/10 |

| 路径 | 完成 / 发起 | 超时 / 回退 | 首条错码 / 错页 | Top-5 含错码 / 错页 | HTTP tags / embed / rerank | P50 / P95（ms） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 默认规则 | 36/36 | 0 / 0 | 5 / 0 | 6 / 0 | 0 / 0 / 0 | 1.59 / 10.57 |
| 冻结 RRF | 36/36 | 0 / 0 | 0 / 0 | 0 / 0 | 68 / 34 / 0 | 1,035.54 / 1,215.10 |
| 修复规则＋旧窗口 | 36/36 | 0 / 0 | 5 / 0 | 7 / 0 | 0 / 0 / 0 | 1.65 / 21.72 |
| 修复规则＋完整项 | 36/36 | 0 / 0 | 5 / 0 | 7 / 0 | 0 / 0 / 0 | 1.74 / 21.84 |
| 先融合再判定 | 36/36 | 0 / 0 | 6 / 1 | 17 / 1 | 68 / 34 / 0 | 1,091.87 / 1,221.52 |
| 加固定术语 | 34/36 | 2 / 0 | 4 / 0 | 16 / 1 | 67 / 33 / 0 | 1,139.11 / 10,043.17 |
| 再加英文向量 | 33/36 | 3 / 0 | 4 / 0 | 15 / 1 | 87 / 42 / 0 | 1,188.86 / 10,053.62 |
| 仅加成对重排 | 36/36 | 0 / 0 | 4 / 0 | 17 / 1 | 68 / 34 / 34 | 7,776.93 / 11,317.85 |
| 固定术语＋重排 | 35/36 | 1 / 0 | 4 / 0 | 16 / 1 | 67 / 33 / 34 | 8,695.48 / 20,147.91 |

计时采用 Windows、Python 3.13.13、内网 Ollama 查询请求 24 线程、本机 AMD Radeon Vulkan。各臂各预热一次，所有端到端分位数含服务失败。各臂的两道范围负例均直接返回约束状态。固定术语各臂应用 11/36 次，校验漂移 0 次，额外英文向量未提高候选并集覆盖。带重排与对应不带重排臂的融合候选集合在成功共有请求中逐项一致，原问对照 34 组、固定词项对照 31 组。超时均出在向量服务，分别为固定词项 N10/X03、额外英文向量 B04/N10/X03、固定词项＋重排 B03。未重跑或替换盲测数值。

### 收益与失分

- **确定性修复。** 在 34 道可答回归题上，默认规则、修复＋旧窗、修复＋完整项的可见证据通过数为 17、18、19。领域无答案回归误返回从 6/9 到 0/9。盲测的完整项和旧窗口同为 3/24，H11 是回归中由完整项修复的典型例子。默认配置的 42 个 v1 公共工具结果与提交 `d7720f0` 的模型可见内容逐字一致。
- **融合。** B03 找到 F01040 第 826 页的两条完整操作，X04 在第 906 和 907 页覆盖断线与短路两项。7 道中文支持题是 B03、B04、B10、B12、B14、B17、B19。B01、B02、B11、B18、B22 在拒答前均有标注段，最终仍误拒。B05 检索到第 871 页段，摘录漏了 50 ms；B20 在第 911 页选出预充电电容过大的条目，原问询问预充次数过多。两题均算摘录不支持。
- **重排。** B10 的 F07802 第 881 页、B14 的 A07920 第 887 页、B17 的 F30012 第 906 页和 X03 的 F01040 第 826 页都提升到首位。B05 和 B20 的正确段落也前移，但英文摘录仍缺少所问属性。重排的证据支持总数持平。
- 错码、错页按冻结标签计分。事后核对 X03 的 p0971，A01028 第 823 页也有有效的保存参数说明。冻结标签只纳入第 826 页，正式计分保持原样。当前证据门槛使用词项、主题和段类型；上下文中提到时间或密码时也可能被识别为现场属性，需与真实问题样本继续校验。

## 路由建议

先融合再判定的单路满足事先冻结的四项预算。建议只对 **G120 英文第 4.2 节中的中文无代码问题** 试点 `mode="evidence_rrf"`，明确故障码继续走精确查询或按码取证，全局默认保留 `hybrid_concepts`。已测的固定改写、英文向量和重排无需放进试点默认组合。上线前仍需复核其他版本、规则触发歧义、并发长尾以及模型生成回答时的引用使用。

```python
from tuagent.manual import EvidenceOptions, register_manual_tools

register_manual_tools(
    registry,
    index_dir=".manual/index",
    mode="evidence_rrf",
    evidence=EvidenceOptions(depth=20, min_anchors=2),
)
```
