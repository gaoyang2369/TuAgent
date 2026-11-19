# G120 中英手册检索评测

## 冻结题集与范围

题集 `g120-zh-en-v1` 共 42 题、39 个语义组。开发集 17 题，包含已有漏检及近似改写。保留集 25 题，含 14 道可答检索题、8 道负例和 3 道精确查码题。题集以 LF 计算 SHA-256，值为 `625d8b6c7e00fa1328ff32983ef8672ce6277ea4dda635c9fe9bb0a638ceb861`。[分组和逐题证据](../tests/manual_eval/questions.json) · [修改检索前保存的开发集基线](../tests/manual_eval/baseline-dev.json)

原规则使用提交 `79696befd298ba85af71371363c596472fb7e3ae`。四路都使用 A5E33839529、02/2023、en-US、CU240E-2、4.7 SP14、第 4.2 节的同一份 PDF，SHA-256 为 `d6a1e4b9728de4d577b00dd5a81af761c7f57809afbe2557791d3870054912af`。SQLite 中有 352 个条目和 2,436 段。比较限定为原规则开放的 1,380 段，含标题、原因、处理、诊断值、交叉引用，语料指纹为 `3f37d793f726b4d229dc362a0f340dd2853ab3d16184a828bfd4e5e2a52a79b3`。保留集调用在配置冻结后执行。[Siemens 原版手册](https://cache.industry.siemens.com/dl/files/919/109817919/att_1133702/v1/G120_CU240BE-2_list_man_0223_en-US.pdf)

## 安装、建库与复现

从仓库根目录用 PowerShell 运行。PDF 和 SQLite 缺失时先执行手册导入。[手册导入命令](manual-usage.md)

```powershell
python -m pip install -e ".[manual,manual-vector,mysql,chat-completions]"
$env:OLLAMA_BASE_URL = 'http://10.108.13.254:11434'
$env:EMBEDDING_MODEL = 'qwen3-embedding:8b'
$index = '.manual/index'
$pdf = '.manual/g120-cu240be-2-0223-en.pdf'
$env:OLLAMA_EMBED_THREADS = '24'
tuagent-build-manual-vectors --index-dir $index --batch-size 16 --timeout 180
Remove-Item Env:OLLAMA_EMBED_THREADS

python -m unittest discover -s tests -v
python tests/live_manual.py --pdf $pdf --index-dir $index

New-Item -ItemType Directory -Force .manual/eval | Out-Null
git archive --format=zip --output=.manual/eval/baseline-source.zip 79696befd298ba85af71371363c596472fb7e3ae src
Expand-Archive -LiteralPath .manual/eval/baseline-source.zip -DestinationPath .manual/eval/baseline -Force
python tests/manual_eval/calibrate.py --index-dir $index --out .manual/eval/calibration.json
$previousPythonPath = $env:PYTHONPATH
try {
    $env:PYTHONPATH = (Resolve-Path .manual/eval/baseline/src).Path
    python tests/manual_eval/evaluate.py --mode baseline --split dev --verify-labels $pdf --out .manual/eval/baseline-dev-replay.json
    python tests/manual_eval/evaluate.py --mode baseline --split holdout --out .manual/eval/baseline-holdout.json
} finally {
    $env:PYTHONPATH = $previousPythonPath
}
foreach ($mode in 'hybrid_concepts', 'vector', 'rrf') {
    python tests/manual_eval/evaluate.py --mode $mode --split holdout --out ".manual/eval/$mode-holdout.json"
}
python tests/live_manual.py --pdf $pdf --index-dir $index --vector-mode rrf --vector-query 'F30002 的原因'
```

题号 `d02` 是“刹车时直流侧电压往上窜”。命令输出模型实际收到的 Top-5、摘录和判定。用 `--case h03` 可复现保留集单题。

```powershell
foreach ($mode in 'hybrid_concepts', 'vector', 'rrf') {
    python tests/manual_eval/evaluate.py --mode $mode --case d02 --diagnostics --out ".manual/eval/$mode-d02.json"
}
python tests/manual_eval/evaluate.py --mode rrf --case d02 --rrf-min-similarity 0.625 --diagnostics --out .manual/eval/rrf-d02-relaxed.json
```

## 检索与计分口径

- 应用通过 `register_manual_tools(registry, index_dir=".manual/index", mode="rrf")` 选择模式。另可选 `vector`、`hybrid_concepts` 和 `keyword_only`。旧用法 `semantic=False` 仍选择关键词。规则运行可只安装原有依赖。
- 文档侧编码代码、英文标题、段落类型和原文。查询侧以 `Instruct: …\nQuery: …` 拼接固定英文任务与用户原问。两套模板、模型 digest、维度及语料指纹记录在缓存清单。批量生成经维度与数值校验后归一化为 float32，原子发布到本地单一文件。中断后可用相同命令从已校验的批次续跑。运行时先核对范围和代码，再检查缓存指纹和 Ollama digest，计算范围内精确余弦。RRF 按稳定段落标识去重，使用开发集确定的深度及截断条件。[Qwen 模型卡](https://huggingface.co/Qwen/Qwen3-Embedding-8B) · [Ollama 向量接口](https://docs.ollama.com/api/embed)
- 结果包含 `requested_mode`、`actual_mode` 和 `fallback_reason`。范围或代码错误可在任何排序发生前终止。只有显式设置 `fallback_to_rules=True` 才会在向量服务或缓存出错时改走规则。正式对照关闭回退。缓存失配时根据消息运行建库命令。
- Hit@1、Hit@5 和 MRR@5 分母均为可答检索题。多证据题以最后一个必需证据首次出现的排名计分，比较题因此至少需要两条。段落命中核对 PDF 版次、代码、页码、段落类型及标注。同位置多段时通过可见摘录匹配原段。英文摘录支持要求每组最短证据短语确实出现在 `ToolOutput.content` 中。
- “无答案”只含领域内手册无法回答和域外问题。其余负例单独核算范围与状态。错码是首条证据代码不在标注集合，错页是代码对但 PDF 页不在标注集合。Top-5 错码、错页另记在逐题 JSON。查码按状态、释义、范围和页码验收，不进入检索指标。
- 开发集校准覆盖 8 个余弦阈值 0.45 至 0.75，RRF 另外组合 12 / 24 候选深度、10 / 20 排名常数和 1.0 / 1.5 / 2.0 向量权重。按 Top-5 摘录支持数扣除两倍负例误命中和两倍范围失败计分。平分时依次优先中文无代码摘录支持、首条支持、Top-5 命中及 MRR。校准复用开发查询向量，保留集测量关闭该复用。
- 冻结的纯向量阈值为 0.70。RRF 每路深度 12，常数 20，规则权重 1，向量权重 1.5，向量闸值 0.75。带码组可用已按码约束的规则证据。无代码时也接受两路前三名共同命中且余弦达到 0.70 的同一段落。融合基于排名计分，证据门槛另行判定。[保留集冻结清单](../tests/manual_eval/holdout-freeze.json)
- P95 使用每模式同一台客户端顺序调用公共工具，包含参数验证、SQLite、HTTP、排序及模型消息序列化。每模式在计时前跑一次固定预热查询，逐题不复用查询向量。取最近秩 P95，包含范围拒绝的快速调用，排除预热和 3 道精确查码。

## 正式保留集结果

同一台 Windows 客户端、Python 3.13.13、内网 Ollama 0.20.0。四次各 22 道检索请求，包含 14 道可答题、8 道负例。9 道可答题是中文无代码问法。四路精确查码均通过 3 / 3。表中 HTTP 超时保留为失败，回退次数均为 0。[逐题四路 Top-5、模型可见摘录与判定](../tests/manual_eval/holdout-results.json)

| 指标 | 原规则 | 修复后规则 | 纯向量 | RRF |
| --- | ---: | ---: | ---: | ---: |
| Hit@1 | 2 / 14 | 2 / 14 | 2 / 14 | 2 / 14 |
| Hit@5 | 5 / 14 | 6 / 14 | 4 / 14 | 4 / 14 |
| MRR@5 | 0.2381 | 0.2738 | 0.2024 | 0.2143 |
| 中文无代码 Hit@1 / Hit@5 | 0 / 9 · 1 / 9 | 0 / 9 · 1 / 9 | 0 / 9 · 1 / 9 | 0 / 9 · 0 / 9 |
| 中文无代码 Top-5 摘录支持 | 1 / 9 | 1 / 9 | 1 / 9 | 0 / 9 |
| Top-5 英文摘录支持 | 4 / 14 | 5 / 14 | 3 / 14 | 3 / 14 |
| 有正确段落但摘录不足 | 1 | 1 | 1 | 1 |
| 无答案误命中 / 已完成的无答案题 | 3 / 4 | 3 / 4 | 1 / 4 | 0 / 3 |
| 全部负例的误命中数 / 发起数 | 4 / 8 | 3 / 8 | 1 / 8 | 0 / 8 |
| 首条错码 / 错页题数 | 3 / 0 | 3 / 0 | 3 / 0 | 1 / 0 |
| Top-5 含错码 / 错页题数 | 8 / 1 | 8 / 1 | 3 / 0 | 3 / 0 |
| 适用范围过滤失败 | 1 | 0 | 0 | 0 |
| HTTP 超时 | 0 | 0 | 1 | 2 |
| 端到端 P95 | 39.96 ms | 40.35 ms | 2,053.09 ms | 10,094.97 ms |

预热各调用一次，耗时依次为 49.96 ms、50.96 ms、1,489.79 ms 和 2,041.03 ms。正式查询使用 Ollama 默认线程配置，包含模型 digest 核验和查询向量 HTTP 往返，服务已加载模型。未单测冷模型启动、并发负载或其他服务器。P95 取 22 个观测中的第 21 个，用时包含同批超时。纯向量的 H09 是该组最慢请求，RRF 的 H24 和 H02 分别耗时 10,094.97 和 11,781.45 ms。

- **规则修复收益。** H06 比较温度报警与故障，分码检索补齐 A07910 第 887 页和 F07011 第 870 页，增加 1 道命中且摘录足够的题。H19 的未知报警码不再串答。开发题 D04 的带码“如何排查”返回 F30002 第 905 页的处理段。
- **向量增强收益及退步。** 纯向量在 H04 找回 A07910 第 887 页的无传感器温度模型条件，正确证据排第 3，英文摘录支持回答。两种向量模式都把 H12 的堵转参数判据从第 3 提到第 1。纯向量丢失规则命中的 H02、H05、H06。RRF 保住 H06，但少掉 H02、H05，整体较修复后规则少命中 2 道，中文无代码少命中 1 道。
- **拒答与引用。** 纯向量仍误答维修价格 H24，首条为 A30502 第 919 页。RRF 对该题首轮超时，其余 3 道无答案题均未返回证据。用冻结参数单题复核，H09、H02、H24 均恢复响应并返回 `no_results`，耗时 2,020.68、1,974.31、1,796.59 ms，首轮数字保持原样。H11 四路都命中过流原因段第 904 页，模型可见短摘录均漏掉电缆长度的关键短语。

## 刹车时直流侧电压往上窜

开发题 D02 的原规则、修复后规则、纯向量和冻结 RRF 均返回 `no_results`。两个向量模式的阈值前候选一致。F30002 第 905 页的正确原因段排第 4，余弦为 0.627744。放宽到 0.625 的 RRF 同样在第 4 返回该段，英文摘录支持“电机回馈能量过多”。前三条依次为 F07404 第 877 页的处理段、F30002 第 905 页的故障值、A07401 第 876 页的原因段。0.625 仅用于这一开发题的复现。[Siemens 第 905 页](https://cache.industry.siemens.com/dl/files/919/109817919/att_1133702/v1/G120_CU240BE-2_list_man_0223_en-US.pdf#page=905)

真实 PDF 规则验收完成 14 次工具回合。加上 RRF 的 F30002 带码查询，`Agent.run` 共通过 15 次工具回合。脚本模型收到的 RRF 首条是第 905 页原因段，内容逐字位于 F30003 标题之前，附 CU240E-2 / 4.7 SP14 的匹配范围。离线全量 58 项通过。

## 建库与默认模式

内网 `qwen3-embedding:8b` digest 为 `64b933495768fbd3b87c20583d379728a07471e0c66733a9df87cd1901b3c44b`。1,380 × 4,096 的 float32 矩阵为 21.56 MiB，完整缓存为 21.90 MiB。首轮批次请求超时，改用 16 段一批和可续跑检查点。成功建库合计约 36 分钟，其中最后一段过程从 432 段续跑，耗时 1,127.08 秒。同语料和模型重跑返回 `unchanged`，耗时 0.089 秒。

**默认保留修复后的 `hybrid_concepts`。** 本次 RRF 减少了误返回和首条错码，同时降低了可答题命中并引入秒级延迟。向量模式可显式启用做证据探索。扩大中文口语和领域无答案样本、改善摘录选段、复测服务长尾之后，再决定是否用于默认检索。

### 可用于 README

TuAgent 支持带真实 PDF 页码的 G120 中英手册检索。1,380 段、25 道保留题的对照中，修复后规则命中 6 / 14 道可答题，端到端 P95 为 40.35 ms。Ollama 纯向量和 RRF 可按需启用。

### 简历表述

实现基于 1,380 段 Siemens 英文手册的中文规则、Ollama 向量及 RRF 检索，在 25 道独立保留题中将规则 Top-5 命中由 5 / 14 提至 6 / 14，并用真实引用、拒答和 40.35 ms 的规则检索 P95 评估上线取舍。
