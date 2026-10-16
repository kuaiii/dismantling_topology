# AutoSND

AutoSND 是一个简洁、可读的三阶段搜索框架实现，用于发现完整的网络拆解启发式算法。
本仓库仅包含网络拆解（ND）任务：AutoSND 搜索、v18 论文使用的 12 个主要网络、
固定搜索代理 `Powerlaw_500`，以及论文对比所用的 9 个原生 ND 基线。

仓库不包含实验输出、归档的 LLM 响应、冻结候选结果、影响力最大化（IM）代码、
绘图代码或本地凭据。

## 发布范围

- Stage I：从 HDA 根候选出发进行宽泛树搜索。
- Stage II：对每个 Stage-I 归档候选调用 LLM-Struct，将执行证据归纳为结构策略。
- Stage III：在结构策略约束下执行 Q、S、B 和 repair 四类搜索分支。
- 搜索指标：ANC@30% (GCC) 与候选运行时间。
- 部署指标：完整序列 ANC (GCC)。
- 12 个主要网络：CEnew、Collaboration、condmat、crime、email、Grid、hamster、
  HepPh、Yeast、HI-II-14、Digg 和 Gnutella31。
- 合成搜索代理：`Powerlaw_500`，单独存放，不计入 12 个主要网络。
- 原生基线：HDA、CoreHD、DC、CI、GND、MinSum、BPD、KCore 和 CLUC。

GND、MinSum 和 BPD 是公开、可检查的 clean-room Python 近似实现，不是原作者的
官方实现。CLI 会报告其 fidelity 标记，比较结果中不得隐藏这一差别。

## 安装

需要 Python 3.10 或更高版本。

```bash
python -m venv .venv
```

Linux/macOS：

```bash
source .venv/bin/activate
python -m pip install -e .
```

PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

SciPy 是必需依赖，因为 GND 的谱模式属于 9 个公开基线的固定组成部分。

## 配置必需的 LLM 接口

AutoSND 不内置默认 API URL 或 API key。`autosnd search` 使用兼容 OpenAI
chat-completions 的接口，并强制要求同时设置 `AUTOSND_API_URL` 和
`AUTOSND_API_KEY`。缺少任一配置时，搜索会直接报错退出。

Linux/macOS：

```bash
export AUTOSND_API_URL="https://your-provider.example/v1"
export AUTOSND_API_KEY="replace-with-your-key"
export AUTOSND_MODEL="your-model-name"  # 可选：覆盖 YAML 中的模型名
```

PowerShell：

```powershell
$env:AUTOSND_API_URL = "https://your-provider.example/v1"
$env:AUTOSND_API_KEY = "replace-with-your-key"
$env:AUTOSND_MODEL = "your-model-name"  # 可选：覆盖 YAML 中的模型名
```

API key 和 API URL 不会写入运行清单。请勿提交 `.env` 文件；`.env` 和
`*.key` 已加入忽略规则。

## 本地检查

以下命令只检查安装、数据或原生基线，不会启动 AutoSND 搜索，也不会验证 LLM
接口连通性。

在不访问 API 的情况下检查安装和固定搜索代理：

```bash
autosnd doctor --config configs/smoke.yaml --offline
```

验证随仓库提供的 12 个真实网络：

```bash
autosnd datasets --verify
```

运行一个离线原生基线：

```bash
autosnd baseline --method HDA --datasets CEnew crime --rate 1.0
```

将基线结果写入本地忽略目录：

```bash
autosnd baseline \
  --method CoreHD \
  --datasets CEnew email \
  --rate 1.0 \
  --output outputs/corehd.csv
```

## 运行 AutoSND

smoke 配置采用 `(B_1, B_2, B_3)=(4, 2, 4)`，并使用 HDA 根候选：

```bash
autosnd search --config configs/smoke.yaml
```

论文规模的预算配置单独提供：

```bash
autosnd search --config configs/paper.yaml
```

论文配置使用 `(B_1, B_2, B_3)=(300, 10, 200)`，Stage III 种子配额为
`(8, 6, 6, 4)`，分支预算为 `(60, 60, 50, 30)`。完整运行可能产生较高 API
费用，请先检查配置和服务商定价。

YAML 中的 `seed` 控制本地搜索选择，包括 Stage I 的初始探索方向。默认不向
服务商发送该字段，因为部分兼容 OpenAI 的接口会拒绝它。只有服务商明确支持
seeded generation 时，才设置可选环境变量 `AUTOSND_PROVIDER_SEED`；即使设置，
不同服务商或模型版本之间也不保证逐字节一致。

生成物写入 Git 忽略的 `runs/` 目录。每次成功完成的运行记录 Stage I/III 候选、
Stage II 证据与来源、分支种子、非敏感的实际 LLM 请求设置，以及最终选择的
`AutoSND-Q.py` 和 `AutoSND-S.py`。客户端边界会移除凭据和服务商 URL 的具体值，
这些值不会被有意记录。

## Stage II：LLM 结构分析与策略归纳

Stage II 按以下信息流执行：

```text
Stage I 归档 E1
  -> 对每个候选调用 LLM-Struct，得到结构表示 z_i
  -> 正例/负例对照集 C1
  -> 结构统计 T1
  -> B2 次固定模式的 LLM 策略提议 q_j
  -> 确定性合并并限制边界，得到策略 Pi
  -> Stage III 策略约束生成
```

关键规则：

- 每个 Stage-I 归档候选都必须通过一次 LLM-Struct 请求生成 `z_i`。
- `z_i` 记录算法族、结构信号、邻域范围、更新范围、高开销模式和可复用结构签名。
- LLM-Struct 响应必须符合固定 JSON 模式。请求失败、JSON 无效或必需字段缺失时，
  本次搜索立即终止，对应候选不会进入后续结构统计。
- 正例来自领先的 ANC-时间 Pareto 层。
- 负例互斥划分为质量、时间或执行失败三组。
- `T1` 同时保留汇总负面证据和上述三个负例子组。
- 每个采样对照项都标注为正例、负例-质量、负例-时间或负例-失败。
- Stage II 按配置执行 `B_2` 次 LLM 策略提议；这些响应也必须符合
  固定 JSON 模式。未知字段和格式错误的响应不会进入合并；如果没有任何有效提议，
  本次搜索立即终止。
- 数值上限使用中位数合并，并限制在固定安全范围内。
- 编译后策略的每个字段都记录来源。
- Stage III 的 balanced 分支使用持久化的 `z_i` 进行结构多样性选择。

搜索期间严格要求候选输出完整节点排列，不会静默修复缺失节点或重复节点，从而
保持搜索期与部署期的有效性定义一致。

超时候选仍视为无效，但其运行时间按配置的截止值记录，并标记
`runtime_censored=true`。这种截断时间记录不会使超时候选进入有效候选排序。

repair 种子规则按既定协议执行：对 Q 种子和慢候选的并集再次使用 Q 分数排序。
因此，其前 4 个候选可能与 Q 分支的前 4 个候选相同。

## 候选代码执行安全

生成的 Python 代码会经过限制性语法安全检查，并在有超时限制的独立进程中执行。
这属于纵深防护，不是经过加固的安全沙箱。对于不可信端点生成的代码，应在操作系统
级容器或一次性账户中运行。

## 数据与许可证

网络规模和规范化边集的 SHA-256 值记录在
`src/autosnd/data/manifest.json`。12 个真实网络位于
`src/autosnd/data/main12/`，固定合成代理位于
`src/autosnd/data/proxy/`。

使用或再发布网络文件前，必须阅读
[DATA_SOURCES_AND_LICENSES.md](DATA_SOURCES_AND_LICENSES.md)。源研究快照没有
仓库级代码许可证，也没有完整的逐数据集许可证文本，因此本发布版不会自行推定或
创建许可证。公开可访问不代表获得额外权利。

## 开发验证

```bash
python -m pip install -e ".[dev]"
pytest
```

测试覆盖：

- 随仓库提供的全部网络规模、连通性、连续节点 ID 和哈希；
- 固定的 9 个原生基线注册表；
- 原生基线完整排列输出的确定性；
- reverse-DSU ANC 与朴素删除评估器的一致性；
- 候选代码安全检查和超时 worker 执行；
- 生成器物化、截断超时计时和敏感信息移除；
- Stage II 的 LLM 结构响应校验、对照集和上限限制行为；
- 使用本地伪 LLM 客户端完成三阶段端到端测试。

## 仓库结构

```text
configs/                    smoke 配置与论文规模预算
src/autosnd/
  baselines/                9 个原生 ND 基线
  data/main12/              12 个真实基准网络
  data/proxy/               固定的 Powerlaw_500 搜索代理
  candidate.py              候选校验与执行
  datasets.py               网络加载与完整性检查
  metrics.py                严格 ANC (GCC) 评估器
  stage2.py                 LLM 结构分析与证据到策略编译
  search.py                 三阶段搜索编排
tests/                      离线契约测试与端到端测试
```
