# 工单质量分析助手

客服主管经常需要回答三个连在一起的问题：这段时间有多少工单超时、问题集中在哪类、统计口径是否可信。固定报表能展示预设数字，但临时追问、规则核对和原始记录审计仍要在多个入口间切换。本项目把这些操作组织成一条可追溯的对话流程，使用**完全合成的数据**演示，不涉及真实客户信息。

用户用自然语言提出需求；指标 Skill 按固定规则取数，资料检索给出口径出处，SQL 子流程支持原始记录核对。日期不清楚时先澄清，结果始终关联本轮查询或检索证据。

![合成工单质量复盘的完整中文页面演示](docs/demo.gif)

演示依次展示整体指标、分类追问、规则出处、原始记录审计和日期澄清；画面来自本项目真实网页，等待段按讲解节奏压缩。问题如何被发现、为什么这样设计以及每一步的验证证据，见[设计与故障复盘](docs/DESIGN.md)。

## 能演示什么

| 用户需求 | 系统行为 | 可核对结果 |
| --- | --- | --- |
| 看一段时间的整体超时率 | 校验 UTC 时间范围，调用固定口径的只读指标工具 | 40%（4/10），附分子、分母和规则版本 |
| 沿用条件，按类别追问 | 在当前会话保留日期并改变分组方式 | 账号 3/4、账务 0/2、配送 1/4 |
| 问“未解决工单算不算” | 检索业务规则片段，展示原文和来源编号 | 分母包含未解决的有效工单 |
| 核对原始表最大编号 | 交给 SQL 专项子流程，展示实际执行语句与结果 | `MAX(id)=15`；不会误用指标的业务排除条件 |
| 只说“这周超时率” | 先询问明确的 UTC 起止日期 | 不拿旧会话日期冒充本周数据 |

上述数字来自 15 条人为设计的边界样本。[需求说明](docs/REQUIREMENTS.md)展示用户场景与验收标准；[设计说明](docs/DESIGN.md)解释各环节职责；[演示脚本](example/support_quality/PROJECT_CASE_STUDY.md)给出逐步操作；[完整操作说明](example/support_quality/README.md)列出数据范围和更多反例。

## 本地演示

需要 Windows PowerShell、Python 3.12 和运行中的 [Ollama](https://ollama.com/)；本机模型为 `qwen2.5:3b`。默认使用 BM25 资料检索，无需下载 embedding 模型或配置付费 API。CPU 上模型调用可能耗时数十秒，SQL 多步查询更久。

在仓库根目录执行：

```powershell
ollama pull qwen2.5:3b
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m openchatbi.support_demo
$env:CONFIG_FILE = (Resolve-Path '.local/support-demo/config.local.yaml').Path
.\.venv\Scripts\python.exe run_streamlit_ui.py
```

打开 [本机演示页面](http://127.0.0.1:8501/)，按[演示脚本](example/support_quality/PROJECT_CASE_STUDY.md)依次提问。准备合成数据的命令不会启动网页；如果 `.local/support-demo` 已存在，准备程序会拒绝覆盖。页面沿用仓库原有入口；若安全软件出现告警，应停止并核验，不关闭防护或换入口规避。

每轮回答下方会显示本机实测的总处理与准备耗时；有可展示的执行事件时还会显示首次事件等待。[设计文档中的实测 Demo](docs/DESIGN.md#演示结果与延迟)列出完整操作的结果、逐轮读数与解释；[延迟复现步骤](docs/LATENCY_DEMO.md)给出对比提问。

## 验证范围

在 PowerShell 运行 `pip install -e '.[test]'` 安装测试依赖后，可执行 `.\scripts\test-baseline.ps1` 进行离线回归。离线替身验证代码行为，不等于模型质量；本机实际网页曾核对整体与分类统计、规则来源、会话恢复和只读 SQL 结果。项目尚未测量生产环境的 p95、并发吞吐、端到端超时或真实 API 成本，不把单机演示时间写成性能承诺。

## 来源与许可

本项目基于 [OpenChatBI](https://github.com/zhongyu09/openchatbi)（MIT）进行工单场景改进；保留原许可证和 Git 历史，原有 LangGraph 主图、Text2SQL 子图、检查点及基础页面归原作者，业务指标、规则检索增强、范围与证据约束及中文演示为本项目工作。
