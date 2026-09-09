# Hello-Agents 学习项目与笔记

这是我跟随 Hello-Agents 学习过程中积累的代码练习、实验案例和知识笔记，记录从基础 Agent、工具调用，到记忆与 RAG、上下文工程、智能体通信，再到 Agentic RL 的学习过程。

仓库中既有基于 `hello_agents` 第三方包的扩展和使用示例，也有用于理解原理的代码草稿。部分模块尚未完成，适合结合笔记逐个阅读、运行和完善。

## 目录导航

```text
hello-agents/
├── readme.md                  # 项目说明与学习入口
├── docs/                      # 学习笔记
│   ├── memory_system.md       # 记忆与 RAG 系统架构
│   ├── 8_RAG系统.md           # RAG 流程与检索策略
│   ├── 9_上下文工程.md        # 上下文构建与长期任务管理
│   ├── 10_智能体通信.md       # MCP、A2A、ANP
│   └── assets/                # 笔记配图
├── core/                      # Agent、LLM、消息与配置的学习实现
├── tools/                     # 工具接口、注册表、计算器、搜索与工具链
│   └── builtin/               # RAG 工具实现草稿
├── memory/                    # 记忆系统实现草稿
├── context/                   # ContextBuilder 上下文构建器
├── agents/DocAssistant/       # PDF 学习助手实现草稿
├── instances/                 # 综合案例，各案例保留自己的说明与配图
│   ├── CodeBaseMaintainer/    # 代码库维护助手
│   ├── MCPClient/             # MCP 客户端使用练习
│   ├── BuildSelfMCP/          # 自定义天气 MCP 服务
│   ├── SmartDocAssistant/     # GitHub 检索与报告生成助手
│   └── A2A/                   # 多智能体客服协作
├── tests/                     # 功能验证脚本与集成示例
├── Agentic-RL/                # 数据准备、奖励函数、SFT 与 GRPO 训练
└── memory_data/               # 已保留的本地记忆数据
```

`import hello_agents` 使用的是环境中安装的第三方包；`core`、`tools`、`context` 等目录是本仓库的学习代码，两者需要区分。笔记中的框架架构图描述学习对象，不代表本仓库已实现所有模块。

## 学习路线

| 主题 | 笔记与代码入口 | 主要内容 |
| --- | --- | --- |
| 基础 Agent | [core/](core/)、[SimpleAgent 示例](tests/test_simple_agent.py)、[ReAct 示例](tests/test_react_agent.py) | 消息、LLM 封装、对话流程与工具调用 |
| 工具系统 | [tools/](tools/)、[工具链示例](tests/test_tool_chain.py)、[异步执行示例](tests/test_async_tool_executor.py) | 工具注册、参数处理、串行组合与并发执行 |
| 记忆系统 | [记忆架构笔记](docs/memory_system.md)、[记忆示例](tests/test_memory.py) | 工作记忆、情景记忆、语义记忆与感知记忆 |
| RAG | [RAG 笔记](docs/8_RAG系统.md)、[RAG 示例](tests/test_rag.py) | 文档处理、检索增强、多查询扩展与 HyDE |
| 上下文工程 | [上下文笔记](docs/9_上下文工程.md)、[构建器](context/builder.py)、[集成示例](tests/test_note_and_context.py) | 上下文筛选、结构化笔记与长期任务管理 |
| 智能体通信 | [通信协议笔记](docs/10_智能体通信.md)、[协议示例](tests/test_protocols.py) | MCP 工具接入、A2A 协作与 ANP 概念 |
| Agentic RL | [训练笔记](Agentic-RL/readme.md)、[训练配置](Agentic-RL/config.json)、[端到端流程](Agentic-RL/End_to_End_train.py) | 数据格式、奖励函数、SFT、LoRA、GRPO 与评估 |

可以按表格顺序学习：先理解单个 Agent 和工具调用，再学习记忆、检索与上下文管理，最后尝试通信案例和训练实验。

## 综合案例

| 案例 | 入口与说明 | 学习重点 |
| --- | --- | --- |
| PDF 学习助手 | [实现草稿](agents/DocAssistant/base.py) | 将文档问答、记忆与学习统计组合起来，部分实现待补齐 |
| 代码库维护助手 | [说明](instances/CodeBaseMaintainer/readme.md)、[运行入口](instances/CodeBaseMaintainer/test.py) | 组合 ContextBuilder、NoteTool、TerminalTool 和 MemoryTool |
| MCP 客户端 | [说明](instances/MCPClient/readme.md)、[代码](instances/MCPClient/mcp_client.py) | 连接服务、发现和调用工具；脚本中的示例服务及文件路径需自行准备 |
| 自定义 MCP 服务 | [说明](instances/BuildSelfMCP/readme.md)、[服务端](instances/BuildSelfMCP/main.py)、[Agent 调用](instances/BuildSelfMCP/test_with_agent.py) | 封装天气查询并接入 Agent |
| 智能文档助手 | [MCP 版本](instances/SmartDocAssistant/main.py)、[直接调用 API 版本](instances/SmartDocAssistant/main_without_mcp.py)、[示例报告](instances/SmartDocAssistant/report.md) | GitHub 仓库搜索与 Markdown 报告生成 |
| A2A 客服系统 | [说明](instances/A2A/readme.md)、[代码](instances/A2A/main.py) | 接待员、技术专家与销售顾问之间的任务分工 |

## 运行说明

### 环境与依赖

请在安装了 `hello_agents` 的 Python 学习环境中运行。代码还按场景使用 `python-dotenv`、`openai`、`pydantic`、搜索服务客户端及训练相关依赖；仓库目前没有统一的依赖清单和版本锁定文件，需要结合所运行脚本的导入项配置环境。

- 使用 `npx` 启动 MCP 服务的案例需要 Node.js/npm。
- 记忆与 RAG 示例需要按所用框架版本配置嵌入服务和存储后端。
- Agentic RL 本地训练实现需要 CUDA GPU，以及 PyTorch、Transformers、Datasets、TRL、PEFT 等依赖。该实现针对 `hello-agents 0.2.5` 的训练接口做过适配，切换依赖版本时需检查兼容性。

### 模型配置

多数调用模型的示例会通过 `load_dotenv()` 加载配置。在仓库根目录的 `.env` 中填写所用服务的信息，例如：

```dotenv
LLM_MODEL_ID=your-model-id
LLM_API_KEY=your-api-key
LLM_BASE_URL=https://your-provider.example/v1
```

自定义 [MyLLM](core/llm.py) 还支持 `MODELSCOPE_API_KEY` 等提供商配置；[高级搜索工具](tools/my_advanced_search.py) 会读取 `TAVILY_API_KEY`、`SERPAPI_API_KEY`。按实际使用的脚本配置即可，`.env` 已加入 Git 忽略规则。

### 选择一个示例运行

以下命令在仓库根目录执行，使用已配置依赖的 Python 解释器：

```bash
# 基础对话与工具调用：会请求模型服务
python3 -m tests.test_simple_agent

# ReAct Agent：会请求模型服务
python3 -m tests.test_react_agent

# 自定义 MCP 服务的客户端验证：会启动服务并请求天气数据
python3 -m instances.BuildSelfMCP.test_server

# 代码库维护助手：会请求模型服务，并写入案例目录中的 output.txt
python3 instances/CodeBaseMaintainer/test.py

# 查看 Agentic RL 训练参数
python3 Agentic-RL/End_to_End_train.py --help
```

`tests/` 同时包含演示脚本和集成验证，部分脚本在导入时就会执行、调用外部服务或写入本地数据。运行前先阅读对应文件，并逐个选择示例；当前未建立可直接全量运行的离线单元测试套件。

训练环境与模型、数据就绪后，可用 `python3 Agentic-RL/End_to_End_train.py --smoke-test` 执行小规模训练验证。该命令会实际训练和评估，输出位置由配置和脚本决定。

## 整理约定

- 通用学习笔记放在 `docs/`，配图放在 `docs/assets/`，文档使用相对链接。
- 综合案例放在 `instances/`，案例专属说明、图片与示例输出保留在对应目录。
- 训练内容集中在 `Agentic-RL/`；现有日志、训练结果和记忆数据保留，作为学习过程记录。
- 新增代码时按已有目录归类；尚未开始的模块可先记录在笔记中，避免创建无内容的占位文件。
- Python 缓存、虚拟环境、本地模型和新增记忆运行数据由 `.gitignore` 排除；已经纳入版本管理的历史文件继续保留。
