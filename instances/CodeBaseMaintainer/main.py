# 代码库维护助手
# 整合 ContextBuilder + NoteTool + TerminalTool + MemoryTool

from dotenv import load_dotenv
load_dotenv()
from typing import Dict, List, Any, Optional
from datetime import datetime
import inspect
import json

from hello_agents import SimpleAgent, HelloAgentsLLM
from hello_agents.context import ContextBuilder, ContextConfig, ContextPacket
from hello_agents.tools import MemoryTool, NoteTool, TerminalTool
from hello_agents.core.message import Message


class CodeBaseMaintainer:
    """代码库维护助手 --- 长程智能体示例"""

    def __init__(
        self,
        project_name: str,
        codebase_path: str,
        llm: Optional[HelloAgentsLLM] = None
    ):
        self.project_name = project_name
        self.codebase_path = codebase_path
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.llm = llm or HelloAgentsLLM()

        # 初始化工具
        # 简易测试仅启用本地工作记忆；跨会话状态由 NoteTool 持久化。
        # 这样无需在启动阶段连接 Qdrant、Neo4j 或下载嵌入模型。
        self.memory_tool = MemoryTool(
            user_id = project_name,
            memory_types = ["working"]
        )
        self.note_tool   = NoteTool(workspace=f"./{project_name}_notes")
        self.terminal_tool = TerminalTool(workspace = codebase_path, timeout = 30)

        #初始化上下文工具
        self.context_builder = ContextBuilder(
            memory_tool = self.memory_tool,
            rag_tool = None, # 本处案例不涉及 RAG
            config = ContextConfig(
                max_tokens = 4000,
                reserve_ratio = 0.15,
                min_relevance = 0.2,
                enable_compression = True
            )
        )
        self.conversation_history: List[Message] = []
        self.stats = {
            "session_start": datetime.now(),
            "commands_executed": 0,
            "notes_created": 0,
            "issues_found":  0
        }
        print(f"✅ 代码库助手初始化完成: {project_name}")
        print(f"📂 工作目录: {codebase_path}")
        print(f"会话ID: {self.session_id}")


    def run(self, user_input: str, mode: str = "auto") -> str:
        """
            运行助手

            Args:
                - user_input
                - mode: 运行模式
                    - "auto"
                    - "explore": 代码探索
                    - "analyze": 侧重问题分析
                    - "plan": 侧重问题规划
            Returns:
                - str: 助手回答
        """
        print(f"\n{'='*80}")
        print(f"用户: {user_input}")
        print(f"{'='*80}\n")

        # 1. 根据模式预处理
        pre_context = self._preprocess_by_mode(user_input, mode)

        # 2. 检索相关笔记
        relevant_notes = self._retrieve_relevant_notes(user_input)
        note_packets   = self._notes_to_packets(relevant_notes)

        # 3. 构建相关上下文
        # hello-agents 不同版本使用 additional_packets/custom_packets 等参数名。
        packets = note_packets + pre_context
        system_instructions = self._build_system_instructions(mode)
        build_parameters = inspect.signature(self.context_builder.build).parameters
        build_kwargs: Dict[str, Any] = {
            "user_query": user_input,
            "conversation_history": self.conversation_history,
        }

        if "system_instructions" in build_parameters:
            build_kwargs["system_instructions"] = system_instructions
            system_key = "system_instructions"
        elif "system_instruction" in build_parameters:
            build_kwargs["system_instruction"] = system_instructions
            system_key = "system_instruction"
        else:
            raise TypeError("当前 ContextBuilder.build() 不支持系统指令参数")

        packet_key = next(
            (
                name
                for name in ("additional_packets", "custom_packets", "custom_packet")
                if name in build_parameters
            ),
            None,
        )
        if packet_key:
            build_kwargs[packet_key] = packets
        elif packets:
            # 不支持独立信息包的旧版本，退化为将内容并入系统指令。
            packet_context = "\n\n".join(packet.content for packet in packets)
            build_kwargs[system_key] += f"\n\n[补充上下文]\n{packet_context}"

        context = self.context_builder.build(**build_kwargs)

        # 4. 调用llm
        print("🤖 llm 正在思考")
        response = self.llm.invoke([
            {"role": "system", "content": context},
            {"role": "user", "content": user_input},
        ])

        # 5. 后处理
        self._postprocess_response(user_input, response)

        # 6. 更新历史
        self._update_history(user_input, response)
        print(f"\n🤖 助手: {response}\n")
        print(f"{'='*80}\n")

        return response


    def _preprocess_by_mode(self, user_input: str, mode: str) -> List[ContextPacket]:
        """根据模式对用户输入预处理"""
        packets = []

        if mode == "explore" or mode == "auto":
            # 自动查看项目结构
            print("🔍 探索代码库结构...")

            structure = self.terminal_tool.run({
                "command": "find . -type f -name '*.py' | head -n 20"
            })
            self.stats["commands_executed"] += 1

            packets.append(ContextPacket(
                content = f"[代码库结构]\n{structure}",
                timestamp = datetime.now(),
                token_count = len(structure) // 4,
                relevance_score = 0.6,
                metadata = {"type": "code_structure", "source": "terminal"}    
            ))

        if mode == "analyze":
            print("分析代码质量中 ... ")

            # 代码行数
            loc = self.terminal_tool.run({
                "command": "find . -name '*.py' -exec wc -l {} + | tail -n 1"
            })

            # 查找 TODOS 和 FIXME
            todos = self.terminal_tool.run({
                "command": "grep -rn 'TODO\\|FIXME' --include='*.py' | head -n 10"
            })
            self.stats["commands_executed"] += 2

            packets.append(ContextPacket(
                content = f"[代码统计]\n{loc}\n\n[待办事项]\n{todos}",
                timestamp = datetime.now(),
                token_count = (len(loc) + len(todos)) // 4,
                relevance_score = 0.7,
                metadata = {"type": "code_analysis", "source": "terminal"}
            ))

        if mode == "plan":
            """规划模式: 加载最近的笔记"""
            print("📋 加载任务规划...")

            task_notes = self.note_tool.run({
                "action": "list",
                "note_type": "task_state",
                "limit": 3
            })

            if task_notes and "暂无笔记" not in task_notes:
                content = task_notes
                packets.append(ContextPacket(
                    content=f"[当前任务]\n{content}",
                    timestamp=datetime.now(),
                    token_count=len(content) // 4,
                    relevance_score=0.8,
                    metadata={"type": "task_plan", "source": "notes"}
                ))

        return packets


    def _retrieve_relevant_notes(self, query: str, limit: int = 3) -> List[str]:
        """检索相关笔记"""
        try:
            # 优先检索 blocker
            blockers = self.note_tool.run({
                "action": "list",
                "note_type": "blocker",
                "limit": 2
            })
            search_results = self.note_tool.run({
                "action": "search",
                "query": query,
                "limit": limit
            })
            results = []
            if blockers and "暂无笔记" not in blockers:
                results.append(f"[阻塞笔记]\n{blockers}")
            if search_results and "未找到匹配" not in search_results:
                results.append(f"[相关笔记]\n{search_results}")
            return results[:limit]
        except Exception as e:
            print(f"[WARNING] 笔记检索失败: {e}")
            return []


    def _notes_to_packets(self, notes: List[str]) -> List[ContextPacket]:
        """将笔记转化为packet"""
        packets = []

        for note in notes:
            packets.append(ContextPacket(
                content = note,
                timestamp = datetime.now(),
                token_count = len(note) // 4,
                relevance_score = 0.8,
                metadata = {
                    "type": "note",
                    "source": "note_tool"
                }
            ))

        return packets


    def _build_system_instructions(self, mode: str) -> str:
        """构建系统指令"""
        base_instructions = f""" 你是 {self.project_name} 项目的代码库维护助手。

你的核心能力:
1. 使用 TerminalTool 探索代码库(ls, cat, grep, find等)
2. 使用 NoteTool 记录发现和任务
3. 基于历史笔记提供连贯的建议

当前会话ID: {self.session_id}
"""

        mode_specific = {
            "explore": """
当前模式: 探索代码库

你应该:
- 主动使用 terminal 命令了解代码结构
- 识别关键模块和文件
- 记录项目架构到笔记
""",
            "analyze": """
当前模式: 分析代码质量

你应该:
- 查找代码问题(重复、复杂度、TODO等)
- 评估代码质量
- 将发现的问题记录为 blocker 或 action 笔记
""",
            "plan": """
当前模式: 任务规划

你应该:
- 回顾历史笔记和任务
- 制定下一步行动计划
- 更新任务状态笔记
""",
            "auto": """
当前模式: 自动决策

你应该:
- 根据用户需求灵活选择策略
- 在需要时使用工具
- 保持回答的专业性和实用性
"""
        }
        return base_instructions + mode_specific.get(mode, mode_specific["auto"])


    def _postprocess_response(self, user_input: str, response: str):
        """后处理: 分析回答, 自动记录重要信息"""

        # Notes: 发现问题
        if any(keyword in response.lower() for keyword in ["问题", "bug", "错误", "阻塞"]):
            try:
                self.note_tool.run({
                    "action": "create",
                    "title": f"发现问题: {user_input[:30]}...",
                    "content": f"## 用户输入\n{user_input}\n\n## 问题分析\n{response[:500]}...",
                    "note_type": "blocker",
                    "tags": [self.project_name, "auto_detected", self.session_id]
                })
                self.stats["notes_created"] += 1
                self.stats["issues_found"]  += 1
                print("📒 已自动创建问题笔记")
            except Exception as e:
                print(f"[WARNING] 创建笔记失败: {e}")

        # 如果是任务规划,自动创建 action 笔记
        elif any(keyword in user_input.lower() for keyword in ["计划", "下一步", "任务", "todo"]):
            try:
                self.note_tool.run({
                    "action": "create",
                    "title": f"任务规划: {user_input[:30]}...",
                    "content": f"## 讨论\n{user_input}\n\n## 行动计划\n{response[:500]}...",
                    "note_type": "action",
                    "tags": [self.project_name, "planning", self.session_id]
                })
                self.stats["notes_created"] += 1
                print("📝 已自动创建行动计划笔记")
            except Exception as e:
                print(f"[WARNING] 创建笔记失败: {e}")


    def _update_history(self, user_input: str, response: str):
        """更新对话历史"""
        self.conversation_history.append(
            Message(content = user_input, role = "user", timestamp = datetime.now())
        )
        self.conversation_history.append(
            Message(content = response, role = "assistant", timestamp = datetime.now())
        )
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]

    # === 简便方法实现 ===
    def explore(self, target: str = ".") -> str:
        return self.run(f"请探索 {target} 的代码结构", mode="explore")

    def analyze(self, focus: str = "") -> str:
        """分析代码质量"""
        query = f"请分析代码质量" + (f",重点关注{focus}" if focus else "")
        return self.run(query, mode="analyze")

    def plan_next_steps(self) -> str:
        """规划下一步任务"""
        return self.run("根据当前进度,规划下一步任务", mode="plan")

    def execute_command(self, command: str) -> str:
        """执行终端命令"""
        result = self.terminal_tool.run({"command": command})
        self.stats["commands_executed"] += 1
        return result


    def create_note(
        self, 
        title: str,
        content: str,
        note_type: str = "general",
        tags: List[str] = None
    ) -> str:
        result = self.note_tool.run({
            "action": "create",
            "title": title,
            "content": content,
            "note_type": note_type,
            "tags": tags or [self.project_name] 
        })
        self.stats["notes_created"] += 1
        return result


    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        duration = (datetime.now() - self.stats["session_start"]).total_seconds()

        # 获取笔记摘要
        try:
            note_summary = self.note_tool.run({"action": "summary"})
        except:
            note_summary = {}

        return {
            "session_info": {
                "session_id": self.session_id,
                "project": self.project_name,
                "duration_seconds": duration
            },
            "activity": {
                "commands_executed": self.stats["commands_executed"],
                "notes_created": self.stats["notes_created"],
                "issues_found": self.stats["issues_found"]
            },
            "notes": note_summary
        }


    def generate_report(self, save_to_file: bool = True) -> Dict[str, Any]:
        """生成会话报告"""
        report = self.get_stats()

        if save_to_file:
            report_file = f"maintainer_report_{self.session_id}.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2, default=str)
            report["report_file"] = report_file
            print(f"📄 报告已保存: {report_file}")

        return report
    
