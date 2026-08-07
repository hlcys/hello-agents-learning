# TerminalTool、 MemoryTool、NoteTool 和 ContextBuilder 的协同使用。

# (1) 与MemoryTool协同使用
# 使用 Terminaltool发现项目结构
structure = terminal.run({"command": "tree -L 2 src"})
# 存储到语义记忆中
memory_tool.run({
    "action":  "add",
    "content": f"项目结构: \n{structure}",
    "memory_type": "semantic",
    "importance":  0.8,
    "metadata": {"type": "project_structure"}
})

# (2) 与NoteTool协同
log_analysis = terminal.run({"command": "grep 'slow query' app.log | tail -n 10"})
note_tool.run({
    "action": "create",
    "content": "数据库慢查询问题",
    "content": f"## 问题描述\n发现多个慢查询,影响系统性能\n\n## 日志分析\n```\n{log_analysis}\n```\n\n## 下一步\n1. 分析慢查询SQL\n2. 添加索引\n3. 优化查询逻辑",
    "note_type": "blocker",
    "tags": ["performance", "database"]
})

# 探索代码库
code_structure = terminal.run({"command": "ls -R src"})
recent_changes = terminal.run({"command": "git log --oneline -10"})

# 转换为 ContextPacket
from hello_agents.context import ContextPacket
from datetime import datetime

packets = [
    ContextPacket(
        content=f"代码库结构:\n{code_structure}",
        timestamp=datetime.now(),
        token_count=len(code_structure) // 4,
        relevance_score=0.7,
        metadata={"type": "code_structure", "source": "terminal"}
    ),
    ContextPacket(
        content=f"最近提交:\n{recent_changes}",
        timestamp=datetime.now(),
        token_count=len(recent_changes) // 4,
        relevance_score=0.8,
        metadata={"type": "git_history", "source": "terminal"}
    )
]

# 在构建上下文时包含这些信息
context = context_builder.build(
    user_query="如何重构用户服务模块?",
    custom_packets=packets
)