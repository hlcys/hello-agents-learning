# 测试三种通信协议的基本功能

from dotenv import load_dotenv
load_dotenv()

from hello_agents.tools import MCPTool, A2ATool, ANPTool

# MCP
mcp_tool = MCPTool()
result = mcp_tool.run({
    "action": "call_tool",
    "tool_name": "add",
    "arguments": {"a": 10, "b": 20}
})

print(f"计算结果: {result}")

# a2a
a2a_tool = A2ATool("http://localhost:5000")
print("A2A工具创建成功")

# ANP
anp_tool = ANPTool()
anp_tool.run({
    "action": "register_service",
    "service_id": "calculator",
    "service_type": "math",
    "endpoint": "http://localhost:8080"
})
services = anp_tool.run({
    "action": "discover_services",
})
print(f"发现的服务: {services}")