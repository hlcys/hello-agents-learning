# test_react_agent 

from dotenv import load_dotenv
from hello_agents import HelloAgentsLLM, ToolRegistry
from hello_agents.tools import CalculatorTool
from core.my_react_agent import MyReactAgent

load_dotenv()

llm = HelloAgentsLLM()

print("=== 测试1: 基础对话 ===")
react_agent = MyReactAgent(
    name = "基础助手",
    llm  = llm,
    system_prompt = "你是一个友好的AI助手, 请用简洁明了的方式回答问题。",
    tool_registry = ToolRegistry()
)

response1 = react_agent.run("请介绍下你自己")
print(f"基础对话响应: {response1}\n")

print("=== 测试2: 工具增强对话 ===")
tool_registry = ToolRegistry()
calculator    = CalculatorTool()
tool_registry.register_tool(calculator)

enhanced_agent = MyReactAgent(
    name = "增强助手",
    llm  = llm,
    system_prompt = "你是一个智能助手，可以使用工具来帮助用户。",
    tool_registry = tool_registry,
    enable_tool_calling = True
)

response2 = enhanced_agent.run("请计算 12.21 + 2356.5")
print(f"工具响应：{response2}\n")

# 测试3:流式响应
print("=== 测试3:流式响应 ===")
if hasattr(react_agent, "stream_run"):
    print("流式响应: ", end="")
    for chunk in react_agent.stream_run("请解释什么是人工智能"):
        pass  # 内容实时打印
else:
    print("MyReactAgent 当前不支持 stream_run，跳过该测试。")
