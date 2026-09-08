"""在 Agent 中使用天气 MCP 服务器"""

from dotenv import load_dotenv
from hello_agents import HelloAgentsLLM, SimpleAgent
from hello_agents.tools import MCPTool

# 同时兼容直接运行脚本和以包模块方式导入。
try:
    from .main import weather_server
except ImportError:
    from main import weather_server

load_dotenv()

def create_weather_assistant():
    """创建天气助手"""
    llm = HelloAgentsLLM()

    assistant = SimpleAgent(
        name = "天气助手",
        llm  = llm,
        system_prompt = """你是天气助手, 可以查询城市天气。
        使用get_weather工具查询天气, 支持中文城市名。
"""
    )
    # 同进程内直接连接 FastMCP 实例，仍通过 MCP 协议发现并调用天气工具。
    weather_tool = MCPTool(server = weather_server.mcp)
    assistant.add_tool(weather_tool)

    return assistant

def demo():
    # 此处必须调用工厂函数，获得 SimpleAgent 实例，而不是函数对象本身。
    assistant = create_weather_assistant()

    print("\n查询北京天气")
    response = assistant.run("北京今天天气怎么样?")
    print(f"回答: {response}\n")


def interactive():
    """交互模式"""
    assistant = create_weather_assistant()

    while True:
        user_input = input("\n你: ").strip()
        if user_input.lower() in ['quit', 'exit']:
            break
        response = assistant.run(user_input)
        print(f"助手: {response}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo()
    else:
        interactive()
