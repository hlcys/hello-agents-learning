# test: initial rag tool

from dotenv import load_dotenv
load_dotenv()

from hello_agents import SimpleAgent, HelloAgentsLLM, ToolRegistry
from hello_agents.tools import MemoryTool, RAGTool

llm = HelloAgentsLLM()
agent = SimpleAgent(
    name = "智能助手",
    llm = llm,
    system_prompt = "你是一个有记忆和知识检索能力的AI助手"
)

tool_registry = ToolRegistry()
tool_registry.register_tool(MemoryTool(user_id = "user_123"))
tool_registry.register_tool(RAGTool(knowledge_base_path = "./knowledge_base"))

agent.tool_registry = tool_registry

response = agent.run("你好! 请记住我叫张三, 我是一名Python开发者")
print(response)
