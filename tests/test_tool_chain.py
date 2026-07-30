# tests/test_tool_chain.py

import re

from dotenv import load_dotenv
from hello_agents import HelloAgentsLLM
from core.my_simple_agent import MySimpleAgent
from tools.my_advanced_search import MyAdvancedSearchTool
from tools.my_calculator_tool import my_calculate
from tools.tool_chain_manager import ToolChain, ToolChainManager
from tools.registry import ToolRegistry

load_dotenv()
llm = HelloAgentsLLM()
search_tool = MyAdvancedSearchTool()


def extract_price(search_result: str) -> str:
    """让 LLM 从真实搜索结果中提取单一美元价格。"""
    if search_result.startswith("❌"):
        raise RuntimeError(search_result)

    response = llm.invoke([
        {
            "role": "system",
            "content": (
                "你负责从搜索结果中提取建议零售价。"
                "只返回 PRICE=数字，不要使用货币符号、千位分隔符或其他文字。"
                "例如: PRICE=1999.00。无法确定时返回 NOT_FOUND。"
            ),
        },
        {
            "role": "user",
            "content": search_result,
        },
    ])

    match = re.fullmatch(
        r"\s*PRICE\s*=\s*(\d+(?:\.\d+)?)\s*",
        response,
        flags=re.IGNORECASE,
    )
    if not match:
        raise ValueError(f"无法从搜索结果提取价格, LLM 返回: {response}")
    return match.group(1)


# 2. 注册工具
chain_registry = ToolRegistry()
chain_registry.register_function(
    name="search",
    description="通过真实搜索源查询商品价格",
    func=search_tool.search,
)
chain_registry.register_function(
    name="extract_price",
    description="从搜索结果中提取可计算的美元价格",
    func=extract_price,
)
chain_registry.register_function(
    name="my_calculator",
    description="计算数学表达式",
    func=my_calculate,
)


# 3. 创建工具链
research_chain = ToolChain(
    name = "research_and_calculate",
    description = (
        "先搜索价格，再计算购买总价。"
        "参数格式: input = 搜索内容, quantity = 购买数量"
    )
)
research_chain.add_step(
    tool_name = "search",
    input_template = "{input}",
    output_key = "search_result",
)
research_chain.add_step(
    tool_name = "extract_price",
    input_template = "{search_result}",
    output_key = "unit_price",
)
research_chain.add_step(
    tool_name = "my_calculator",
    input_template = "{unit_price} * {quantity}",
    output_key = "calculation_result",
)


# 4. 注册工具链
chain_manager = ToolChainManager(chain_registry)
chain_manager.registry_chain(research_chain)


# 5. 将整条工具链注册为 Agent 可以调用的工具
agent_registry = ToolRegistry()
chain_manager.register_chain_as_tool(
    chain_name="research_and_calculate",
    target_registry=agent_registry,
)


# 6. 创建支持工具链调用的 SimpleAgent
agent = MySimpleAgent(
    name="工具链助手",
    llm=llm,
    system_prompt=(
        "你是一个可以使用工具链的助手。"
        "需要查询真实价格并计算总价时，必须调用 research_and_calculate。"
        "工具参数格式为 input=搜索内容,quantity=购买数量。"
    ),
    tool_registry=agent_registry,
    enable_tool_calling=True,
)


# 7. 测试 Agent -> 工具链 -> Agent 的完整流程
print("\n=== 测试 SimpleAgent 调用工具链 ===")
response = agent.run(
    "请查询 NVIDIA RTX 5090 当前官方建议零售价（美元），"
    "并计算购买 2 件的总价"
)
print(f"Agent 最终响应: {response}")
