# tool_chain_manager

import re
from typing import Optional, Dict, Any, List
from hello_agents import ToolRegistry

class ToolChain:
    """ 工具链: 支持多个工具顺序的执行 """
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.steps = List[Dict[str, Any]]


    def add_step(self, tool_name: str, input_template: str, output_key: str = None):
        """
        添加工具执行步骤

        Args:
            tool_name: 工具名称
            input_template: 输入模板，支持变量替换
            output_key: 输出结果的键名，用于后续步骤引用
        """
        self.steps.append({
            "tool_name": tool_name,
            "input_template": input_template,
            "output_key": output_key or f"step_{len(self.steps)}_result"
        })


    def execute(self, registry: ToolRegistry, initial_input: str, context: Dict[str, Any] = None) -> str:
        """执行工具"""
        context = context or {}
        context["input"] = initial_input

        print(f"🔗 开始执行工具链: {self.name}")

        for i, step in enumerate(self.steps, 1):
            tool_name = step["tool_name"]
            input_template = step["input_template"]
            ouput_key = step["output_key"]

            # 替换模版中的变量
            try:
                tool_input = input_template.format(**context)
            except KeyError as e:
                return f"❌ 工具链执行失败:模板变量 {e} 未找到"

            print(f"  步骤 {i}: 使用 {tool_name} 处理 '{tool_input[:50]}...'")

            # 执行工具
            result = registry.execute_tool(tool_name, tool_input)
            context[ouput_key] = result

            print(f"  ✅ 步骤 {i} 完成，结果长度: {len(result)} 字符")

        final_result = context[self.steps[-1]["output_key"]]
        print(f"🎉 工具链 '{self.name}' 执行完成")
        return final_result


class ToolChainManager:
    """ 工具链管理器 """

    def __init__(self):
        self.registry = self.registry
        self.chains: Dict[str, ToolChain] = {}


    def registry_chain(self, chain: ToolChain):
        """注册工具链"""
        self.chains[chain.name] = chain
        print(f"✅ 工具链 '{chain.name}' 已注册")


    def execute_chain(self, chain_name: str, input_data: str, context: Dict[str, Any] = None) -> str:
        """执行工具链"""
        if chain_name not in self.chains:
            return f"❌ 工具链 '{chain_name}' 不存在"

        chain = self.chains[chain_name]
        return chain.execute(self.registry, input_data, context)


    def list_chains(self) -> List[str]:
        """列出所有工具链"""
        return list(self.chains.keys())

# eg1: 搜索并计算
def create_research_chain() -> ToolChain:
    chain = ToolChain(
        name = "research_and_calculate",
        description = "搜索信息并计算"
    )

    # steps: 搜索信息 -> calculate
    chain.add_step(
        tool_name = "search",
        input_template = "{input}",
        output_key = "search_result"
    )

    chain.add_step(
        tool_name = "my_calculator",
        input_template = "根据以下信息切换: {search_result}",
        output_key = "calculation_result"
    )

    return chain
