# simple_agent

from typing import Optional, Iterator
from hello_agents import SimpleAgent, HelloAgentsLLM, Config, Message, ToolRegistry
import re

class MySimpleAgent(SimpleAgent):
    """
        - 简单对话 Agent
    """

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        sysytem_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        tool_registry: Optional['ToolRegistry'] = None,
        enable_tool_calling: bool = True      
    ):
        super().__init__(name, llm, sysytem_prompt, config)
        self.tool_registry = tool_registry
        self.enable_tool_calling = enable_tool_calling and tool_registry is not None
        print(f"✅ {name} 初始化完成，工具调用: {'启用' if self.enable_tool_calling else '禁用'}")


    def run(
        self,
        input_text: str,
        max_tool_iterations: int = 3, **kwargs
    ):
        print(f" {self.name} 正在处理: {input_text}")

        messages = []

        # 添加系统消息
        enhanced_system_prompt = self._get_enhanced_system_prompt()
        messages.append({"role": "system", "content": enhanced_system_prompt})

        for msg in self._history:
            messages.append({
                "role": msg.role,
                "content": msg.content
            })

        messages.append({
            "role": "user",
            "content": input_text
        })

        # 如果没有启用工具调用
        if not self.enable_tool_calling:
            response = self.llm.invoke(messages, **kwargs)
            self.add_message(Message(input_text, "user"))
            self.add_message(Message(response, "assistant"))
            print(f"✅ {self.name} 响应完成")
            return response       

        # 支持多轮工具调用
        return self._run_with_tools(messages, input_text, max_tool_iterations, **kwargs)


    def _get_enhanced_system_prompt(self) -> str:
        """ 构建增强的系统提示词, 包含工具信息"""
        base_prompt = self.system_prompt or "你是一个有用的AI助手"

        if not self.enable_tool_calling or not self.tool_registry:
            return base_prompt

        # 获取工具描述
        tools_description = self.tool_registry.get_tools_description()
        if not tools_description or tools_description == "暂无可用工具":
            return base_prompt

        tools_section = "\n\n## 可用工具\n"
        tools_section += "你可以使用以下工具来帮助回答问题:\n"
        tools_section += tools_description + "\n"

        tools_section += "\n## 工具调用格式\n"
        tools_section += "当需要使用工具时，请使用以下格式:\n"
        tools_section += "`[TOOL_CALL:{tool_name}:{parameters}]`\n"
        tools_section += "例如:`[TOOL_CALL:search:Python编程]` 或 `[TOOL_CALL:memory:recall=用户信息]`\n\n"
        tools_section += "工具调用结果会自动插入到对话中，然后你可以基于结果继续回答。\n"

        return base_prompt + tools_section


    def _run_with_tools(self, messages: list, input_text: str, max_tool_iterations: int, **kwargs):
        """ 支持工具调用的运行逻辑 """
        current_iteration = 0
        final_response = ""

        while current_iteration < max_tool_iterations:
            response = self.llm.invoke(messages, **kwargs)

            # 检查是否有工具调用
            tool_calls = self._parse_tool_calls(response)

            if tool_calls:
                print(f"🔧 检测到 {len(tool_calls)} 个工具调用")

                # 执行工具调用，收集结果
                tool_results = []
                clean_response = response

                for call in tool_calls:
                    result = self._execute_tool_call(call['tool_name'], call['parameters'])
                    tool_results.append(result)

                    clean_response = clean_response.replace(call['original'], "")

                messages.append({
                    "role": "assistant",
                    "content": clean_response
                })

                tool_results_text = "\n\n".join(tool_results)
                messages.append({
                    "role": "user",
                    "content": f"工具执行结果:\n{tool_results_text}\n\n请基于这些结果给出完整的回答。"
                })

                current_iteration += 1
                continue

            final_response = response
            break

        # 超过最大次数，获取最后一次回答
        if current_iteration >= max_tool_iterations and not final_response:
            final_response = self.llm.invoke(messages, **kwargs)

        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_response, "assistant"))
        print(f"✅ {self.name} 响应完成")

        return final_response


    def _parse_tool_calls(self, text: str) -> list:
        """解析文本中的工具调用"""
        pattern = r'\[TOOL_CALL:([^:]+):([^\]]+)\]'
        matches = re.findall(pattern, text)

        tool_calls = []
        for tool_name, parameters in matches:
            tool_calls.append({
                'tool_name': tool_name.strip(),
                'parameters': parameters.strip(),
                'original': f'[TOOL_CALL:{tool_name}:{parameters}]'
            })

        return tool_calls


    def _execute_tool_call(self, tool_name: str, parameters: str) -> str:
        """ 执行工具调用 """
        if not self.tool_registry:
            return f"❌ 错误:未配置工具注册表"

        # 结果由 llm 生成，具有不确定性
        try:
            if tool_name == "calculator":
                # 计算器工具
                result = self.tool_registry.execute_tool(tool_name, parameters)

            else:
                param_dict = self._parse_tool_parameters(tool_name, parameters)
                tool = self.tool_registry.get_tool(tool_name)
                if not tool:
                    return f"❌ 错误:未找到工具 '{tool_name}'"
                result = tool.run(param_dict)

            return f"🔧 工具 {tool_name} 执行结果:\n{result}"

        except Exception as e:
            return f"工具调用失败 ❌"


    def _parse_tool_parameters(self, tool_name: str, parameters: str) -> dict:
        """ 智能解析工具参数 """
        paramdict = {}

        if '=' in parameters:
            if ',' in parameters:
                pairs = parameters.split(',')
                for pair in pairs:
                    if '=' in pair:
                        key, value = pair.split('=', 1)   # "url=https://example.com?a=1" --> ["url", "https://example.com?a=1"]
                        paramdict[key.strip()] = value.strip()
            else:
                key, value = parameters.split('=', 1)
                paramdict[key.strip()] = value.strip()

        else:
            # 直接传入参数
            if tool_name == 'search':
                paramdict = {'search': parameters}
            elif tool_name == 'memory':
                paramdict = {'action': 'search', 'query': parameters}
            else:
                paramdict = {'input': parameters}

        return paramdict


    def stream_run(self, input_text: str, **kwargs) -> Iterator[str]:
        """
        流式运行方法
        """
        print(f" {self.name} 开始流式处理: {input_text}")

        messages = []

        if self.system_prompt:
            messages.append({
                "role": "system",
                "content": self.system_prompt
            })

        for msg in self.history:
            messages.append({
                "role": msg.role,
                "content": msg.content
            })

        messages.append({
            "role": "user",
            "content": input_text
        })

        # 流式调用 llm (服务器边生成边返回)
        full_response = ""
        print(" 实时响应:  ", end = "")

        for chunk in self.llm.stream_invoke(messages, **kwargs):
            full_response += chunk
            print(chunk, end = "", flush = True)
            yield chunk # yield 不中断，实时返回

        print() 

        self.add_messages(Message(input_text, "user"))
        self.add_messages(Message(full_response, "assistant"))
        print(f" {self.name} 流式调用完成")


    def add_tool(self, tool) -> None:
        """添加工具到 Agent (库方法)"""
        if not self.tool_registry:
            from hello_agents import ToolRegistry
            self.tool_registry = ToolRegistry()
            self.enable_tool_calling = True

        self.tool_registry.register_tool(tool)
        print(f" 工具 '{tool_name}' 已添加")


    def has_tool(self) -> bool:
        """检查是否有可用工具"""
        return self.enable_tool_calling and self.tool_registry is not None


    def remove_tool(self, tool_name: str) -> bool:
        """移除工具 (库方法)"""
        if self.tool_registry:
            self.tool_registry.unregister(tool_name)
            return True
        return False


    def list_tools(self) -> list:
        """列出可用工具"""
        if self.tool_registry:
            return self.tool_registry.list_tools()
        return []
    