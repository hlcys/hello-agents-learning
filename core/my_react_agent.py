# my_react_agent

MY_REACT_PROMPT = """你是一个具备推理和行动能力的AI助手。你可以通过思考分析问题,然后调用合适的工具来获取信息,最终给出准确的答案。

## 可用工具
{tools}

## 工作流程
请严格按照以下格式进行回应,每次只能执行一个步骤:

Thought: 分析当前问题,思考需要什么信息或采取什么行动。
Action: 选择一个行动,格式必须是以下之一:
- `{{tool_name}}[{{tool_input}}]` - 调用指定工具
- `Finish[最终答案]` - 当你有足够信息给出最终答案时

## 重要提醒
1. 每次回应必须包含Thought和Action两部分
2. 工具调用的格式必须严格遵循:工具名[参数]
3. 只有当你确信有足够信息回答问题时,才使用Finish
4. 如果工具返回的信息不够,继续使用其他工具或相同工具的不同参数

## 当前任务
**Question:** {question}

## 执行历史
{history}

现在开始你的推理和行动:
"""

import re
from typing import Optional, List
from hello_agents import ReActAgent, HelloAgentsLLM, Config, Message, ToolRegistry


class MyReactAgent(ReActAgent):
    """
        重写的 ReactAgent
    """

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        tool_registry: ToolRegistry,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        max_steps: int = 5,
        custom_prompt: Optional[str] = None,
        enable_tool_calling: bool = True,
        quality_threshold: float = 8.0,
        max_reflections: int = 2,
    ):
        if not 0 <= quality_threshold <= 10:
            raise ValueError("quality_threshold 必须在 0 到 10 之间")
        if max_reflections < 0:
            raise ValueError("max_reflections 不能小于 0")

        super().__init__(
            name,
            llm,
            tool_registry,
            system_prompt,
            config,
            max_steps,
            custom_prompt,
        )
        self.tool_registry = tool_registry
        self.max_steps = max_steps
        self.current_history: List[str] = []
        self.prompt_template = (
            MY_REACT_PROMPT if custom_prompt is None else custom_prompt
        )

        # 工具调用配置
        self.enable_tool_calling = enable_tool_calling

        # 质量评分配置
        self.quality_threshold = float(quality_threshold)
        self.max_reflections = max_reflections
        self.quality_scores: List[float] = []
        self.reflections: List[str] = []

        print(
            f"✅ {name}初始化完成, 最大步数: {max_steps}, "
            f"质量阈值: {self.quality_threshold}"
        )

    def run(self, input_text: str, **kwargs) -> str:
        """运行React Agent"""
        self.current_history = []
        self.quality_scores = []
        self.reflections = []
        current_step = 0

        print(f"\n🤖 {self.name} 开始处理问题: {input_text}")

        while current_step < self.max_steps:
            current_step += 1
            print(f"\n--- 第 {current_step} 步 ---")

            # 1. 构建提示词
            if self.enable_tool_calling:
                tools_desc = self.tool_registry.get_tools_description()
            else:
                tools_desc = "工具调用已禁用，请直接使用 Finish 返回答案。"

            history_str = "\n".join(self.current_history)
            prompt = self.prompt_template.format(
                tools=tools_desc,
                question=input_text,
                history=history_str,
            )

            # 2. 调用 llm
            messages = [{
                "role": "user",
                "content": prompt,
            }]
            response_text = self.llm.invoke(messages, **kwargs)

            # 3. 解析输出
            thought, action = self._parse_output(response_text)

            # 4. 检查完成条件
            if action and action.startswith("Finish"):
                final_answer = self._parse_action_input(action)

                # 对候选答案进行反思、评分，低于阈值时才继续优化
                final_answer = self._quality_control(
                    input_text,
                    final_answer,
                    **kwargs,
                )

                self.add_message(Message(input_text, "user"))
                self.add_message(Message(final_answer, "assistant"))
                return final_answer

            # 工具调用
            if action:
                if not self.enable_tool_calling:
                    self.current_history.append(
                        "Observation: 工具调用已禁用，请直接给出最终答案。"
                    )
                    continue

                tool_name, tool_input = self._parse_action(action)
                if not tool_name or tool_input is None:
                    self.current_history.append(
                        "Observation: 无效的Action格式, 请检查。"
                    )
                    continue

                observation = self.tool_registry.execute_tool(
                    tool_name,
                    tool_input,
                )
                self.current_history.append(f"Action: {action}")
                self.current_history.append(f"Observation: {observation}")

        final_answer = "抱歉,我无法在限定步数内完成这个任务。"
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_answer, "assistant"))
        return final_answer

    def _quality_control(self, question: str, answer: str) -> str:
        """反思并评分答案，只有分数低于阈值时才继续优化。"""
        current_answer = answer

        for reflection_round in range(self.max_reflections):
            print(f"\n--- 第 {reflection_round + 1} 轮质量检查 ---")

            # 1. 反思当前答案
            reflection_prompt = f"""请反思下面的答案，指出其中在正确性、完整性、
            与工具结果的一致性和表达清晰度方面存在的问题。

            用户问题:
            {question}

            工具执行历史:
            {chr(10).join(self.current_history) or "暂无工具执行历史"}

            当前答案:
            {current_answer}

            只给出反思意见，不要重写答案。"""

            reflection = self.llm.invoke(
                [{"role": "user", "content": reflection_prompt}],
                **kwargs,
            )
            reflection = reflection or "未返回有效的反思意见。"
            self.reflections.append(reflection)
            print(f"🪞 反思: {reflection}")

            # 2. 在反思后对当前答案评分
            score_prompt = f"""请根据用户问题、当前答案和反思意见，
            对当前答案进行0到10分的质量评分。

            用户问题:
            {question}

            当前答案:
            {current_answer}

            反思意见:
            {reflection}

            只输出 SCORE=分数，例如 SCORE=8.5, 不要输出其他内容。"""

            score_kwargs = dict(kwargs)
            score_kwargs["temperature"] = 0.0
            score_response = self.llm.invoke(
                [{"role": "user", "content": score_prompt}],
                **score_kwargs,
            )
            score = self._parse_quality_score(score_response)
            self.quality_scores.append(score)

            print(
                f"📊 当前答案评分: {score}/10, "
                f"质量阈值: {self.quality_threshold}"
            )

            # 3. 达到阈值时提前终止
            if score >= self.quality_threshold:
                print("✅ 当前答案已达到质量阈值，提前终止优化。")
                return current_answer

            # 4. 低于阈值时继续优化
            print("🔄 当前答案低于质量阈值，继续优化。")
            optimize_prompt = f"""请根据反思意见优化当前答案。

用户问题:
{question}

工具执行历史:
{chr(10).join(self.current_history) or "暂无工具执行历史"}

当前答案:
{current_answer}

反思意见:
{reflection}

只输出优化后的最终答案，不要输出评分或修改说明。"""

            optimized_answer = self.llm.invoke(
                [{"role": "user", "content": optimize_prompt}],
                **kwargs,
            )
            if optimized_answer:
                current_answer = optimized_answer.strip()

        return current_answer

    @staticmethod
    def _parse_quality_score(score_text: Optional[str]) -> float:
        """解析 SCORE=8.5 格式的评分, 解析失败时按0分处理。"""
        if not score_text:
            return 0.0

        match = re.search(
            r"(?:SCORE|评分|分数)\s*[:=：]\s*(-?\d+(?:\.\d+)?)",
            score_text,
            re.IGNORECASE,
        )
        if not match:
            return 0.0

        score = float(match.group(1))
        return score if 0 <= score <= 10 else 0.0
