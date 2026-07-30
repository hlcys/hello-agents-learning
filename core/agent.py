""" Agent 基类"""

from abc import ABC, abstractmethod
from typing import Optional, Any
from .message import Message
from .llm import HelloAgentsLLM
from .config import Config, get_config


class Agent(ABC):
    """ Agent 基类 """

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None
    ):
        self.name = name
        self.llm  = llm
        self.system_prompt = system_prompt
        self.config   = (
            config
            if config is not None
            else get_config()
        )
        self._history = list[Message] = []

    @abstractmethod
    def run(self, input_text: str, **kwargs) -> str:
        """运行 Agent"""
        # 强制子类实现此方法
        pass

    def add_message(self, message: Message):
        """添加消息到历史记录"""
        self._history.append(message)


    def clear_history(self):
        self._history.clear()


    def get_history(self) -> list[Message]:
        return self._history


    def __str__(self):
        return f"Agent(name = {self.name}, provider = {self.llm.provider})"