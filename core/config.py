""" 配置管理 """
# - 单例模式: 让配置具备一致性, 避免差异化

import os
from typing import Optional, Dict, Any
from pydantic import BaseModel

class Config(BaseModel):
    """Hello Agents 配置类"""

    model_config = ConfigDict(frozen=True)

    # LLM 配置
    default_model: str = "gpt-3.5-turbo"
    default_provider: str = "openai"
    temperature: float = 0.7
    max_tokens:  Optional[int] = None

    # 系统配置
    debug: bool = False
    log_level: str = "INFO"

    # 其他配置
    max_history_length: int = 100

    @classmethod
    def from_env(cls) -> "Config":
        """ 从环境变量中创建配置 """
        return cls(
            debug = os.getenv("DEBUG", "false").lower() == "true",
            log_level   = os.getenv("LOG_LEVEL", "INFO"),
            temperature = float(os.getenv("TEMPERATURE", "0.7")),
            max_tokens  = int(os.getenv("MAX_TOKENS")) if os.getenv("MAX_TOKENS") else None,
        )


    @lru_cache(maxszie = 1)
    def get_config() -> Config:
        """ 获取当前进程唯一的全局配置实例。"""
        return Config.from_env()
    