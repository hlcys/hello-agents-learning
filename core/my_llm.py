"""
- Hello-Agents LLM

- 1. 多提供商支持：实现对 OpenAI、ModelScope、智谱 AI 等多种主流 LLM 服务商的无缝切换，避免框架与特定供应商绑定。
- 2. 本地模型集成：引入 VLLM 和 Ollama 这两种高性能本地部署方案，对 Hugging Face Transformers 方案的生产级补充，满足数据隐私和成本控制的需求。
- 3. 自动检测机制：建立一套自动识别机制，使框架能根据环境信息智能推断所使用的 LLM 服务类型，简化用户的配置过程。
"""

import os
from typing import Optional
from openai import OpenAI
from hello_agents import HelloAgentsLLM


"""
- 支持多提供商:
- 通过引入 provider
"""

class MyLLM(HelloAgentsLLM):
    """
    - 自定义 LLM 客户端
    """
    def __init__(
        self, 
        model:    Optional[str] = None,
        api_key:  Optional[str] = None,
        base_url: Optional[str] = None,
        provider: Optional[str] = "auto",
        **kwargs
    ):
        # 检查 provider是否是我们要处理的 model scope
        if provider == "modelspace":
            print("正在使用自定义的 ModelScope Provider")
            self.provider = "modelspace"

            # 解析 Modelscope 凭证
            self.api_key  = api_key  or os.getenv("MODELSCOPE_API_KEY")
            self.base_url = base_url or "https://api-inference.modelscope.cn/v1/"
            
            # 验证凭证是否存在
            if not self.api_key:
                raise ValueError("MODELSCOPE not set")
            
            self.model = model or os.getenv("LLM_MODEL_ID") or "Qwen/Qwen2.5-VL-72B-Instruct"
            self.temperature = kwargs.get('temperature', 0.7)
            self.max_tokens  = kwargs.get('max_tokens')
            self.timeout = kwargs.get('timeout', 60)

            self.client = OpenAI(
                api_key  = self.api_key,
                base_url = self.base_url,
                timeout  = self.timeout
            )
        else:
            # 如果不是 modelscope, 则完全使用父类逻辑
            super.__init__(
                model = model, 
                api_key = api_key,
                base_url = base_url,
                provider = provider,
                **kwargs
            )