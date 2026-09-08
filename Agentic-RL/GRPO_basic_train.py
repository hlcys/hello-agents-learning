
from hello_agents.tools import RLTrainingTool

rl_tool = RLTrainingTool()

result = rl_tool.run({
    # 训练配置
    "action" : "train",
    "algorithm": "grpo",

    # 模型配置
    "model_name": "./models/sft_full",
    "output_dir": "./models/grpo_model",

    # 数据配置
    "max_samples": None,

    # 训练参数
    "num_epochs": 3,
    "batch_size": 4,
    "learning_rate": 1e-5, 
    "warmup_ratio": 0.1,

    # GRPO 特定参数
    "num_generation": 4, # 每个问题生成四个答案
    "kl_coef": 0.05,
    "max_new_tokens": 512,
    "temperature": 0.8,
    "clip_range": 0.2,

    # LoRA 配置
    "use_lora": True,
    "lora_rank": 16,
    "lora_alpha": 32,

    # 奖励函数配置
    "reward_type": "combined", # 使用准确率奖励
    "reward_config":{
        "components":[
            {"type": "accuacy", "weight": 1.0},
            {"type": "length_penalty", "weight": 0.5, "target_length": 200},
            {"type": "step", "weight": 0.3, "step_bonus": 0.1}
        ]
    },

    #其他配置
    "save_steps": 500,
    "logging_steps": 100,
})  

import json

result = json.loads(result)

if result.get("status") != "success": 
    print("训练失败：", result.get("message"))
    print(result.get("traceback", ""))
    raise SystemExit(1)