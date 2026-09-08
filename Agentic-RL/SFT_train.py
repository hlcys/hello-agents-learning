## 完整的SFt训练流程

from hello_agents.tools import RLTrainingTool

rl_tool = RLTrainingTool()

result = rl_tool.run({
    "action": "train",
    "algorithm": "sft",

    # 模型配置
    "model_name": "Qwen/Qwen3-0.6B",
    "output_dir": "./model/sft_full",

    # 数据配置, 选择全部参数
    "max_samples": 1000,

    # 训练参数
    "num_epochs": 3,
    "batch_size": 8,
    "learning_rate": 5e-5,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,

    # LoRA配置
    "use_lora": True,
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_target_modules": ["q_proj", "v_proj"],

    # 其他配置:
    "save_steps": 500,
    "logging_steps": 100,
    "eval_steps": 500    
})

print(f"✅ 训练完成! 模型保存在: {result['model_pth']}")