"""
完整的Agentic RL训练流程
从数据准备到模型部署的端到端示例
"""

import argparse
import copy
import json
import os
from datetime import datetime
from pathlib import Path


DEFAULT_CONFIG = {
    "model": {"base_model": "Qwen/Qwen3-0.6B"},
    "data": {"max_samples": 1000},
    "sft": {
        "output_dir": "./models/sft_model", "num_epochs": 3,
        "batch_size": 1, "gradient_accumulation_steps": 32,
        "max_length": 512, "learning_rate": 5e-5,
    },
    "grpo": {
        "output_dir": "./models/grpo_model", "num_epochs": 3,
        "batch_size": 1, "gradient_accumulation_steps": 16,
        "num_generations": 2, "generation_batch_size": 2,
        "max_prompt_length": 384, "max_completion_length": 128,
        "learning_rate": 5e-6,
    },
    "eval": {"max_samples": 200, "max_prompt_length": 384, "max_new_tokens": 128},
    "monitoring": {
        "use_wandb": False, "use_tensorboard": True,
        "wandb_project": "agentic-rl-pipeline",
    },
    "results_path": "training_results.json",
}

class AgenticRLPipeline:
    """Agentic RL训练流水线"""

    def __init__(self, config_path=None, smoke_test=False):
        """
        初始化训练流水线
        
        Args:
            config_path: 配置文件路径
        """
        # 在导入训练库前配置分配器，减少变长样本造成的显存碎片。
        # 同时兼容直接实例化流水线，并保留用户设置的两种分配器环境变量。
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        if "PYTORCH_ALLOC_CONF" not in os.environ:
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        # hello-agents 0.2.5 的原始训练器没有实际应用 LoRA，必须使用本地实现。
        from local_rl_training_tool import LocalRLTrainingTool

        self.rl_tool = LocalRLTrainingTool()
        self.config_path = Path(config_path or Path(__file__).with_name("config.json")).resolve()
        self.config = self.load_config(self.config_path)
        if smoke_test:
            self.config["data"]["max_samples"] = 8
            for name in ("sft", "grpo"):
                self.config[name].update(
                    max_steps=2, batch_size=1, gradient_accumulation_steps=2,
                    output_dir=f"./models/smoke_{name}",
                )
            self.config["grpo"].update(num_generations=2, generation_batch_size=2)
            self.config["eval"]["max_samples"] = 2
            self.config["results_path"] = "training_results.smoke.json"
        for name in ("sft", "grpo"):
            self.config[name]["output_dir"] = str(
                (self.config_path.parent / self.config[name]["output_dir"]).resolve()
            )
        self.results = {"smoke_test": smoke_test, "config": copy.deepcopy(self.config)}

    def load_config(self, config_path):
        config = copy.deepcopy(DEFAULT_CONFIG)
        if config_path.exists():
            with config_path.open(encoding="utf-8") as f:
                supplied = json.load(f)
            for key, value in supplied.items():
                if isinstance(config.get(key), dict) and isinstance(value, dict):
                    config[key].update(value)
                else:
                    config[key] = value
        return config

    def log(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}", flush=True)

    def parse_result(self, result, stage):
        """每个阶段先检查工具状态，保留原始异常，不误报完成。"""
        data = json.loads(result)
        if not isinstance(data, dict):
            raise RuntimeError(f"{stage}返回了非字典结果: {data!r}")
        if data.get("status") != "success":
            if data.get("traceback"):
                print(data["traceback"], flush=True)
            raise RuntimeError(f"{stage}失败: {data.get('message', data)}")
        return data

    def stage1_prepare_data(self):
        """阶段一: 数据准备"""
        self.log("=" * 50)
        self.log("阶段一: 数据准备")
        self.log("=" * 50)

        result = self.rl_tool.run({
            "action": "load_dataset",
            "format": "sft", 
            "max_samples": self.config["data"]["max_samples"]
        })

        # 解析json结果
        dataset_info = self.parse_result(result, "数据准备")

        self.log(f"✓ 数据集加载完成")
        self.log(f"  - 样本数: {dataset_info['dataset_size']}")
        self.log(f"  - 格式: {dataset_info['format']}")
        self.log(f"  - 数据列: {', '.join(dataset_info['sample_keys'])}")

        self.results['data'] = dataset_info

        return dataset_info


    def stage2_sft_training(self):
        """阶段二: SFT训练"""
        self.log("\n" + "=" * 50)
        self.log("阶段二: SFT训练")
        self.log("=" * 50)

        sft_config = self.config["sft"]

        result = self.rl_tool.run({
            **sft_config,
            "action": "train",
            "algorithm": "sft",
            "model_name": self.config["model"]["base_model"],
            "output_dir": sft_config["output_dir"],
            "max_samples": self.config["data"]["max_samples"],
            "num_epochs": sft_config["num_epochs"],
            "batch_size": sft_config["batch_size"],
            "use_lora": True,
            # 训练监控配置
            "use_wandb": self.config.get("monitoring", {}).get("use_wandb", False),
            "use_tensorboard": self.config.get("monitoring", {}).get("use_tensorboard", True),
            "wandb_project": self.config.get("monitoring", {}).get("wandb_project", None),
        })

        result_data = self.parse_result(result, "SFT训练")

        self.log(f"✓ SFT训练完成")
        self.log(f"  - 模型路径: {result_data['output_dir']}")
        self.log(f"  - 状态: {result_data['status']}")
        self.results["sft_training"] = result_data

        return result_data["output_dir"]


    def stage3_sft_evaluation(self, model_path):
        "阶段3: SFT评估"
        self.log("\n" + "=" * 50)
        self.log("阶段三: SFT评估")
        self.log("=" * 50)

        result = self.rl_tool.run({
            **self.config["eval"],
            "action": "evaluate",
            "model_path": model_path,
            "max_samples": self.config["eval"]["max_samples"],
            "use_lora": True,
        })
        eval_data = self.parse_result(result, "SFT评估")

        self.log(f"✓ SFT评估完成")
        self.log(f"  - 准确率: {eval_data['accuracy']}")
        self.log(f"  - 平均奖励: {eval_data['average_reward']}")

        self.results["sft_evaluation"] = eval_data

        return eval_data


    def stage4_grpo_training(self, sft_model_path):
        """阶段四: GRPO 训练"""
        self.log("\n" + "=" * 50)
        self.log("阶段四: GRPO 训练")
        self.log("=" * 50)

        grpo_config = self.config["grpo"]
        result = self.rl_tool.run({
            **grpo_config,
            "action": "train",
            "algorithm": "grpo",
            "model_name": sft_model_path,
            "output_dir": grpo_config["output_dir"],
            "max_samples": self.config["data"]["max_samples"],
            "num_epochs": grpo_config["num_epochs"],
            "batch_size": grpo_config["batch_size"],
            "use_lora": True,
            # 训练监控配置
            "use_wandb": self.config.get("monitoring", {}).get("use_wandb", False),
            "use_tensorboard": self.config.get("monitoring", {}).get("use_tensorboard", True),
            "wandb_project": self.config.get("monitoring", {}).get("wandb_project", None),
        })

        result_data = self.parse_result(result, "GRPO训练")
        self.log(f"✓ GRPO训练完成")
        self.log(f"  - 模型路径: {result_data['output_dir']}")
        self.log(f"  - 状态: {result_data['status']}")

        self.results["grpo_training"] = result_data
        return result_data["output_dir"]


    def stage5_grpo_evaluation(self, model_path):
        """阶段五: GRPO评估"""
        self.log("\n" + "=" * 50)
        self.log("阶段五: GRPO评估")
        self.log("=" * 50)

        result = self.rl_tool.run({
            **self.config["eval"],
            "action": "evaluate",
            "model_path": model_path,
            "max_samples": self.config["eval"]["max_samples"],
            "use_lora": True,
        })
        eval_data = self.parse_result(result, "GRPO评估")

        self.log(f"✓ GRPO评估完成")
        self.log(f"  - 准确率: {eval_data['accuracy']}")
        self.log(f"  - 平均奖励: {eval_data['average_reward']}")

        self.results["grpo_evaluation"] = eval_data

        return eval_data


    def stage6_save_results(self):
        """阶段6: 保存结果"""
        self.log("\n" + "=" * 50)
        self.log("阶段6: 保存结果")
        self.log("=" * 50)
        
        # 保存训练结果
        results_path = self.config_path.parent / self.config["results_path"]
        with results_path.open('w', encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        self.log(f"✓ 结果已保存到: {results_path}")


    def run(self):
        "运行完整流程"
        try:
            runtime = self.rl_tool.check_environment()
            self.results["runtime"] = runtime
            self.log(
                f"GPU: {runtime['gpu_name']}，可用显存 "
                f"{runtime['free_memory_gib']:.2f}/{runtime['total_memory_gib']:.2f} GiB，"
                f"训练精度: {runtime['precision']}，使用 LoRA + 梯度检查点"
            )
            self.stage1_prepare_data()

            sft_model_path = self.stage2_sft_training()

            self.stage3_sft_evaluation(sft_model_path)

            grpo_model_path = self.stage4_grpo_training(sft_model_path)

            self.stage5_grpo_evaluation(grpo_model_path)
            self.stage6_save_results()
            self.log("\n" + "=" * 50)
            self.log("✓ 训练流程完成!")
            self.log("=" * 50)

        except Exception as e:
            self.log(f"\n训练失败!: {str(e)}")
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="6 GB GPU 上的 SFT → GRPO 训练和评估")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--smoke-test", action="store_true", help="8 条训练样本，每阶段 2 步，评估 2 题")
    parser.add_argument("--offline", action="store_true", help="仅使用本地 Hugging Face 缓存")
    args = parser.parse_args()

    # 必须在导入训练库之前设置离线模式。
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
    pipeline = AgenticRLPipeline(args.config, smoke_test=args.smoke_test)
    pipeline.run()
