"""为本示例补齐 hello-agents 0.2.5 未实现的 LoRA 和显存控制。

保留 RLTrainingTool 的 JSON 接口，不修改虚拟环境中的第三方源码。
"""

import gc
import json
from pathlib import Path

from hello_agents.tools import RLTrainingTool


class LocalRLTrainingTool(RLTrainingTool):
    @staticmethod
    def _precision():
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("未检测到可用 CUDA GPU，请检查驱动及运行环境的 GPU 访问权限")
        bf16 = torch.cuda.is_bf16_supported()
        return torch.bfloat16 if bf16 else torch.float16, bf16

    def check_environment(self):
        """加载数据和模型前检查 CUDA，避免显存统计 API 掩盖环境错误。"""
        import torch

        _, bf16 = self._precision()
        free_memory, total_memory = torch.cuda.mem_get_info()
        return {
            "gpu_name": torch.cuda.get_device_name(),
            "free_memory_gib": round(free_memory / 1024**3, 3),
            "total_memory_gib": round(total_memory / 1024**3, 3),
            "precision": "bf16" if bf16 else "fp16",
        }

    @staticmethod
    def _release_memory():
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_model(self, model_path, trainable=False, parameters=None):
        import torch
        from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        parameters = parameters or {}
        dtype, _ = self._precision()
        is_adapter = (Path(model_path) / "adapter_config.json").is_file()
        base_model = (
            PeftConfig.from_pretrained(model_path).base_model_name_or_path
            if is_adapter else model_path
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        # 显式以半精度加载到单张 GPU；device_map="auto" 是推理分发策略。
        model = AutoModelForCausalLM.from_pretrained(
            base_model, dtype=dtype, attn_implementation="sdpa",
        ).to(torch.device("cuda", torch.cuda.current_device()))
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.use_cache = not trainable
        if is_adapter:
            model = PeftModel.from_pretrained(model, model_path, is_trainable=trainable)
            print(f"已加载 LoRA 权重: {model_path}", flush=True)
        elif trainable:
            model = get_peft_model(model, LoraConfig(
                task_type="CAUSAL_LM",
                r=parameters.get("lora_r", 16),
                lora_alpha=parameters.get("lora_alpha", 32),
                lora_dropout=0.05,
                target_modules=["q_proj", "v_proj"],
            ))
        if trainable:
            model.print_trainable_parameters()
        else:
            model.eval()
        return model, tokenizer

    @staticmethod
    def _rl_dataset(parameters, tokenizer, split):
        from hello_agents.rl.datasets import GSM8KDataset

        # 与 SFT 的 Question / Let's solve 提示保持一致，避免 Qwen3 在评估时
        # 被另一套 chat template 切换到 thinking 模式，耗尽回答长度。
        dataset = GSM8KDataset(
            split=split, max_samples=parameters.get("max_samples", 1000),
            format_type="rl",
        ).get_dataset()
        max_prompt_length = parameters.get("max_prompt_length", 384)

        def prepare(example):
            return {"prompt": example["prompt"] + "\n"}

        dataset = dataset.map(prepare)
        original_size = len(dataset)
        # 当前 TRL 的 GRPOConfig 没有 max_prompt_length，不能直接传这个参数。
        # 过滤过长题目，避免截断题意或在 rollout 时超过显存预算。
        dataset = dataset.filter(
            lambda example: len(tokenizer.encode(example["prompt"])) <= max_prompt_length
        )
        if len(dataset) != original_size:
            print(f"跳过 {original_size - len(dataset)} 条超出 {max_prompt_length} tokens 的题目")
        if not len(dataset):
            raise ValueError("没有可用样本，请增加 max_samples 或 max_prompt_length")
        return dataset

    def _handle_train(self, parameters):
        import torch
        from hello_agents.rl import create_accuracy_reward, create_sft_dataset
        from transformers import set_seed
        from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer

        algorithm = parameters.get("algorithm", "sft").lower()
        if algorithm not in {"sft", "grpo"}:
            raise ValueError(f"不支持的算法: {algorithm}")
        if not parameters.get("use_lora", True):
            raise ValueError("本示例针对 6 GB 显卡配置，请保持 use_lora=True")
        model = tokenizer = trainer = None
        self._release_memory()
        try:
            _, bf16 = self._precision()
            torch.cuda.reset_peak_memory_stats()
            set_seed(parameters.get("seed", 42))
            model, tokenizer = self._load_model(
                parameters["model_name"], trainable=True, parameters=parameters,
            )
            report_to = []
            if parameters.get("use_tensorboard", True):
                report_to.append("tensorboard")
            if parameters.get("use_wandb", False):
                import os
                if parameters.get("wandb_project"):
                    os.environ["WANDB_PROJECT"] = parameters["wandb_project"]
                report_to.append("wandb")
            common_args = dict(
                output_dir=parameters["output_dir"],
                num_train_epochs=parameters.get("num_epochs", 3),
                per_device_train_batch_size=parameters.get("batch_size", 1),
                gradient_accumulation_steps=parameters.get("gradient_accumulation_steps", 4),
                learning_rate=parameters.get("learning_rate", 5e-5),
                max_steps=parameters.get("max_steps", -1),
                warmup_steps=0,
                bf16=bf16, fp16=not bf16,
                gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
                optim="adamw_torch",
                logging_steps=1,
                save_strategy="no",  # 每阶段结束后保存 adapter 和 tokenizer
                report_to=report_to,
                seed=parameters.get("seed", 42),
                dataloader_num_workers=0,
            )
            if algorithm == "sft":
                dataset = create_sft_dataset(max_samples=parameters.get("max_samples", 1000))
                args = SFTConfig(
                    **common_args, max_length=parameters.get("max_length", 512),
                    packing=False,
                )
                trainer = SFTTrainer(
                    model=model, args=args, train_dataset=dataset, processing_class=tokenizer,
                )
            else:
                dataset = self._rl_dataset(parameters, tokenizer, "train")
                args = GRPOConfig(
                    **common_args,
                    num_generations=parameters.get("num_generations", 2),
                    generation_batch_size=parameters.get("generation_batch_size", 2),
                    max_completion_length=parameters.get("max_completion_length", 128),
                    beta=parameters.get("beta", 0.0),
                    temperature=0.7,
                    use_vllm=False,
                )
                # 直接继续训练 SFT adapter，不能重新创建一个随机 LoRA。
                trainer = GRPOTrainer(
                    model=model, args=args, train_dataset=dataset,
                    processing_class=tokenizer, reward_funcs=create_accuracy_reward(),
                )
            print(
                f"{algorithm.upper()}: batch={args.per_device_train_batch_size}, "
                f"gradient_accumulation={args.gradient_accumulation_steps}, "
                f"dtype={'bf16' if bf16 else 'fp16'}", flush=True,
            )
            outcome = trainer.train()
            trainer.save_model(parameters["output_dir"])
            tokenizer.save_pretrained(parameters["output_dir"])
            result = {
                "status": "success", "algorithm": algorithm.upper(),
                "model": parameters["model_name"], "output_dir": parameters["output_dir"],
                "dataset_size": len(dataset), "global_step": outcome.global_step,
                "metrics": outcome.metrics,
                "peak_gpu_memory_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
            }
            print(f"峰值 GPU 张量显存: {result['peak_gpu_memory_gib']} GiB", flush=True)
            return json.dumps(result, ensure_ascii=False, indent=2)
        finally:
            del trainer, model, tokenizer
            self._release_memory()

    def _handle_evaluate(self, parameters):
        import torch
        from hello_agents.rl import create_accuracy_reward
        from tqdm import tqdm

        model = tokenizer = None
        self._release_memory()
        try:
            model_path = parameters["model_path"]
            model, tokenizer = self._load_model(model_path)
            dataset = self._rl_dataset(parameters, tokenizer, "test")
            completions = []
            for sample in tqdm(dataset, desc="评估", unit="题"):
                inputs = tokenizer(sample["prompt"], return_tensors="pt").to(model.device)
                with torch.inference_mode():
                    outputs = model.generate(
                        **inputs, max_new_tokens=parameters.get("max_new_tokens", 128),
                        do_sample=False, pad_token_id=tokenizer.pad_token_id,
                        use_cache=True,
                    )
                completions.append(tokenizer.decode(
                    outputs[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
                ))
                del inputs, outputs
            rewards = create_accuracy_reward()(completions, ground_truth=dataset["ground_truth"])
            accuracy = sum(rewards) / len(rewards)
            return json.dumps({
                "status": "success", "model_path": model_path,
                "num_samples": len(dataset), "accuracy": f"{accuracy:.2%}",
                "average_reward": f"{accuracy:.4f}", "device": str(model.device),
            }, ensure_ascii=False, indent=2)
        finally:
            del model, tokenizer
            self._release_memory()
