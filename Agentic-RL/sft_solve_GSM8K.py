# 使用没有经过SFT的模型解决GSM8K问题

from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载预训练模型
model_name = "Qwen/Qwen3-0.6B"
tokenizer  = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


question = """Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"""
prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(**inputs, max_new_tokens = 200)
response = tokenizer.decode(outputs[0], skip_special_tokens = False)

print("预训练模型的回答: ")
print(response)
'''
预训练模型的回答: 
<|im_start|>user
Natalia sold clips to 48 of her friends in April, and then she sold half as many clipsin May. How many clips did Natalia sell altogether in April and May?<|im_end|>
<|im_start|>assistant
<think>
Okay, let me try to figure out how many clips Natalia sold in April and May. So, the problem says she sold 48 clips to her friends in April, and then she sold half as many in May. Hmm, let me break this down step by step.
First, in April, she sold 48 clips. That seems straightforward. Now, in May, she sold half as many. So, half of 48? Wait, half of 48 would be... let me calculate that. 48 divided by 2 is 24. So, in May, she sold 24 clips.
Now, to find the total number of clips sold in both months, I just need to add the April sales and the May sales together. So, 48 (April) plus 24 (May) equals... let me add those numbers. 48 plus 24... that equals 72. So,


SFT 模型的回答:
<|im_start|>user
Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?<|im_end|>
<|im_start|>assistant
Let me solve this step by step.

Step 1: Calculate clips sold in May
Natalia sold half as many clips in May as in April.
Clips in May = 48 / 2 = 24

Step 2: Calculate total clips
Total = April + May = 48 + 24 = 72

Final Answer: 72<|im_end|>
'''


