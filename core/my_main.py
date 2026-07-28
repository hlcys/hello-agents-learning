from dotenv import load_dotenv
from my_llm import MyLLM

load_dotenv()

llm = MyLLM(provider="modelscope")


messages = [
    {"role": "user", "content": "你好, 请介绍下自己"}
]

reponse_stream = llm.think(messages)

print("Modelscope Response:")
for chunk in response_stream:
    pass

