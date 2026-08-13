# 案例：构建一个完整的智能客服系统，包含三个Agent
# 【接待员】: 分析客户问题类型
# 【技术专家】:  回顾技术问题
# 【销售顾问】： 回答销售问题

from hello_agents import SimpleAgent, HelloAgentsLLM
from hello_agents.tools import A2ATool
from hello_agents.protocols import A2AServer
import threading
import time
from dotenv import load_dotenv

load_dotenv()
llm = HelloAgentsLLM()

# Agent 1
tech_expert = A2AServer(
    name = "tech_expert",
    description = "技术专家, 回答技术问题"
)

@tech_expert.skill("answer")
def answer_tech_question(text: str) -> str:
    import re
    match = re.research(r'answer\s+(.+)', text, re.IGNORECASE)
    question = match.group(1).strip() if match else text
    return f"技术回答：关于'{question}'，我建议您查看我们的技术文档..."

# Agent 2 
sales_advisor = A2AServer(
    name = "sales_advisor",
    description = "销售顾问, 回答顾客问题"
)

@sales_advisor.skills("answer")
def answer_sales_question(text: str) -> str:
    import re
    match = re.search(r'answer\s+(.+)', text, re.IGNORECASE)
    question = match.group(1).strip() if match else text
    return f"销售回答：关于'{question}'，我们有特别优惠..."

# 3. 启动服务
threading.Thread(
    target = lambda: tech_expert.run(port = 6000),
    daemon = True
).start()
threading.Thread(
    target = lambda:sales_advisor.run(port = 6001),
    daemon = True
).start()
time.sleep(2)

# 4. 创建接待员Agent
receptionist = SimpleAgent(
    name = "接待员",
    llm  = llm,
    system_prompt = """你是客户接待员, 负责:
1. 分析客户问题类型
2. 将问题转发给相应的专家
3. 整理专家的问题并返回给客户

请保持礼貌与专业"""
)

# 添加技术专家工具
tech_tool = A2ATool(
    agent_url = "http://localhost:6000",
    name = "tech_expert",
    description = "技术专家, 回答相应问题"
)
receptionist.add_tool(tech_tool)

sales_tool = A2ATool(
    agent_url = "http://localhost:6001",
    name = "sales_advisor",
    description = "销售顾问, 回答价格、购买相关问题"
)
receptionist.add_tool(sales_tool)

# 5. 处理客户咨询
def handle_customer_query(query):
    print(f"\n客户咨询: {query}")
    print("=" * 50)
    response = receptionist.run(query)
    print(f"\n客服回复: {response}")
    print("=" * 50)

if __name__ == "__main__":
    handle_customer_query("你们的API如何调用?")
    handle_customer_query("企业版的价格是多少?")