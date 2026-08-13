# 连接到MCP服务器 --基于FastMCP 2.0
# 实现 MCP Client 完整功能

# 设计到资源的问题，需要关注关闭/释放操作

from dotenv import load_dotenv
load_dotenv()

import asyncio
from hello_agents.protocols import MCPClient

#(1) mcp连接
async def connect_to_server():
    # 方式一: 连接到社区提供的文件系统服务器
    client = MCPClient([
        "npx", "-y",
        "@modelcontextprotocol/server-filesystem",
        "."
    ])
    async with client:
        tools = await client.list_tools()
        print(f"可用工具: {[t['name'] for t in tools]}")

    # 方式二: 连接到自定义的MCP服务器
    client = MCPClient(["python", "my_mcp_server.py"])
    async with client:
        # 使用client...
        pass

asyncio.run(connect_to_server())

#(2) 发现工具
async def discover_tools():
    client =  MCPClient([
        "npx", "-y", "@modelcontextprotocol/server-filesystem", "."
    ])
    async with client:
        tools = await client.list_tools()

        print(f"服务器提供了 {len(tools)} 个工具：")
        for tool in tools:
            print(f"\n工具名称: {tool['name']}")
            print(f"描述: {tool.get('description', '无描述')}")

            if 'inputSchema' in tool:
                schema = tool['inputSchema']
                if 'properties' in schema:
                    print('参数')
                    for param_name, param_info in schema['properties'].items():
                        param_type = param_info.get('type', 'any')
                        param_desc = param_info.get('description', '')
                        print(f" - {param_name} ({param_type}): {param_desc}")

asyncio.run(discover_tools())


#(3) 调用工具:
async def use_tools():
    client = MCPClient(["npx", "-y", "@modelcontextprotocol/server-filesystem", "."])

    async with client:
        # 读取文件
        result = await client.call_tool("read_file", {"path": "my_README.md"})
        print(f"文件内容：\n{result}")

        # 列出目录
        result = await client.call_tool("list_directory", {"path": "."})
        print(f"当前目录文件：{result}")

        # 写入文件
        result = await client.call_tool("write_file", {
            "path": "output.txt",
            "content": "Hello from MCP!"
        })
        print(f"写入结果：{result}")

asyncio.run(use_tools())

# 一种更安全的方式:
async def safe_tool_call():
    client = MCPClient(["npx", "-y", "@modelcontextprotocol/server-filesystem", "."])

    async with client:
        try:
            # 尝试读取可能不存在的文件
            result = await client.call_tool("read_file", {"path": "nonexistent.txt"})
            print(result)
        except Exception as e:
            print(f"工具调用失败: {e}")
            # 可以选择重试、使用默认值或向用户报告错误

asyncio.run(safe_tool_call())

#(4) 访问资源:
resources = client.list_resources()
print(f"可用资源：{[r['uri'] for r in resources]}")

resources_content = client.read_resources("file//path/to/resource")
print(f"资源内容: {resources_content}")

#(5) 使用提示模板:
prompts = client.list_prompts()
print(f"可用提示: {p['name'] for p in prompts}")

prompt = client.get_prompt("code_review", {"language": "python"})
print(f"提示内容: {prompt}")