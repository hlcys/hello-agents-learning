"""测试天气查询 MCP 服务器"""

import asyncio
import json

from hello_agents.protocols.mcp.client import MCPClient

# 同时兼容直接运行脚本和以包模块方式导入。
try:
    from .main import weather_server
except ImportError:
    from main import weather_server


async def test_weather_server():
    try:
        # 测试代码与服务端在同一进程中运行，避免 stdio 受 Python/依赖版本影响。
        client = MCPClient(weather_server.mcp)
        async with client:
            # 测试1: 获取服务器信息
            info = json.loads(await client.call_tool(
                "get_server_info", {}
            ))
            print(f"服务器: {info['name']} v{info['version']}")

            # 测试2: 列出支持的城市
            cities = json.loads(await client.call_tool(
                "list_supported_cities", {}
            ))
            print(f"支持城市: {cities['count']}个")

            # 测试3: 查询北京天气
            weather = json.loads(await client.call_tool(
                "get_weather", {"city": "北京"}
            ))
            if "error" not in weather:
                print(f"\n北京天气: {weather['temperature']}°C, {weather['condition']}")

            weather = json.loads(await client.call_tool("get_weather", {"city": "深圳"}))
            if "error" not in weather:
                print(f"深圳天气: {weather['temperature']}°C, {weather['condition']}")

            print("\n 所有测试完成!")
    except Exception as e:  # 测试入口需要汇总所有连接或调用错误。
        print(f"❌ 测试失败: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(test_weather_server())
