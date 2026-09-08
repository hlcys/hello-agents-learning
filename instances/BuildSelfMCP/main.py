"""天气查询 MCP 服务器"""

import json
from datetime import datetime
from typing import Any

import requests
from hello_agents.protocols import MCPServer

weather_server = MCPServer(
    name = "weather-server",
    description = "真实天气查询"
)

CITY_MAP = {
    "北京": "Beijing", "上海": "Shanghai", "广州": "Guangzhou",
    "深圳": "Shenzhen", "杭州": "Hangzhou", "成都": "Chengdu",
    "重庆": "Chongqing", "武汉": "Wuhan", "西安": "Xi'an",
    "南京": "Nanjing", "天津": "Tianjin", "苏州": "Suzhou"
}

def get_weather_data(city: str) -> dict[str, Any]:
    """从 wttr.in 中获取天气"""
    # wttr.in 对英文城市名的识别更稳定，因此优先转换已知中文城市名。
    city_en = CITY_MAP.get(city, city)
    url = f"https://wttr.in/{city_en}?format=j1"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.json()
    current = data["current_condition"][0]

    return {
        "city": city,
        "temperature": float(current["temp_C"]),
        "feels_like":  float(current["FeelsLikeC"]),
        "humidity": int(current["humidity"]),
        "condition": current["weatherDesc"][0]["value"],
        "wind_speed": round(float(current["windspeedKmph"]) / 3.6, 1),
        "visibility": float(current["visibility"]),
        # 记录本机时区偏移，避免时间戳含义不明确。
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds")
    }

# 定义工具函数
def get_weather(city: str) -> str:
    """查询指定城市的实时天气，并返回 JSON 字符串。"""
    try:
        weather_data = get_weather_data(city)
        return json.dumps(weather_data, ensure_ascii = False, indent = 2)
    except (requests.RequestException, KeyError, IndexError, TypeError, ValueError) as e:
        # MCP 工具统一返回结构化错误，避免外部接口异常中断服务器。
        return json.dumps({
            "error": str(e),
            "city":  city
        }, ensure_ascii = False)


def list_supported_cities() -> str:
    """列出所有支持的中文城市"""
    result = {
        "cities": list(CITY_MAP.keys()),
        "count":  len(CITY_MAP)
    }
    return json.dumps(
        result,
        ensure_ascii = False,
        indent = 2
    )


def get_server_info() -> str:
    """获取服务器信息"""
    info = {
        "name" : "Weather MCP Server",
        "version" : "1.0.0",
        "tools": [
            "get_weather",
            "list_supported_cities",
            "get_server_info"
        ]
    }
    return json.dumps(info, ensure_ascii=False, indent=2)

# 将工具注册到服务器
weather_server.add_tool(get_weather)
weather_server.add_tool(list_supported_cities)
weather_server.add_tool(get_server_info)


if __name__ == "__main__":
    weather_server.run()
