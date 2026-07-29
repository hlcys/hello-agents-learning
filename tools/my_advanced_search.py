# my_advanced_search

import os
from typing import Optional, Dict, Any, List
from hello_agents import ToolRegistry

class MyAdvancedSearchTool:
    """
    自定义高级搜索工具
    """

    def __init__(self):
        self.name = "my_advanced_search"
        self.description = "智能搜索工具"
        self.search_sources = []
        self._setup_search_sources()


    def _setup_search_sources(self):
        """设置可用的搜索源"""
        if os.getenv("TAVILY_API_KEY"):
            try:
                from tavily import TavilyClient
                self.tavily_client = TavilyClient(api_key = os.getenv("TAVILY_API_KEY"))
                self.search_sources.append("tavily")
                print(f"✅ Tavily 搜索源已启动")

            except ImportError:
                print(f"⚠️ warning: Tavily 库未安装")

        if os.getenv("SERPAPI_API_KEY"):
            try:
                import serpapi
                self.search_sources.append("serpapi")
                print(f"✅ Serpapi 搜索源已启动")
            except ImportError:
                print(f"⚠️ warning: Serpapi 库未安装")

        if self.search_sources:
            print(f"🔧 可用搜索源: {', '.join(self.search_sources)}")
        else:
            print("⚠️ 没有可用的搜索源, 请配置API密钥")


    def search(self, query: str) -> str:
        """执行智能搜索"""
        if not query.strip():
            return f"❌ error: 搜索查询为空!"

        if not self.search_sources:
            return f"❌ error: 没有可用搜索源"

        print(f"🔍 开始搜索: {query}")

        for source in self.search_sources:
            try:
                if source == "tavily":
                    result = self._search_with_tavily(query)
                    if result and "未找到" not in result:
                        return f"📊 Tavily AI搜索结果:\n\n{result}"

                elif source == "serpapi":
                    result = self._search_with_serpapi(query)
                    if result and "未找到" not in result:
                        return f"🌐 SerpApi Google搜索结果:\n\n{result}"

            except Exception as e:
                print(f"⚠️ {source} 搜索失败: {e}")
                continue

        return "❌ error : 所有搜索源都失败, 请检查网络连接和API密钥配置"

    
    def _search_with_tavily(self, query: str) -> str:
        """使用Tavily搜索"""
        response = self.tavily_client.search(query=query, max_results=3)

        if response.get('answer'):
            result = f"💡 AI直接答案:{response['answer']}\n\n"
        else:
            result = ""

        result += "🔗 相关结果:\n"
        for i, item in enumerate(response.get('results', [])[:3], 1):
            result += f"[{i}] {item.get('title', '')}\n"
            result += f"    {item.get('content', '')[:150]}...\n\n"

        return result
    

    def _search_with_serpapi(self, query: str) -> str:
        """使用 serpapi """
        import serpapi

        search = serpapi.GoogleSearch({
            "q": query,
            "api_key": os.getenv("SERPAPI_API_KEY"),
            "num": 3
        })

        results = search.get_dict()
        result  = "🔗 Google搜索结果:\n"

        if "organic_results" in results:
            for i, res in enumerate(results["organic_results"][:3], 1):
                result += f"[{i}] {res.get('title', '')}\n"
                result += f"    {res.get('snippet', '')}\n\n"

        return result


def create_advanced_search_registry():
    """ 创建包含高级搜索工具的注册表 """
    registry = ToolRegistry()
    search_tool = MyAdvancedSearchTool()

    registry.register_function(
        name = "advanced_search",
        description = "高级搜索工具",
        func = search_tool.search
    )

    return registry