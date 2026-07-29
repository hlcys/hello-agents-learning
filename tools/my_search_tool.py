# Search_Tool
from typing import Optional, Any
from .base  import Tool

class SearchTool(Tool):
    """
    智能混合搜索

    支持多种搜索源引擎后端
    1. 混合模式(Hybrid) - 智能选择
    2. Tavily API - 专业 AI 搜索
    3. SerpApi (serapi) - 传统 google 搜索
    """

    def __init__(
        self,
        backend: str = "hybrid", 
        tavily_key: Optional[str] = None,
        serpapi_key: Optional[str] = None,
    ):
        super.__init__(
            name = "search",
            description = "一个智能的s"
        )
        self.backend = backend
        self.tavily_key = tavily_key or os.getenv("TAVILY_API_KEY")
        self.serpapi_key = serpapi_key or os.getenv("SERP_API_KEY")
        self.available_backends = []
        self._setup_backends()


    def _search_hybrid(self, query: str) -> str:
        """混合搜索 - 选择最佳来源"""
        if "tavily" in self.available_backends:
            try:
                return self._search_tavily(query)
            except Exception as e:
                print("⚠️ warning: Tavily 搜索失败: {e}")
                if "serpapi" in self.available_backends:
                    print("🔄 切换到SerpApi搜索")
                    return self._search_serpapi(query)
                
        elif "serpapi" in self.available_backends:
            try:
                return self._search_serpapi(query)
            except Exception as e:
                print(f"⚠️ SerpApi搜索失败: {e}")

        return "❌ 没有可用的搜索源, 请配置TAVILY_API_KEY或SERPAPI_API_KEY环境变量"


    def _search_tavily(self, query: str) -> str:
        response = self.tavily_client_search(
            query = query,
            search_depth   = "basic",
            include_answer = True,
            max_results = 3
        )

        result = f"🎯 Tavily AI搜索结果:{response.get("answer", "result: 未找到直接答案")}\n\n"
        for i, item in enumerate(response.get('result', [])[:3], 1):
            result += f"[{i}] {item.get('title', '')}\n"
            result += f"    {item.get('content', '')[:200]}...\n"
            result += f"    来源: {item.get('url', '')}\n\n"

        return result