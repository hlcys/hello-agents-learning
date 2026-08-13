"""
不使用 MCP 的多 Agent 智能文档助手。

与 main.py 的区别：
- 手写 GitHubSearchTool，直接调用 GitHub REST API。
- 手写 MarkdownFileTool，直接使用 pathlib 写入文件。
- 由 Python 代码显式编排“搜索 -> 整理 -> 写作 -> 保存”流程。

运行示例：
    python main_without_mcp.py "AI Agent" --limit 5

可选环境变量：
    GITHUB_TOKEN 或 GITHUB_PERSONAL_ACCESS_TOKEN
    配置 Token 可提高 GitHub API 的访问限额。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from hello_agents import HelloAgentsLLM, SimpleAgent
from hello_agents.tools import Tool, ToolParameter


SCRIPT_DIR = Path(__file__).resolve().parent


class GitHubSearchTool(Tool):
    """不通过 MCP Server，直接封装 GitHub 仓库搜索 API。"""

    def __init__(self, timeout: float = 15.0):
        super().__init__(
            name="github_search",
            description="根据关键词搜索 GitHub 仓库，返回仓库名、描述、链接和 Star 数。",
        )
        self.timeout = timeout

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="query",
                type="string",
                description="GitHub 仓库搜索关键词",
            ),
            ToolParameter(
                name="limit",
                type="integer",
                description="返回的仓库数量（1-10）",
                required=False,
                default=5,
            ),
        ]

    def run(self, parameters: dict[str, Any]) -> str:
        query = str(parameters.get("query") or parameters.get("input") or "").strip()
        if not query:
            raise ValueError("query 不能为空")

        try:
            limit = max(1, min(int(parameters.get("limit", 5)), 10))
        except (TypeError, ValueError) as exc:
            raise ValueError("limit 必须是 1-10 之间的整数") from exc

        url = "https://api.github.com/search/repositories?" + urlencode(
            {
                "q": query,
                "sort": "stars",
                "order": "desc",
                "per_page": limit,
            }
        )
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "Hello-Agents-SmartDocAssistant",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        token = os.getenv("GITHUB_TOKEN") or os.getenv(
            "GITHUB_PERSONAL_ACCESS_TOKEN"
        )
        if token:
            headers["Authorization"] = f"Bearer {token}"

        request = Request(url, headers=headers)
        try:
            with urlopen(request, timeout=self.timeout) as response:
                payload = json.load(response)
        except HTTPError as exc:
            message = self._read_api_error(exc)
            raise RuntimeError(f"GitHub API 请求失败（{exc.code}）: {message}") from exc
        except URLError as exc:
            raise RuntimeError(f"无法连接 GitHub API: {exc.reason}") from exc

        repositories = [
            {
                "name": item.get("full_name", ""),
                "description": item.get("description") or "暂无描述",
                "url": item.get("html_url", ""),
                "stars": item.get("stargazers_count", 0),
            }
            for item in payload.get("items", [])[:limit]
        ]
        return json.dumps(repositories, ensure_ascii=False, indent=2)

    @staticmethod
    def _read_api_error(exc: HTTPError) -> str:
        try:
            payload = json.loads(exc.read().decode("utf-8"))
            return payload.get("message", str(exc.reason))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return str(exc.reason)


class MarkdownFileTool(Tool):
    """不通过 MCP Server，直接使用 Python 写入 Markdown 文件。"""

    def __init__(self, workspace: Path):
        super().__init__(
            name="write_markdown",
            description="将 Markdown 内容保存到文档助手目录。",
        )
        self.workspace = workspace.resolve()

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="path",
                type="string",
                description="相对于工作目录的 Markdown 文件路径",
            ),
            ToolParameter(
                name="content",
                type="string",
                description="要写入的 Markdown 内容",
            ),
        ]

    def run(self, parameters: dict[str, Any]) -> str:
        relative_path = str(parameters.get("path") or "").strip()
        content = parameters.get("content")
        if not relative_path:
            raise ValueError("path 不能为空")
        if not isinstance(content, str):
            raise ValueError("content 必须是字符串")

        target = (self.workspace / relative_path).resolve()
        try:
            target.relative_to(self.workspace)
        except ValueError as exc:
            raise ValueError("不允许将文件写入工作目录之外") from exc
        if target.suffix.lower() not in {".md", ".markdown"}:
            raise ValueError("输出文件必须使用 .md 或 .markdown 后缀")

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return str(target)


def create_agents(llm: HelloAgentsLLM) -> tuple[SimpleAgent, SimpleAgent]:
    """创建搜索结果整理 Agent 和文档写作 Agent。"""
    github_searcher = SimpleAgent(
        name="GitHub搜索专家",
        llm=llm,
        system_prompt="""你是一个 GitHub 搜索结果整理专家。
请仅根据用户提供的 API 结果进行整理，不得虚构仓库。
对每个仓库输出名称、简要描述、链接和 Star 数，保持简洁。
""",
    )
    document_writer = SimpleAgent(
        name="文档生成专家",
        llm=llm,
        system_prompt="""你是一个文档生成专家。
请根据给定的 GitHub 仓库信息生成结构化 Markdown 报告，包括：
- 标题
- 简介
- 主要内容（分项列出项目名称、描述、链接和 Star 数）
- 总结
只输出完整的 Markdown 内容，不要包裹代码块。
""",
    )
    return github_searcher, document_writer


def normalize_markdown(content: str) -> str:
    """移除模型偶尔添加的 Markdown 代码块外壳。"""
    result = content.strip()
    if result.startswith("```markdown") and result.endswith("```"):
        result = result[len("```markdown") : -len("```")].strip()
    elif result.startswith("```") and result.endswith("```"):
        result = result[len("```") : -len("```")].strip()
    return result + "\n"


def build_report(query: str, limit: int, output: str) -> Path:
    """显式编排完整文档生成流程。"""
    github_tool = GitHubSearchTool()
    file_tool = MarkdownFileTool(SCRIPT_DIR)
    github_searcher, document_writer = create_agents(HelloAgentsLLM())

    print(f"\n[1/4] 搜索 GitHub 仓库: {query}")
    raw_repositories = github_tool.run({"query": query, "limit": limit})

    print("[2/4] GitHub 搜索专家正在整理结果...")
    search_summary = github_searcher.run(
        f"搜索关键词：{query}\n\nGitHub API 结果：\n{raw_repositories}"
    )

    print("[3/4] 文档生成专家正在撰写报告...")
    markdown = document_writer.run(
        f"请为搜索主题“{query}”生成报告。\n\n仓库信息：\n{search_summary}"
    )
    markdown = normalize_markdown(markdown)

    print("[4/4] 保存 Markdown 报告...")
    saved_path = file_tool.run({"path": output, "content": markdown})
    return Path(saved_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="不使用 MCP 的智能文档助手")
    parser.add_argument(
        "query",
        nargs="?",
        default="AI Agent",
        help="GitHub 仓库搜索关键词（默认：AI Agent）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        choices=range(1, 11),
        metavar="1-10",
        help="返回的仓库数量（默认：5）",
    )
    parser.add_argument(
        "--output",
        default="github_report.md",
        help="保存在当前脚本目录下的 Markdown 文件（默认：github_report.md）",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    print("=" * 70)
    print("多智能体协作的智能文档助手（无 MCP 版）")
    print("=" * 70)

    try:
        saved_path = build_report(args.query, args.limit, args.output)
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"\n执行失败: {exc}") from exc

    print(f"\n报告已保存: {saved_path}")


if __name__ == "__main__":
    main()
