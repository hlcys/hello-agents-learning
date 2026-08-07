"""对当前 hello-agents 仓库执行一次简易的代码库探索测试。"""

import json
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except ModuleNotFoundError as exc:
    raise SystemExit(
        "未找到 python-dotenv, 请先在当前 Python 环境中安装项目依赖。"
    ) from exc


INSTANCE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = INSTANCE_DIR.parents[1]
OUTPUT_FILE = INSTANCE_DIR / "output.txt"

# 无论从哪个目录启动脚本，都加载项目根目录中的配置并导入同目录的 main.py。
load_dotenv(PROJECT_ROOT / ".env")
sys.path.insert(0, str(INSTANCE_DIR))

try:
    from main import CodeBaseMaintainer
except ModuleNotFoundError as exc:
    if exc.name and exc.name.startswith("hello_agents"):
        raise SystemExit(
            "未找到 hello_agents 包，请先在当前 Python 环境中安装该项目依赖。"
        ) from exc
    raise


def main() -> None:
    """初始化维护助手，并分析当前 hello-agents 代码库。"""
    os.chdir(PROJECT_ROOT)

    print(f"待分析代码库: {PROJECT_ROOT}")
    maintainer = CodeBaseMaintainer(
        project_name="hello-agents",
        codebase_path=str(PROJECT_ROOT),
    )

    response = maintainer.run(
        user_input=(
            "请快速分析当前 hello-agents 代码库的目录结构"
            "指出目前最值得关注的三个维护的具体问题, 给出重构计划"
        ),
        mode="explore",
    )

    stats_json = json.dumps(
        maintainer.get_stats(),
        ensure_ascii=False,
        indent=2,
        default=str,
    )
    output = (
        f"待分析代码库: {PROJECT_ROOT}\n\n"
        f"测试结果:\n{response}\n\n"
        f"运行统计:\n{stats_json}\n"
    )
    OUTPUT_FILE.write_text(output, encoding="utf-8")

    print("\n测试结果:")
    print(response)
    print("\n运行统计:")
    print(stats_json)
    print(f"\n结果已保存到: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
