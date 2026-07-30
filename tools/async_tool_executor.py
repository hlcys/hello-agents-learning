# async_tool_executor
# 异步工具支持

import asyncio
import concurrent.futures
import threading
import time
from typing import Dict, Any, Optional, List, Callable
from hello_agents import ToolRegistry


class AsyncToolExecutor:
    """异步工具执行器"""
    def __init__(self, registry: ToolRegistry, max_workers: int = 4):
        self.registry = registry
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers = max_workers)


    async def execute_tool_async(self, tool_name: str, input_data: str) -> str:
        """异步执行单个工具"""
        loop = asyncio.get_event_loop()

        def _execute():
            return self.registry.execute_tool(tool_name, input_data)

        result = await loop.run_in_executor(self.executor, _execute)
        return result


    async def execute_tools_parallel(self, tasks: List[Dict[str, str]]) -> List[str]:
        """并行执行多个任务"""
        print(f"🚀 开始并行执行 {len(tasks)} 个工具任务")

        # 创建异步任务
        async_tasks = []
        for task in tasks:
            tool_name  = task["tool_name"]
            input_data = task["input_data"]
            async_task = self.execute_tool_async(tool_name, input_data)
            async_tasks.append(async_task)

        results = await asyncio.gather(*async_tasks)
        print(f"✅ 所有工具任务执行完成")
        return results


    def __del__(self):
        """清理资源"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)


# eg1: 测试单个工具的异步执行
async def test_single_tool_execution():
    """验证同步工具可以在线程池中异步执行并返回结果。"""
    registry = ToolRegistry()
    registry.register_function(
        name="echo",
        description="返回接收到的内容",
        func=lambda input_data: f"echo: {input_data}",
    )
    executor = AsyncToolExecutor(registry, max_workers=1)

    try:
        result = await executor.execute_tool_async("echo", "Hello Agent")
        assert result == "echo: Hello Agent"
        print(f"✅ 单工具异步执行测试通过: {result}")
    finally:
        executor.executor.shutdown(wait=True)


# eg2: 测试多个工具的并发执行
async def test_parallel_execution():
    """验证多个阻塞工具会同时进入线程池，并保持结果顺序。"""
    active_count = 0
    max_active_count = 0
    count_lock = threading.Lock()

    def delayed_echo(input_data: str) -> str:
        """用 sleep 模拟网络请求等阻塞型 I/O。"""
        nonlocal active_count, max_active_count

        with count_lock:
            active_count += 1
            max_active_count = max(max_active_count, active_count)

        try:
            time.sleep(0.2)
            return f"finished: {input_data}"
        finally:
            with count_lock:
                active_count -= 1

    registry = ToolRegistry()
    registry.register_function(
        name="delayed_echo",
        description="等待一小段时间后返回接收到的内容",
        func=delayed_echo,
    )
    executor = AsyncToolExecutor(registry, max_workers=4)
    tasks = [
        {"tool_name": "delayed_echo", "input_data": "task-1"},
        {"tool_name": "delayed_echo", "input_data": "task-2"},
        {"tool_name": "delayed_echo", "input_data": "task-3"},
        {"tool_name": "delayed_echo", "input_data": "task-4"},
    ]

    started_at = time.perf_counter()
    try:
        results = await executor.execute_tools_parallel(tasks)
    finally:
        executor.executor.shutdown(wait=True)
    elapsed = time.perf_counter() - started_at

    expected_results = [
        "finished: task-1",
        "finished: task-2",
        "finished: task-3",
        "finished: task-4",
    ]
    assert results == expected_results
    assert max_active_count > 1, "任务没有并发进入线程池"

    for i, result in enumerate(results, 1):
        print(f"任务 {i} 结果: {result}")
    print(
        f"✅ 并发测试通过: 最大同时执行 {max_active_count} 个任务, "
        f"总耗时 {elapsed:.2f} 秒"
    )


# eg3: 测试空任务列表
async def test_empty_tasks():
    """没有任务时应直接返回空列表。"""
    registry = ToolRegistry()
    executor = AsyncToolExecutor(registry)

    try:
        results = await executor.execute_tools_parallel([])
        assert results == []
        print("✅ 空任务列表测试通过")
    finally:
        executor.executor.shutdown(wait=True)


async def run_tests():
    """依次运行本文件中的测试。"""
    await test_single_tool_execution()
    await test_parallel_execution()
    await test_empty_tasks()


if __name__ == "__main__":
    asyncio.run(run_tests())
