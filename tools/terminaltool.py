# TerminalTool: 终端工具
# 访问日志文件、为Agent提供安全的命令行执行能力
# 场景特点： 需要实时、轻量级的文件系统访问, 而不是预先索引和向量化

# 1. 设置白名单命令
# 只允许安全的只读命令
ALLOWED_COMMANDS = {
    # 文件列表与信息
    'ls', 'dir', 'tree',
    # 文件内容查看
    'cat', 'head', 'tail', 'less', 'more',
    # 文件搜索
    'find', 'grep', 'egrep', 'fgrep',
    # 文本处理
    'wc', 'sort', 'uniq', 'cut', 'awk', 'sed',
    # 目录操作
    'pwd', 'cd',
    # 文件信息
    'file', 'stat', 'du', 'df',
    # 其他
    'echo', 'which', 'whereis',
}
# ❌ 不允许的命令: rm

# 2. 工作目录设置
# 只能访问指定的工作目录
terminal.run({"command": "cat ./src/main.py"})  # ✅
terminal.run({"command": "cat /etc/passwd"})  # ❌ 不允许访问工作目录外的路径

# 3. 时间限制，防止无限制消耗
terminal = TerminalTool(
    workspace = "./project",
    timeout   = 30
)

# 4. 输出大小限制
terminal = TerminalTool(
    workspace = "./project",
    max_output_size = 10 * 1024 * 1024
)

# 核心功能：
# (1) 命令执行：
# (2) 目录导航：

def _execute(self, command: str) -> str:
    """执行命令"""
    try:
        # 当前目录下执行
        result = subprocess.run(
            command,
            shell = True,
            cwd = str(self.current_dir),
            capture_output = True,
            text = True,
            timeout = self.timeout,
            env = os.environ.copy()
        )

        output = result.stdout
        if result.stderr:
            output += f"\n[stderr]\n{result.stderr}"

        if len(output) > self.max_output_size:
            output = output[:self.max_output_size]
            output += f"\n\n ⚠️ 输出被截断（超过 {self.max_output_size} 字节）"

        if result.returncode != 0:
            output = f"⚠️ 命令返回码: {result.returncode}\n\n{output}"

        return output if output else "命令执行成功"

    except subprocess.TimeoutExpired:
        return f"❌ 命令执行超时（超过 {self.timeout} 秒）"
    except Exception as e:
        return f"❌ 命令执行失败: {e}"


def _handle_cd(self, parts: List[str]) -> str:
    """处理 cd 命令, 支持智能体在文件系统中导航"""
    if not self.allow_cd:
        return "❌ cd 命令已禁用"

    if len(parts) < 2:
        return f"当前目录: {self.current_dir}"

    target_dir = parts[1]
    if target_dir == "..":
        new_dir = self.current_dir.parent
    elif target_dir == ".":
        new_dir = self.current_dir
    elif target_dir == "~":
        new_dir = self.workspace
    else:
        # 拼接路径 -> resolve() 转换
        new_dir = (self.current_dir / target_dir).resolve()

    # 检查是否在工作目录内
    try:
        new_dir.relative_to(self.workspace)
    except ValueError:
        return f"❌ 不允许访问工作目录外的路径: {new_dir}"

    if not new_dir.exists():
        return f"❌ 目录不存在: {new_dir}"
    if not new_dir.is_dir():
        return f"❌ {new_dir} 不是目录"

    self.current_dir = new_dir
    return f"✅ 切换到目录: {self.current_dir}"

# TerminalTtool 支持 Agent进行文件系统探索
# 能够连接数据文件的结构和内容, 以及日志内容