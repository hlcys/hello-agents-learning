#### 创建自己的MCP服务器

主要目的：
+ **封装业务逻辑**：将企业内部业务封装为标准化的MCP工具
+ **访问私有数据**：创建一个安全可控的接口或代理
+ **性能专项优化**：针对高频调用的场景，进行深度优化
+ **功能定制拓展**：实现标准MCP服务未提供的特定功能

通过 https://smithery.ai/ 提交自己的mcp server, 发布后可以通过CLI使用:

```
npm install -g @smithery/cli
smithery install weather-mcp-server
```

也可以在 Hello-Agents中使用
```
weather_tool = MCPTool(
    server_command = ["smithery", "run", "weather-mcp-server"]
)
```
