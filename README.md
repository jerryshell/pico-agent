# Pico Agent

Pico 是一个轻量级 LLM Agent，内置了以下工具

| 工具    | 说明            |
| ------- | --------------- |
| `read`  | 读取文件        |
| `write` | 写入文件        |
| `edit`  | 编辑文件        |
| `bash`  | 执行 shell 命令 |
| `view`  | 查看图片        |
| `skill` | 按需加载技能    |

Pico 会从以下目录按需加载技能

- `~/.agents/skills/`
- `./.agents/skills/`

## 快速开始

1. 全局安装

```bash
npm install -g @jerryshell/pico-agent
```

2. 配置 Kimi Code API Key

```bash
pa config apiKey yourKimiCodeApiKey
```

配置保存在 `~/.pico-agent/config.json`。也可通过 `API_KEY` 环境变量指定。

3. 运行

```bash
pa "简单总结我的系统状态"
```
