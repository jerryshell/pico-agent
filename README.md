# Pico Agent

Pico 是一个轻量级 LLM Agent

```bash
npm install -g @jerryshell/pico-agent
```

## 工具

| 工具    | 说明            |
| ------- | --------------- |
| `read`  | 读取文件        |
| `write` | 写入文件        |
| `edit`  | 编辑文件        |
| `bash`  | 执行 shell 命令 |
| `view`  | 查看图片        |
| `skill` | 按需加载技能    |

## 技能系统

在 `~/.agents/skills/` 或 `.agents/skills/` 中按 `SKILL.md` 格式编写，LLM 通过 `skill` 工具按需加载。

## 使用

```bash
pa config apiKey yourKimiCodeApiKey
```

配置保存在 `~/.pico-agent/config.json`。也可通过 `API_KEY` 环境变量指定。

```bash
pa "简单总结我的系统状态"
```

### 环境变量

- `API_KEY` — API 密钥（优先级高于配置文件）
- `NO_COLOR` — 禁用彩色输出

## 开发

```bash
bun install
bun run index.ts "简单总结我的系统状态"
bun run build    # 输出到 dist/
```
