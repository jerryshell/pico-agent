# Pico Agent

Pico 是一个轻量级 LLM Agent

## 工具

| 工具    | 说明                                         |
| ------- | -------------------------------------------- |
| `read`  | 读取文件（>50KB 自动截断）                   |
| `write` | 创建或重写文件（自动创建父目录）             |
| `edit`  | 编辑文件（支持单次多处编辑，oldText 须唯一） |
| `bash`  | 运行 shell 命令（可设 timeout 防挂起）       |
| `skill` | 按需加载技能详细内容（见下方技能系统）       |

## 技能系统

自动发现以下目录中的技能（SKILL.md 或 .md 文件）：

- `~/.agents/skills/` — 全局技能
- `.agents/skills/` — 项目级技能

技能文件须含 YAML frontmatter 定义 `name` 和 `description`，LLM 通过 `skill` 工具按需加载。

## 使用

### Kimi Code

适配 Kimi Code 供应商：

```bash
pa config apiKey yourKimiCodeApiKey
```

配置保存在 `~/.pico-agent/config.json`。也可通过 `API_KEY` 环境变量临时指定。

```bash
pa "简单总结我的系统状态"
```

## 开发

```bash
bun install
bun run index.ts "简单总结我的系统状态"
bun run build    # 输出到 dist/
```
