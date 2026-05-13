# Pico Agent

Pico 是一个轻量级 Agent，给 LLM 提供了 4 种工具：

- `read` - 读取文件
- `write` - 创建或重写文件
- `edit` - 编辑文件
- `bash` - 运行 shell 命令

## 快速开始

目前只适配了 Kimi Code 供应商

### 1. 全局安装

```bash
npm install -g @jerryshell/pico-agent
```

### 2. 配置 API Key

```bash
pa config apiKey yourKimiCodeApiKey
```

配置保存在 `~/.pico-agent/config.json`

### 3. 运行

```bash
pa "简单总结我的系统状态"
```

> 如果未配置 apiKey，会提示：`没有检测到 apiKey，请使用 pa config apiKey yourKimiCodeApiKey 进行配置`

你也可以通过 `API_KEY` 环境变量临时指定：

```bash
API_KEY=xxx pa "简单总结我的系统状态"
```

## 开发环境

安装依赖

```bash
bun install
```

运行

```bash
bun run index.ts "简单总结我的系统状态"
```

构建

```bash
bun run build
```
