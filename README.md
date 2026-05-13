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

### 2. 运行

```bash
pa --api-key=xxx "简单总结我的系统状态"
```

## 开发环境

安装依赖

```bash
bun install
```

运行

```bash
bun run index.ts --api-key=xxx "简单总结我的系统状态"
```
