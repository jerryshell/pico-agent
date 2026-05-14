#!/usr/bin/env node
import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import { ToolLoopAgent, isLoopFinished, tool, zodSchema } from "ai";
import { exec } from "child_process";
import { mkdir, readdir, readFile, stat, writeFile } from "fs/promises";
import { homedir } from "os";
import { join } from "path";
import { parseArgs, promisify } from "util";
import { z } from "zod";

const execAsync = promisify(exec);

const CONFIG_DIR = join(homedir(), ".pico-agent");
const CONFIG_FILE = join(CONFIG_DIR, "config.json");
const HOME_SKILLS_DIR = join(homedir(), ".agents", "skills");
const LOCAL_SKILLS_DIR = join(process.cwd(), ".agents", "skills");

// === 最高层 ===

async function main() {
  const command = parseCommand();

  if (command.mode === "config") {
    await saveConfig({ apiKey: command.value });
    console.log(`apiKey 已保存到 ${CONFIG_FILE}`);
    return;
  }

  const apiKey = await resolveApiKey();
  if (!apiKey) {
    console.error("没有检测到 apiKey，请使用 pa config apiKey yourKimiCodeApiKey 进行配置");
    process.exit(1);
  }

  await runAgent(apiKey, command.prompt);
}

// === 命令解析 ===

function parseCommand() {
  const { positionals } = parseArgs({
    strict: false,
    allowPositionals: true,
  });

  if (positionals[0] === "config") {
    const [, key, value] = positionals;
    if (key === "apiKey" && value) {
      return { mode: "config" as const, key, value };
    }
    console.error("用法: pa config apiKey yourKimiCodeApiKey");
    process.exit(1);
  }

  if (positionals.length === 0) {
    console.error("用法: pa <prompt>");
    process.exit(1);
  }

  return {
    mode: "run" as const,
    prompt: positionals.join(" "),
  };
}

// === apiKey 解析 ===

async function resolveApiKey(): Promise<string | undefined> {
  const envKey = process.env.API_KEY;
  if (envKey) return envKey;

  const fileConfig = await loadConfig();
  return fileConfig.apiKey;
}

// === Agent 运行 ===

async function runAgent(apiKey: string, prompt: string) {
  const provider = createOpenAICompatible({
    name: "Kimi For Coding",
    apiKey,
    baseURL: "https://api.kimi.com/coding/v1",
  });

  const model = provider("kimi-for-coding");

  const skills = await discoverSkills();
  skills.sort((a, b) => a.name.localeCompare(b.name));
  if (skills.length > 0) {
    console.log(`发现 ${skills.length} 个技能:`);
    for (const skill of skills) {
      console.log(`  ${cyan("◉")} ${bold(skill.name)}: ${skill.description}`);
    }
    console.log();
  }
  const baseTools = createTools();
  const tools = skills.length > 0 ? { ...baseTools, skill: createSkillTool(skills) } : baseTools;

  const agent = new ToolLoopAgent({
    model,
    headers: { "User-Agent": "KimiCLI/1.5" },
    instructions: buildSkillsPrompt(skills),
    tools,
    stopWhen: isLoopFinished(),
  });

  const result = await agent.stream({ prompt });

  for await (const chunk of result.fullStream) {
    printChunk(chunk);
  }
}

// === 工具 ===

function createTools() {
  const read = tool({
    description: "读取文件内容",
    inputSchema: zodSchema(z.object({ path: z.string().describe("文件路径") })),
    execute: async ({ path }: { path: string }) => {
      try {
        return await readFile(path, "utf-8");
      } catch (error) {
        throw new Error(`无法读取文件 ${path}: ${error}`);
      }
    },
  });

  const write = tool({
    description: "创建或重写文件",
    inputSchema: zodSchema(
      z.object({
        path: z.string().describe("文件路径"),
        content: z.string().describe("文件内容"),
      }),
    ),
    execute: async ({ path, content }: { path: string; content: string }) => {
      try {
        await writeFile(path, content, "utf-8");
        return `文件 ${path} 已写入`;
      } catch (error) {
        throw new Error(`无法写入文件 ${path}: ${error}`);
      }
    },
  });

  const edit = tool({
    description: "编辑文件（替换指定字符串）",
    inputSchema: zodSchema(
      z.object({
        path: z.string().describe("文件路径"),
        oldStr: z.string().describe("要替换的原始字符串"),
        newStr: z.string().describe("替换后的新字符串"),
      }),
    ),
    execute: async ({ path, oldStr, newStr }: { path: string; oldStr: string; newStr: string }) => {
      try {
        const content = await readFile(path, "utf-8");
        if (!content.includes(oldStr)) {
          throw new Error(`文件 ${path} 中未找到要替换的字符串`);
        }
        await writeFile(path, content.replace(oldStr, newStr), "utf-8");
        return `文件 ${path} 已更新`;
      } catch (error) {
        throw new Error(`编辑文件失败: ${error}`);
      }
    },
  });

  const bash = tool({
    description: "运行 shell 命令",
    inputSchema: zodSchema(z.object({ command: z.string().describe("要执行的 shell 命令") })),
    execute: async ({ command }: { command: string }) => {
      try {
        const { stdout, stderr } = await execAsync(command);
        return stdout + stderr;
      } catch (error: any) {
        return `命令执行失败: ${error.message}`;
      }
    },
  });

  return { read, write, edit, bash };
}

// === Chunk 输出 ===

function printChunk(chunk: any) {
  switch (chunk.type) {
    case "text-delta":
      process.stdout.write(chunk.text);
      break;
    case "text-start":
    case "text-end":
      process.stdout.write("\n");
      break;
    case "reasoning-start":
      process.stdout.write(`\n${separator()}\n${yellow(bold("思考"))}\n${separator()}\n`);
      break;
    case "reasoning-delta":
      process.stdout.write(yellow(chunk.text));
      break;
    case "reasoning-end":
      process.stdout.write(`\n${separator()}\n`);
      break;
    case "tool-call":
      process.stdout.write(
        `\n${separator()}\n${cyan(bold("工具调用"))} ${cyan(chunk.toolName)}\n${separator()}\n`,
      );
      process.stdout.write(cyan(JSON.stringify(chunk.input, null, 2)) + "\n");
      break;
    case "tool-input-start":
      process.stdout.write(`\n${separator()}\n${cyan(bold("工具输入"))} ${cyan(chunk.toolName)}\n`);
      break;
    case "tool-input-delta":
      process.stdout.write(cyan(chunk.delta));
      break;
    case "tool-input-end":
      process.stdout.write(`\n${separator()}\n`);
      break;
    case "tool-result":
      process.stdout.write(`${green(bold("工具结果"))} ${green(chunk.toolName)}\n`);
      const output =
        typeof chunk.output === "string" ? chunk.output : JSON.stringify(chunk.output, null, 2);
      process.stdout.write(green(output) + "\n");
      break;
    case "tool-error":
      process.stdout.write(`${red(bold("工具错误"))} ${red(chunk.toolName)}\n`);
      process.stdout.write(red(String(chunk.error)) + "\n");
      break;
    case "tool-output-denied":
      process.stdout.write(`${yellow(bold("输出被拒绝"))} ${yellow(chunk.toolName)}\n`);
      break;
    case "tool-approval-request":
      process.stdout.write(`${magenta(bold("等待审批"))} ${magenta(chunk.toolCall.toolName)}\n`);
      break;
    case "source":
      process.stdout.write(
        `${dim("[来源]")} ${dim(chunk.sourceType === "url" ? chunk.url : chunk.title)}\n`,
      );
      break;
    case "file":
      process.stdout.write(`${blue(bold("文件"))} ${blue(chunk.file.mediaType)}\n`);
      break;
    case "start-step":
      process.stdout.write(
        `\n${dim("=".repeat(40))}\n${bold("步骤开始")}\n${dim("=".repeat(40))}\n`,
      );
      break;
    case "finish-step":
      process.stdout.write(
        `${dim(`步骤结束 | 结束原因: ${chunk.finishReason} | tokens: ${JSON.stringify(chunk.usage)}`)}\n`,
      );
      break;
    case "start":
      process.stdout.write(`${bold("开始")}\n`);
      break;
    case "finish":
      process.stdout.write(
        `${bold("完成")} | 结束原因: ${chunk.finishReason} | tokens: ${JSON.stringify(chunk.totalUsage)}\n`,
      );
      break;
    case "abort":
      process.stdout.write(`${red(bold("中断"))}${chunk.reason ? `: ${chunk.reason}` : ""}\n`);
      break;
    case "error":
      process.stdout.write(`${red(bold("错误"))}: ${String(chunk.error)}\n`);
      break;
    case "raw":
      break;
  }
}

// === 配置读写 ===

interface Config {
  apiKey?: string;
}

async function loadConfig(): Promise<Config> {
  try {
    const content = await readFile(CONFIG_FILE, "utf-8");
    return JSON.parse(content) as Config;
  } catch {
    return {};
  }
}

async function saveConfig(config: Config): Promise<void> {
  await mkdir(CONFIG_DIR, { recursive: true });
  await writeFile(CONFIG_FILE, JSON.stringify(config, null, 2) + "\n", "utf-8");
}

// === Skill 自动发现 ===

interface Skill {
  name: string;
  description: string;
  dirPath: string;
}

async function discoverSkills(): Promise<Skill[]> {
  const all = new Map<string, Skill>();

  for (const dir of [HOME_SKILLS_DIR, LOCAL_SKILLS_DIR]) {
    try {
      const entries = await readdir(dir);
      for (const entry of entries) {
        const dirPath = join(dir, entry);
        const entryStat = await stat(dirPath);
        if (!entryStat.isDirectory()) continue;

        const skillMdPath = join(dirPath, "SKILL.md");
        let content: string;
        try {
          content = await readFile(skillMdPath, "utf-8");
        } catch {
          continue;
        }

        const parsed = parseFrontmatter(content);
        if (parsed.name) {
          all.set(parsed.name, {
            name: parsed.name,
            description: parsed.description ?? "",
            dirPath,
          });
        }
      }
    } catch {
      // 目录不存在，跳过
    }
  }

  return [...all.values()];
}

function parseFrontmatter(content: string): { name?: string; description?: string } {
  const match = content.match(/^---\n([\s\S]*?)---\n/);
  if (!match) return {};

  const lines = match[1]!.split("\n");
  const result: { name?: string; description?: string } = {};

  for (let i = 0; i < lines.length; i++) {
    const header = lines[i]!.match(/^(\w+):\s*(.*)$/);
    if (!header) continue;

    const key = header[1]!;
    let value = header[2]!;
    const isBlock = value === ">" || value === "|";

    if (isBlock) {
      const parts: string[] = [];
      i++;
      while (i < lines.length && (lines[i]!.startsWith(" ") || lines[i]!.startsWith("\t"))) {
        parts.push(lines[i]!.trim());
        i++;
      }
      i--;
      value = value === "|" ? parts.join("\n").trim() : parts.join(" ").replace(/\s+/g, " ").trim();
    }

    if (key === "name") result.name = value;
    else if (key === "description") result.description = value;
  }

  return result;
}

function buildSkillsPrompt(skills: Skill[]): string | undefined {
  if (skills.length === 0) return undefined;

  return [
    "你可以通过 skill 工具按需加载特定技能的详细内容（如技术参考、代码模板、库用法等）。",
    "",
    "可用技能：",
    ...skills.map(
      (skill) => `  - ${skill.name}${skill.description ? `: ${skill.description}` : ""}`,
    ),
    "",
    "当用户请求涉及以上领域时，先调用 skill 工具加载对应内容，再按文档执行。",
  ].join("\n");
}

function createSkillTool(skills: Skill[]) {
  const index = new Map(skills.map((s) => [s.name, s]));

  return tool({
    description: "加载指定技能的详细内容",
    inputSchema: zodSchema(z.object({ name: z.string().describe("技能名称") })),
    execute: async ({ name }: { name: string }) => {
      const skill = index.get(name);
      if (!skill) throw new Error(`未知技能 "${name}"，可用: ${[...index.keys()].join(", ")}`);
      return await readFile(join(skill.dirPath, "SKILL.md"), "utf-8");
    },
  });
}

// === 视觉辅助 ===

const dim = (s: string) => `\x1b[2m${s}\x1b[22m`;
const cyan = (s: string) => `\x1b[36m${s}\x1b[39m`;
const green = (s: string) => `\x1b[32m${s}\x1b[39m`;
const yellow = (s: string) => `\x1b[33m${s}\x1b[39m`;
const red = (s: string) => `\x1b[31m${s}\x1b[39m`;
const blue = (s: string) => `\x1b[34m${s}\x1b[39m`;
const magenta = (s: string) => `\x1b[35m${s}\x1b[39m`;
const bold = (s: string) => `\x1b[1m${s}\x1b[22m`;
const separator = () => dim("─".repeat(40));

main();
