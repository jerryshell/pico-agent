#!/usr/bin/env node
import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import { ToolLoopAgent, isLoopFinished, tool, zodSchema } from "ai";
import type { TextStreamPart } from "ai";
import { exec } from "child_process";
import { mkdir, readdir, readFile, stat, writeFile } from "fs/promises";
import { homedir } from "os";
import { basename, dirname, join } from "path";
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

  process.on("SIGINT", () => {
    process.stderr.write("\n");
    process.exit(130);
  });

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
    log(`发现 ${skills.length} 个技能: ${skills.map((s) => bold(s.name)).join(", ")}`);
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

function printChunk(chunk: TextStreamPart<any>) {
  switch (chunk.type) {
    case "text-delta":
      process.stdout.write(chunk.text);
      break;
    case "text-start":
    case "text-end":
      process.stdout.write("\n");
      break;
    case "reasoning-start":
      log(`\n${separator()}\n${yellow(bold("思考"))}\n${separator()}`);
      break;
    case "reasoning-delta":
      process.stdout.write(yellow(chunk.text));
      break;
    case "reasoning-end":
      log(separator());
      break;
    case "tool-call":
      log(`\n${separator()}\n ${cyan(bold("工具调用"))} ${cyan(chunk.toolName)}\n${separator()}`);
      log(`  ${cyan(JSON.stringify(chunk.input, null, 2))}`);
      break;
    case "tool-input-start":
      log(`\n${separator()}\n ${cyan(bold("工具输入"))} ${cyan(chunk.toolName)}`);
      break;
    case "tool-input-delta":
      process.stderr.write(cyan(chunk.delta));
      break;
    case "tool-input-end":
      log(separator());
      break;
    case "tool-result": {
      log(` ${green(bold("工具结果"))} ${green(chunk.toolName)}`);
      const output =
        typeof chunk.output === "string" ? chunk.output : JSON.stringify(chunk.output, null, 2);
      log(`  ${green(output)}`);
      break;
    }
    case "tool-error":
      log(` ${red(bold("工具错误"))} ${red(chunk.toolName)}`);
      log(`  ${red(String(chunk.error))}`);
      break;
    case "tool-output-denied":
      log(` ${yellow(bold("输出被拒绝"))} ${yellow(chunk.toolName)}`);
      break;
    case "tool-approval-request":
      log(` ${magenta(bold("等待审批"))} ${magenta(chunk.toolCall.toolName)}`);
      break;
    case "source":
      log(`${dim("[来源]")} ${dim(chunk.sourceType === "url" ? chunk.url : chunk.title)}`);
      break;
    case "file":
      log(`${blue(bold("文件"))} ${blue(chunk.file.mediaType)}`);
      break;
    case "start-step":
      log(`\n${separator()}\n${bold("步骤开始")}\n${separator()}`);
      break;
    case "finish-step":
      log(`${dim(`步骤结束 | ${chunk.finishReason}`)}`);
      break;
    case "start":
      log(`${bold("开始")}`);
      break;
    case "finish":
      log(formatFinish(chunk));
      break;
    case "abort":
      log(`${red(bold("中断"))}${chunk.reason ? `: ${chunk.reason}` : ""}`);
      break;
    case "error":
      log(`${red(bold("错误"))}: ${String(chunk.error)}`);
      break;
    case "raw":
      break;
  }
}

function formatFinish(chunk: Extract<TextStreamPart<any>, { type: "finish" }>): string {
  const u = chunk.totalUsage;
  const parts: string[] = [bold("完成")];
  if (chunk.finishReason) parts.push(`原因: ${chunk.finishReason}`);
  if (u) {
    const f = (n: number) =>
      n >= 1000000
        ? `${(n / 1000000).toFixed(1)}M`
        : n >= 1000
          ? `${(n / 1000).toFixed(1)}k`
          : String(n);
    parts.push(`↑${f(u.inputTokens ?? 0)}`);
    parts.push(`↓${f(u.outputTokens ?? 0)}`);
    parts.push(`∑${f(u.totalTokens ?? 0)}`);
  }
  return parts.join(" | ");
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
  filePath: string;
}

async function discoverSkills(): Promise<Skill[]> {
  const skills = new Map<string, Skill>();

  for (const dir of [HOME_SKILLS_DIR, LOCAL_SKILLS_DIR]) {
    try {
      await stat(dir);
    } catch {
      continue;
    }
    await discoverIn(dir, skills, true);
  }

  return [...skills.values()];
}

async function discoverIn(
  dir: string,
  skills: Map<string, Skill>,
  rootLevel: boolean,
): Promise<void> {
  let names: string[];
  try {
    names = await readdir(dir);
  } catch {
    return;
  }

  if (names.includes("SKILL.md")) {
    const s = await loadSkill(join(dir, "SKILL.md"));
    if (s) addSkill(skills, s);
    return;
  }

  for (const name of names.sort()) {
    if (name.startsWith(".") || name === "node_modules") continue;
    const fp = join(dir, name);
    let st;
    try {
      st = await stat(fp);
    } catch {
      continue;
    }

    if (st.isDirectory()) {
      await discoverIn(fp, skills, false);
    } else if (rootLevel && name.endsWith(".md")) {
      const s = await loadSkill(fp);
      if (s) addSkill(skills, s);
    }
  }
}

async function loadSkill(filePath: string): Promise<Skill | null> {
  let content: string;
  try {
    content = await readFile(filePath, "utf-8");
  } catch {
    return null;
  }

  const fm = parseFrontmatter(content);
  const name = fm.name || basename(dirname(filePath));

  if (!fm.description || !fm.description.trim()) {
    process.stderr.write(`skill ${filePath}: description 为空，跳过\n`);
    return null;
  }
  if (!/^[a-z0-9-]+$/.test(name)) {
    process.stderr.write(
      `skill ${filePath}: name "${name}" 格式不规范（建议小写字母、数字、连字符）\n`,
    );
  }

  return { name, description: fm.description, filePath };
}

function addSkill(skills: Map<string, Skill>, skill: Skill): void {
  const existing = skills.get(skill.name);
  if (existing) {
    process.stderr.write(`skill "${skill.name}" 碰撞: ${existing.filePath} <- ${skill.filePath}\n`);
  }
  skills.set(skill.name, skill);
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

      const filePath = skill.filePath;
      let content = await readFile(filePath, "utf-8");

      const dir = dirname(filePath);
      const extra = (await readdir(dir))
        .filter((f) => f.endsWith(".md") && f !== basename(filePath))
        .sort();
      for (const f of extra) {
        content += `\n\n--- ${f} ---\n\n${await readFile(join(dir, f), "utf-8")}`;
      }

      return content;
    },
  });
}

// === 视觉与输出辅助 ===

const noColor = !process.stdout.isTTY || !!process.env.NO_COLOR;
const style = (open: string, close: string) =>
  noColor ? (s: string) => s : (s: string) => `${open}${s}${close}`;
const dim = style("\x1b[2m", "\x1b[22m");
const cyan = style("\x1b[36m", "\x1b[39m");
const green = style("\x1b[32m", "\x1b[39m");
const yellow = style("\x1b[33m", "\x1b[39m");
const red = style("\x1b[31m", "\x1b[39m");
const blue = style("\x1b[34m", "\x1b[39m");
const magenta = style("\x1b[35m", "\x1b[39m");
const bold = noColor ? (s: string) => s : (s: string) => `\x1b[1m${s}\x1b[22m`;
const separator = () => dim("─".repeat(40));
const log = (msg: string) => process.stderr.write(msg + "\n");

main();
