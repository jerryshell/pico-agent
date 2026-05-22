#!/usr/bin/env node
import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import { ToolLoopAgent, isLoopFinished, tool } from "ai";
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

const noColor = !process.stdout.isTTY || !!process.env.NO_COLOR;
const style = (open: string, close: string) =>
  noColor ? (text: string) => text : (text: string) => `${open}${text}${close}`;
const dim = style("\x1b[2m", "\x1b[22m");
const cyan = style("\x1b[36m", "\x1b[39m");
const green = style("\x1b[32m", "\x1b[39m");
const yellow = style("\x1b[33m", "\x1b[39m");
const red = style("\x1b[31m", "\x1b[39m");
const blue = style("\x1b[34m", "\x1b[39m");
const magenta = style("\x1b[35m", "\x1b[39m");
const bold = noColor ? (s: string) => s : (s: string) => `\x1b[1m${s}\x1b[22m`;
const log = (msg: string) => process.stderr.write(msg + "\n");

// === entry ===

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

function parseConfigCommand(positionals: string[]) {
  if (positionals[0] !== "config") return null;
  const [, key, value] = positionals;
  if (key === "apiKey" && value) {
    return { mode: "config" as const, key, value };
  }
  console.error("用法: pa config apiKey yourKimiCodeApiKey");
  process.exit(1);
}

function parseCommand() {
  const { positionals } = parseArgs({
    strict: false,
    allowPositionals: true,
  });

  const config = parseConfigCommand(positionals);
  if (config) return config;

  if (positionals.length === 0) {
    console.error("用法: pa <prompt>");
    process.exit(1);
  }

  return {
    mode: "run" as const,
    prompt: positionals.join(" "),
  };
}

// === config ===

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
  await writeFile(CONFIG_FILE, JSON.stringify(config) + "\n", "utf-8");
}

async function resolveApiKey(): Promise<string | undefined> {
  const envKey = process.env.API_KEY;
  if (envKey) return envKey;

  const fileConfig = await loadConfig();
  return fileConfig.apiKey;
}

// === skills discovery ===

interface Skill {
  name: string;
  description: string;
  filePath: string;
}

function parseFrontmatter(content: string): { name?: string; description?: string } {
  const match = content.match(/^---\n([\s\S]*?)---\n/);
  if (!match) return {};

  const result: { name?: string; description?: string } = {};
  const lines = match[1]!.split("\n");

  for (let i = 0; i < lines.length; i++) {
    const header = lines[i]!.match(/^(\w+):\s*(.*)$/);
    if (!header) continue;

    const key = header[1]!;
    let value = header[2]!;

    if (value === ">" || value === "|") {
      const blockLines: string[] = [];
      i++;
      while (i < lines.length && (lines[i]!.startsWith(" ") || lines[i]!.startsWith("\t"))) {
        blockLines.push(lines[i]!.trim());
        i++;
      }
      i--;
      value =
        value === "|"
          ? blockLines.join("\n").trim()
          : blockLines.join(" ").replace(/\s+/g, " ").trim();
    }

    if (key === "name") result.name = value;
    else if (key === "description") result.description = value;
  }

  return result;
}

function addSkill(skills: Map<string, Skill>, skill: Skill): void {
  const existing = skills.get(skill.name);
  if (existing) {
    process.stderr.write(`skill "${skill.name}" 碰撞: ${existing.filePath} <- ${skill.filePath}\n`);
  }
  skills.set(skill.name, skill);
}

function validateSkill(name: string, description: string | undefined, filePath: string): boolean {
  if (!description || !description.trim()) {
    process.stderr.write(`skill ${filePath}: description 为空，跳过\n`);
    return false;
  }
  if (!/^[a-z0-9-]+$/.test(name)) {
    process.stderr.write(
      `skill ${filePath}: name "${name}" 格式不规范（建议小写字母、数字、连字符）\n`,
    );
  }
  return true;
}

async function loadSkill(filePath: string): Promise<Skill | null> {
  let content: string;
  try {
    content = await readFile(filePath, "utf-8");
  } catch {
    return null;
  }

  const frontmatter = parseFrontmatter(content);
  const name = frontmatter.name || basename(dirname(filePath));

  if (!validateSkill(name, frontmatter.description, filePath)) return null;

  return { name, description: frontmatter.description!, filePath };
}

async function safeReadDir(dir: string): Promise<string[] | null> {
  try {
    return await readdir(dir);
  } catch {
    return null;
  }
}

async function tryAddSkill(filePath: string, skills: Map<string, Skill>): Promise<void> {
  const skill = await loadSkill(filePath);
  if (skill) addSkill(skills, skill);
}

async function safeStat(path: string) {
  try {
    return await stat(path);
  } catch {
    return null;
  }
}

function isVisibleEntry(entry: string): boolean {
  return !entry.startsWith(".") && entry !== "node_modules";
}

function shouldLoadSkill(entry: string, rootLevel: boolean): boolean {
  return rootLevel && entry.endsWith(".md");
}

async function processDirectoryEntry(
  fullPath: string,
  entry: string,
  skills: Map<string, Skill>,
  rootLevel: boolean,
): Promise<void> {
  const entryStats = await safeStat(fullPath);
  if (!entryStats) return;

  if (entryStats.isDirectory()) {
    await discoverIn(fullPath, skills, false);
    return;
  }

  if (shouldLoadSkill(entry, rootLevel)) {
    await tryAddSkill(fullPath, skills);
  }
}

async function discoverIn(
  dir: string,
  skills: Map<string, Skill>,
  rootLevel: boolean,
): Promise<void> {
  const entries = await safeReadDir(dir);
  if (!entries) return;

  if (entries.includes("SKILL.md")) {
    await tryAddSkill(join(dir, "SKILL.md"), skills);
    return;
  }

  const visible = entries.filter(isVisibleEntry).sort();
  for (const entry of visible) {
    await processDirectoryEntry(join(dir, entry), entry, skills, rootLevel);
  }
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

// === tools ===

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

async function buildSkillContent(skill: Skill): Promise<string> {
  const dir = dirname(skill.filePath);
  let content = await readFile(skill.filePath, "utf-8");

  const extraFiles = (await readdir(dir))
    .filter((file) => file.endsWith(".md") && file !== basename(skill.filePath))
    .sort();
  for (const file of extraFiles) {
    content += `\n\n--- ${file} ---\n\n${await readFile(join(dir, file), "utf-8")}`;
  }

  return content;
}

function createSkillTool(skills: Skill[]) {
  const index = new Map(skills.map((s) => [s.name, s]));

  return tool({
    description: "加载指定技能的详细内容",
    inputSchema: z.object({ name: z.string().describe("技能名称") }),
    execute: async ({ name }: { name: string }) => {
      const skill = index.get(name);
      if (!skill) throw new Error(`未知技能 "${name}"，可用: ${[...index.keys()].join(", ")}`);

      return buildSkillContent(skill);
    },
  });
}

// tool definitions

function truncateContent(content: string, maxBytes: number): string {
  const total = Buffer.byteLength(content, "utf-8");
  if (total <= maxBytes) return content;

  const lines = content.split("\n");
  let bytes = 0;
  const keptLines: string[] = [];
  for (const line of lines) {
    const lineBytes = Buffer.byteLength(line + "\n", "utf-8");
    if (bytes + lineBytes > maxBytes) break;
    keptLines.push(line);
    bytes += lineBytes;
  }
  return `${keptLines.join("\n")}\n\n[输出截断：显示 ${(bytes / 1024).toFixed(0)}KB / 共 ${(total / 1024).toFixed(0)}KB，剩余约 ${lines.length - keptLines.length} 行]`;
}

const read = tool({
  description: "读取文件内容。超过 50KB 自动截断并提示剩余大小。",
  inputSchema: z.object({ path: z.string().describe("文件路径") }),
  execute: async ({ path }: { path: string }) => {
    try {
      const content = await readFile(path, "utf-8");
      return truncateContent(content, 50 * 1024);
    } catch (error) {
      throw new Error(`无法读取文件 ${path}: ${error}`);
    }
  },
});

const write = tool({
  description: "创建或重写文件。自动创建父目录。",
  inputSchema: z.object({
    path: z.string().describe("文件路径"),
    content: z.string().describe("文件内容"),
  }),
  execute: async ({ path, content }: { path: string; content: string }) => {
    try {
      await mkdir(dirname(path), { recursive: true });
      await writeFile(path, content, "utf-8");
      return `文件 ${path} 已写入（${Buffer.byteLength(content, "utf-8")} 字节）`;
    } catch (error) {
      throw new Error(`无法写入文件 ${path}: ${error}`);
    }
  },
});

function applyEdit(content: string, oldText: string, newText: string, path: string): string {
  const first = content.indexOf(oldText);
  if (first === -1) throw new Error(`文件 ${path} 中未找到 "${oldText}"`);
  if (content.indexOf(oldText, first + 1) !== -1) throw new Error(`"${oldText}" 在文件中出现多次`);
  return content.replace(oldText, newText);
}

const edit = tool({
  description: "编辑文件（支持单次多处编辑）。每项 edits[].oldText 必须在文件中唯一。",
  inputSchema: z.object({
    path: z.string().describe("文件路径"),
    edits: z
      .array(
        z.object({
          oldText: z.string().describe("原始字符串（须在文件中唯一）"),
          newText: z.string().describe("替换后的字符串"),
        }),
      )
      .describe("替换列表（按顺序逐项应用）"),
  }),
  execute: async ({
    path,
    edits,
  }: {
    path: string;
    edits: { oldText: string; newText: string }[];
  }) => {
    try {
      let content = await readFile(path, "utf-8");
      for (const { oldText, newText } of edits) {
        content = applyEdit(content, oldText, newText, path);
      }
      await writeFile(path, content, "utf-8");
      return `文件 ${path} 已更新 (${edits.length} 处编辑)`;
    } catch (error) {
      throw new Error(`编辑文件失败: ${error}`);
    }
  },
});

function formatExecError(error: any): string {
  const partial = String(error.stdout || "") + String(error.stderr || "");
  const suffix = partial ? "\n\n[以上为超时前的部分输出]" : "";
  return `${partial}命令执行失败: ${error.message}${suffix}`;
}

const bash = tool({
  description: "运行 shell 命令。可设 timeout（秒）防挂起。",
  inputSchema: z.object({
    command: z.string().describe("要执行的 shell 命令"),
    timeout: z.number().optional().describe("超时秒数"),
  }),
  execute: async ({ command, timeout }: { command: string; timeout?: number }) => {
    try {
      const options = timeout ? { timeout: timeout * 1000 } : undefined;
      const { stdout, stderr } = await execAsync(command, options);
      return String(stdout) + String(stderr);
    } catch (error: any) {
      return formatExecError(error);
    }
  },
});

const view = tool({
  description: "指定文件路径，将在下一步发送给模型查看",
  inputSchema: z.object({ path: z.string().describe("文件路径") }),
  execute: async ({ path }: { path: string }) => {
    return path;
  },
});

function createTools() {
  return { read, write, edit, bash, view };
}

// === output ===

const logRule = () => log(dim("─".repeat(40)));
function logSection(title: string) {
  logRule();
  log(title);
}

function formatTokenCount(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`;
  return String(n);
}

function formatFinish(chunk: Extract<TextStreamPart<any>, { type: "finish" }>): string {
  const usage = chunk.totalUsage;
  const segments: string[] = [bold("完成")];
  if (chunk.finishReason) segments.push(`原因: ${chunk.finishReason}`);
  if (usage) {
    const { inputTokens = 0, outputTokens = 0, totalTokens = 0 } = usage;
    segments.push(`↑${formatTokenCount(inputTokens)}`);
    segments.push(`↓${formatTokenCount(outputTokens)}`);
    segments.push(`∑${formatTokenCount(totalTokens)}`);
  }
  return dim(segments.join(" | "));
}

const chunkHandlers: Record<string, (chunk: any) => void> = {
  "text-delta": (c) => process.stdout.write(c.text),
  "text-start": () => process.stdout.write("\n"),
  "text-end": () => process.stdout.write("\n"),
  "reasoning-start": () => logSection(yellow(bold("思考"))),
  "reasoning-delta": (c) => process.stdout.write(yellow(c.text)),
  "reasoning-end": () => process.stdout.write("\n"),
  "tool-call": (c) => {
    logSection(`${cyan(bold("工具调用"))} ${cyan(c.toolName)}`);
    log(cyan(JSON.stringify(c.input)));
  },
  "tool-result": (c) => {
    log(`${green(bold("工具结果"))} ${green(c.toolName)}`);
    const output = typeof c.output === "string" ? c.output : JSON.stringify(c.output);
    for (const line of output.split("\n")) {
      log(green(line));
    }
  },
  "tool-error": (c) => {
    log(`${red(bold("工具错误"))} ${red(c.toolName)}`);
    log(red(String(c.error)));
  },
  "tool-output-denied": (c) => log(`${yellow(bold("输出被拒绝"))} ${yellow(c.toolName)}`),
  "tool-approval-request": (c) =>
    log(`${magenta(bold("等待审批"))} ${magenta(c.toolCall.toolName)}`),
  "start-step": () => logSection(bold("步骤开始")),
  "finish-step": (c) => log(dim(`步骤结束 | ${c.finishReason}`)),
  start: () => log(bold("开始")),
  finish: (c) => log(formatFinish(c)),
  abort: (c) => log(`${red(bold("中断"))}${c.reason ? `: ${c.reason}` : ""}`),
  error: (c) => log(`${red(bold("错误"))}: ${String(c.error)}`),
  source: (c) => log(`${dim("[来源]")} ${dim(c.sourceType === "url" ? c.url : c.title)}`),
  file: (c) => log(`${blue(bold("文件"))} ${blue(c.file.mediaType)}`),
};

function printChunk(chunk: TextStreamPart<any>) {
  const handler = chunkHandlers[chunk.type];
  if (handler) handler(chunk);
}

// === agent ===

async function collectViewImages(step: any): Promise<Buffer[]> {
  const results = step.toolResults.filter(
    (r: any) => r.toolName === "view" && typeof r.output === "string",
  );
  const images: Buffer[] = [];
  for (const r of results) {
    try {
      images.push(await readFile(r.output));
    } catch {}
  }
  return images;
}

async function runAgent(apiKey: string, prompt: string) {
  const provider = createOpenAICompatible({
    name: "Kimi Code",
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
    prepareStep: async ({ steps, messages }) => {
      const lastStep = steps.at(-1);
      if (!lastStep) return;
      const images = await collectViewImages(lastStep);
      if (images.length === 0) return;
      return {
        messages: [
          ...messages,
          { role: "user", content: images.map((buf) => ({ type: "image" as const, image: buf })) },
        ],
      };
    },
  });

  const messages = [{ role: "user" as const, content: prompt }];

  const result = await agent.stream({ messages });

  for await (const chunk of result.fullStream) {
    printChunk(chunk);
  }
}

main();
