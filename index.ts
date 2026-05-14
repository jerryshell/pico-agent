#!/usr/bin/env node
import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import { ToolLoopAgent, isLoopFinished, tool, zodSchema } from "ai";
import type { ImagePart, ModelMessage, TextStreamPart } from "ai";
import { exec } from "child_process";
import { mkdir, readdir, readFile, stat, writeFile } from "fs/promises";
import { homedir } from "os";
import { basename, dirname, join } from "path";
import { parseArgs, promisify } from "util";
import { z } from "zod";

const execAsync = promisify(exec);

const viewImageStore: Array<{ mimeType: string; data: string }> = [];
const MIME: Record<string, string> = {
  png: "image/png",
  jpg: "image/jpeg",
  jpeg: "image/jpeg",
  gif: "image/gif",
  webp: "image/webp",
};

const CONFIG_DIR = join(homedir(), ".pico-agent");
const CONFIG_FILE = join(CONFIG_DIR, "config.json");
const HOME_SKILLS_DIR = join(homedir(), ".agents", "skills");
const LOCAL_SKILLS_DIR = join(process.cwd(), ".agents", "skills");

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

async function resolveApiKey(): Promise<string | undefined> {
  const envKey = process.env.API_KEY;
  if (envKey) return envKey;

  const fileConfig = await loadConfig();
  return fileConfig.apiKey;
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

// === agent ===

interface Skill {
  name: string;
  description: string;
  filePath: string;
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
    prepareStep: async ({ messages }) => {
      if (viewImageStore.length === 0) return;
      const images = viewImageStore.splice(0);
      return {
        messages: [
          ...messages,
          {
            role: "user",
            content: images.map((img) => ({
              type: "image" as const,
              image: img.data,
              mediaType: img.mimeType,
            })),
          },
        ],
      };
    },
  });

  const { text, images } = await extractImagesFromPrompt(prompt);
  const messages: Array<ModelMessage> =
    images.length > 0
      ? [{ role: "user", content: [{ type: "text" as const, text }, ...images] }]
      : [{ role: "user", content: text }];

  const result = await agent.stream({
    messages,
  } as any);

  for await (const chunk of result.fullStream) {
    printChunk(chunk);
  }
}

async function extractImagesFromPrompt(
  raw: string,
): Promise<{ text: string; images: ImagePart[] }> {
  const IMAGE_RE = /([\w.\/\\-]+\.(?:png|jpg|jpeg|gif|webp))/gi;
  const cwd = process.cwd();
  const seen = new Set<string>();
  let resultText = raw;
  const resultImages: ImagePart[] = [];

  for (const m of raw.matchAll(IMAGE_RE)) {
    const name = m[1]!;
    if (seen.has(name)) continue;
    const resolved = join(cwd, name);
    try {
      await stat(resolved);
    } catch {
      continue;
    }
    seen.add(name);
    try {
      const buf = await readFile(resolved);
      const ext = name.toLowerCase().split(".").pop()!;
      resultImages.push({ type: "image", image: buf.toString("base64"), mediaType: MIME[ext] });
    } catch {
      seen.delete(name);
    }
  }

  for (const name of seen) {
    resultText = resultText.replaceAll(name, "");
  }
  resultText = resultText.trim();

  return { text: resultText, images: resultImages };
}

// === tools ===

function createTools() {
  const read = tool({
    description: "读取文件内容。超过 50KB 自动截断并提示剩余大小。",
    inputSchema: zodSchema(z.object({ path: z.string().describe("文件路径") })),
    execute: async ({ path }: { path: string }) => {
      try {
        const content = await readFile(path, "utf-8");
        const total = Buffer.byteLength(content, "utf-8");
        const MAX = 50 * 1024;
        if (total <= MAX) return content;

        const lines = content.split("\n");
        let buf = 0;
        const take: string[] = [];
        for (const line of lines) {
          const add = Buffer.byteLength(line + "\n", "utf-8");
          if (buf + add > MAX) break;
          take.push(line);
          buf += add;
        }
        return `${take.join("\n")}\n\n[输出截断：显示 ${(buf / 1024).toFixed(0)}KB / 共 ${(total / 1024).toFixed(0)}KB，剩余约 ${lines.length - take.length} 行]`;
      } catch (error) {
        throw new Error(`无法读取文件 ${path}: ${error}`);
      }
    },
  });

  const write = tool({
    description: "创建或重写文件。自动创建父目录。",
    inputSchema: zodSchema(
      z.object({
        path: z.string().describe("文件路径"),
        content: z.string().describe("文件内容"),
      }),
    ),
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

  const edit = tool({
    description: "编辑文件（支持单次多处编辑）。每项 edits[].oldText 必须在文件中唯一。",
    inputSchema: zodSchema(
      z.object({
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
    ),
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
          const first = content.indexOf(oldText);
          if (first === -1) throw new Error(`文件 ${path} 中未找到 "${oldText}"`);
          if (content.indexOf(oldText, first + 1) !== -1)
            throw new Error(`"${oldText}" 在文件中出现多次`);
          content = content.replace(oldText, newText);
        }
        await writeFile(path, content, "utf-8");
        return `文件 ${path} 已更新 (${edits.length} 处编辑)`;
      } catch (error) {
        throw new Error(`编辑文件失败: ${error}`);
      }
    },
  });

  const bash = tool({
    description: "运行 shell 命令。可设 timeout（秒）防挂起。",
    inputSchema: zodSchema(
      z.object({
        command: z.string().describe("要执行的 shell 命令"),
        timeout: z.number().optional().describe("超时秒数"),
      }),
    ),
    execute: async ({ command, timeout }: { command: string; timeout?: number }) => {
      try {
        const opts = timeout ? { timeout: timeout * 1000 } : undefined;
        const { stdout, stderr } = await execAsync(command, opts);
        return String(stdout) + String(stderr);
      } catch (error: any) {
        const partial = String(error.stdout || "") + String(error.stderr || "");
        const suffix = partial ? "\n\n[以上为超时前的部分输出]" : "";
        return `${partial}命令执行失败: ${error.message}${suffix}`;
      }
    },
  });

  const view = tool({
    description: "查看图片文件（支持 png/jpg/gif/webp）。用户提及图片时调用此工具。",
    inputSchema: zodSchema(z.object({ path: z.string().describe("图片文件路径") })),
    execute: async ({ path }: { path: string }) => {
      const ext = path.toLowerCase().split(".").pop()!;
      if (!MIME[ext]) throw new Error(`不支持的图片格式: .${ext}`);
      const buf = await readFile(path);
      viewImageStore.push({ mimeType: MIME[ext], data: buf.toString("base64") });
      return `已读取图片 ${basename(path)}`;
    },
  });

  return { read, write, edit, bash, view };
}

function createSkillTool(skills: Skill[]) {
  const index = new Map(skills.map((s) => [s.name, s]));

  return tool({
    description: "加载指定技能的详细内容",
    inputSchema: zodSchema(z.object({ name: z.string().describe("技能名称") })),
    execute: async ({ name }: { name: string }) => {
      const skill = index.get(name);
      if (!skill) throw new Error(`未知技能 "${name}"，可用: ${[...index.keys()].join(", ")}`);

      const dir = dirname(skill.filePath);
      let content = await readFile(skill.filePath, "utf-8");

      const extra = (await readdir(dir))
        .filter((f) => f.endsWith(".md") && f !== basename(skill.filePath))
        .sort();
      for (const f of extra) {
        content += `\n\n--- ${f} ---\n\n${await readFile(join(dir, f), "utf-8")}`;
      }

      return content;
    },
  });
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

// === output ===

const rule = () => log(dim("─".repeat(40)));
const open = (title: string) => {
  rule();
  log(title);
};

function printChunk(chunk: TextStreamPart<any>) {
  switch (chunk.type) {
    // ── text output (stdout) ──
    case "text-delta":
      process.stdout.write(chunk.text);
      break;
    case "text-start":
    case "text-end":
      process.stdout.write("\n");
      break;

    // ── reasoning (stdout) ──
    case "reasoning-start":
      open(yellow(bold("思考")));
      break;
    case "reasoning-delta":
      process.stdout.write(yellow(chunk.text));
      break;
    case "reasoning-end":
      process.stdout.write("\n");
      break;

    // ── tool interaction (stderr) ──
    case "tool-call":
      open(`${cyan(bold("工具调用"))} ${cyan(chunk.toolName)}`);
      log(`  ${cyan(JSON.stringify(chunk.input))}`);
      break;
    case "tool-result": {
      log(` ${green(bold("工具结果"))} ${green(chunk.toolName)}`);
      const output =
        typeof chunk.output === "string" ? chunk.output : JSON.stringify(chunk.output);
      for (const line of output.split("\n")) {
        log(`  ${green(line)}`);
      }
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

    // ── step lifecycle (stderr) ──
    case "start-step":
      open(bold("步骤开始"));
      break;
    case "finish-step":
      log(dim(`步骤结束 | ${chunk.finishReason}`));
      break;

    // ── session lifecycle (stderr) ──
    case "start":
      log(bold("开始"));
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

    // ── metadata (stderr) ──
    case "source":
      log(`${dim("[来源]")} ${dim(chunk.sourceType === "url" ? chunk.url : chunk.title)}`);
      break;
    case "file":
      log(`${blue(bold("文件"))} ${blue(chunk.file.mediaType)}`);
      break;

    // ── skip (tool-input, raw, etc) ──
    default:
      break;
  }
}

function formatFinish(chunk: Extract<TextStreamPart<any>, { type: "finish" }>): string {
  const usage = chunk.totalUsage;
  const parts: string[] = [bold("完成")];
  if (chunk.finishReason) parts.push(`原因: ${chunk.finishReason}`);
  if (usage) {
    const fmt = (n: number) =>
      n >= 1000000
        ? `${(n / 1000000).toFixed(1)}M`
        : n >= 1000
          ? `${(n / 1000).toFixed(1)}k`
          : String(n);
    parts.push(`↑${fmt(usage.inputTokens ?? 0)}`);
    parts.push(`↓${fmt(usage.outputTokens ?? 0)}`);
    parts.push(`∑${fmt(usage.totalTokens ?? 0)}`);
  }
  return dim(parts.join(" | "));
}

// === skills ===

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
    const skill = await loadSkill(join(dir, "SKILL.md"));
    if (skill) addSkill(skills, skill);
    return;
  }

  for (const name of names.sort()) {
    if (name.startsWith(".") || name === "node_modules") continue;
    const fullPath = join(dir, name);
    let stats;
    try {
      stats = await stat(fullPath);
    } catch {
      continue;
    }

    if (stats.isDirectory()) {
      await discoverIn(fullPath, skills, false);
    } else if (rootLevel && name.endsWith(".md")) {
      const skill = await loadSkill(fullPath);
      if (skill) addSkill(skills, skill);
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

  const frontmatter = parseFrontmatter(content);
  const name = frontmatter.name || basename(dirname(filePath));

  if (!frontmatter.description || !frontmatter.description.trim()) {
    process.stderr.write(`skill ${filePath}: description 为空，跳过\n`);
    return null;
  }
  if (!/^[a-z0-9-]+$/.test(name)) {
    process.stderr.write(
      `skill ${filePath}: name "${name}" 格式不规范（建议小写字母、数字、连字符）\n`,
    );
  }

  return { name, description: frontmatter.description, filePath };
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

// === styling ===

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

main();
