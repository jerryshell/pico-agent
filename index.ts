import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import { ToolLoopAgent, tool, zodSchema } from "ai";
import { exec } from "child_process";
import { readFile, writeFile } from "fs/promises";
import { parseArgs, promisify } from "util";
import { z } from "zod";

const execAsync = promisify(exec);

const dim = (s: string) => `\x1b[2m${s}\x1b[22m`;
const cyan = (s: string) => `\x1b[36m${s}\x1b[39m`;
const green = (s: string) => `\x1b[32m${s}\x1b[39m`;
const yellow = (s: string) => `\x1b[33m${s}\x1b[39m`;
const red = (s: string) => `\x1b[31m${s}\x1b[39m`;
const blue = (s: string) => `\x1b[34m${s}\x1b[39m`;
const magenta = (s: string) => `\x1b[35m${s}\x1b[39m`;
const bold = (s: string) => `\x1b[1m${s}\x1b[22m`;
const separator = () => dim("─".repeat(40));

async function main() {
  const { apiKey, prompt } = getConfig();

  const provider = createOpenAICompatible({
    name: "Kimi For Coding",
    apiKey,
    baseURL: "https://api.kimi.com/coding/v1",
  });

  const model = provider("kimi-for-coding");

  const read = tool({
    description: "读取文件内容",
    inputSchema: zodSchema(
      z.object({
        path: z.string().describe("文件路径"),
      }),
    ),
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
    inputSchema: zodSchema(
      z.object({
        command: z.string().describe("要执行的 shell 命令"),
      }),
    ),
    execute: async ({ command }: { command: string }) => {
      try {
        const { stdout, stderr } = await execAsync(command);
        return stdout + stderr;
      } catch (error: any) {
        return `命令执行失败: ${error.message}`;
      }
    },
  });

  const agent = new ToolLoopAgent({
    model,
    headers: { "User-Agent": "KimiCLI/1.5" },
    tools: { read, write, edit, bash },
  });

  const result = await agent.stream({ prompt });

  for await (const chunk of result.fullStream) {
    switch (chunk.type) {
      case "text-delta":
        process.stdout.write(chunk.text);
        break;
      case "text-start":
      case "text-end":
        process.stdout.write("\n");
        break;
      case "reasoning-start":
        process.stdout.write(`\n${separator()}\n${yellow(bold("🤔 思考"))}\n${separator()}\n`);
        break;
      case "reasoning-delta":
        process.stdout.write(yellow(chunk.text));
        break;
      case "reasoning-end":
        process.stdout.write(`\n${separator()}\n`);
        break;
      case "tool-call":
        process.stdout.write(
          `\n${separator()}\n${cyan(bold("🔧 工具调用"))} ${cyan(chunk.toolName)}\n${separator()}\n`,
        );
        process.stdout.write(cyan(JSON.stringify(chunk.input, null, 2)) + "\n");
        break;
      case "tool-input-start":
        process.stdout.write(
          `\n${separator()}\n${cyan(bold("📥 工具输入"))} ${cyan(chunk.toolName)}\n`,
        );
        break;
      case "tool-input-delta":
        process.stdout.write(cyan(chunk.delta));
        break;
      case "tool-input-end":
        process.stdout.write(`\n${separator()}\n`);
        break;
      case "tool-result":
        process.stdout.write(`${green(bold("✅ 工具结果"))} ${green(chunk.toolName)}\n`);
        const output =
          typeof chunk.output === "string" ? chunk.output : JSON.stringify(chunk.output, null, 2);
        process.stdout.write(green(output) + "\n");
        break;
      case "tool-error":
        process.stdout.write(`${red(bold("❌ 工具错误"))} ${red(chunk.toolName)}\n`);
        process.stdout.write(red(String(chunk.error)) + "\n");
        break;
      case "tool-output-denied":
        process.stdout.write(`${yellow(bold("⛔ 输出被拒绝"))} ${yellow(chunk.toolName)}\n`);
        break;
      case "tool-approval-request":
        process.stdout.write(
          `${magenta(bold("⏳ 等待审批"))} ${magenta(chunk.toolCall.toolName)}\n`,
        );
        break;
      case "source":
        process.stdout.write(
          `${dim("[来源]")} ${dim(chunk.sourceType === "url" ? chunk.url : chunk.title)}\n`,
        );
        break;
      case "file":
        process.stdout.write(`${blue(bold("📄 文件"))} ${blue(chunk.file.mediaType)}\n`);
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
}

function getConfig() {
  const { values, positionals } = parseArgs({
    options: { "api-key": { type: "string" } },
    strict: false,
    allowPositionals: true,
  });

  if (positionals.length === 0) {
    throw new Error("请提供要询问的内容");
  }

  return {
    apiKey: getApiKey(values),
    prompt: positionals.join(" "),
  };
}

function getApiKey(args: Record<string, unknown>): string {
  const cliKey = args["api-key"] as string | undefined;
  if (cliKey) return cliKey;

  const envKey = process.env.API_KEY;
  if (envKey) return envKey;

  throw new Error("请使用 --api-key 参数或设置 API_KEY 环境变量");
}

main();
