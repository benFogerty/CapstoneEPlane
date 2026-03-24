import { execFile } from "child_process";
import { existsSync } from "fs";
import path from "path";
import { promisify } from "util";

const execFileAsync = promisify(execFile);

export function resolveFrontendRoot() {
  const cwd = process.cwd();
  const frontendCwd = path.join(cwd, "frontend");
  if (existsSync(path.join(cwd, "scripts", "live_plane_data.py"))) {
    return cwd;
  }
  if (existsSync(path.join(frontendCwd, "scripts", "live_plane_data.py"))) {
    return frontendCwd;
  }
  throw new Error(`Unable to locate frontend scripts directory from cwd: ${cwd}`);
}

type PythonInvocation = {
  command: string;
  argsPrefix: string[];
};

function pythonInvocations(): PythonInvocation[] {
  const configured = process.env.PYTHON?.trim();
  const invocations: PythonInvocation[] = [];

  if (configured) {
    invocations.push({ command: configured, argsPrefix: [] });
  }

  const cwd = process.cwd();
  const unixCandidates = [
    path.join(cwd, ".venv", "bin", "python"),
    path.join(cwd, "..", ".venv", "bin", "python")
  ];
  const windowsCandidates = [
    path.join(cwd, ".venv", "Scripts", "python.exe"),
    path.join(cwd, "..", ".venv", "Scripts", "python.exe")
  ];

  for (const candidate of [...unixCandidates, ...windowsCandidates]) {
    if (existsSync(candidate)) {
      invocations.push({ command: candidate, argsPrefix: [] });
    }
  }

  if (process.platform === "win32") {
    invocations.push({ command: "python", argsPrefix: [] });
    invocations.push({ command: "py", argsPrefix: ["-3"] });
    invocations.push({ command: "python3", argsPrefix: [] });
    return invocations;
  }

  invocations.push({ command: "python3", argsPrefix: [] });
  invocations.push({ command: "python", argsPrefix: [] });
  return invocations;
}

export async function runPythonJson<T>(
  scriptName: string,
  args: string[]
): Promise<T> {
  const frontendRoot = resolveFrontendRoot();
  const scriptPath = path.join(frontendRoot, "scripts", scriptName);

  if (!existsSync(scriptPath)) {
    throw new Error(`Live data script not found: ${scriptPath}`);
  }

  let lastError: unknown;
  for (const invocation of pythonInvocations()) {
    try {
      const { stdout } = await execFileAsync(
        invocation.command,
        [...invocation.argsPrefix, scriptPath, ...args],
        {
          cwd: frontendRoot,
          maxBuffer: 8 * 1024 * 1024,
          windowsHide: true,
          env: {
            ...process.env,
            PYTHONUTF8: "1"
          }
        }
      );
      return JSON.parse(stdout) as T;
    } catch (error) {
      lastError = error;
    }
  }

  const error = lastError as NodeJS.ErrnoException & {
    stderr?: string;
    stdout?: string;
  };
  const stderr = error?.stderr?.trim();
  const stdout = error?.stdout?.trim();
  const details = [stderr, stdout].filter(Boolean).join(" | ");

  throw new Error(
    `Unable to execute ${scriptName} from ${frontendRoot} using available Python launchers.${details ? ` ${details}` : ""}`
  );
}
