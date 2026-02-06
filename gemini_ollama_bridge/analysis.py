from __future__ import annotations

import json
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Iterable

import requests

DEFAULT_EXCLUDES = {
    ".git",
    ".venv",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    ".pytest_cache",
    ".ruff_cache",
}

DEFAULT_INCLUDES = [
    "**/*.py",
    "**/*.js",
    "**/*.ts",
    "**/*.php",
    "**/*.md",
    "**/*.json",
    "**/*.yaml",
    "**/*.yml",
]


@dataclass(frozen=True)
class FileChunk:
    path: Path
    content: str


def _matches_any(path: Path, patterns: Iterable[str]) -> bool:
    path_str = path.as_posix()
    for pattern in patterns:
        if fnmatch(path_str, pattern):
            return True
        if pattern.startswith("**/") and fnmatch(path_str, pattern[3:]):
            return True
    return False


def _is_excluded(rel_path: Path, exclude_patterns: Iterable[str]) -> bool:
    if _matches_any(rel_path, exclude_patterns):
        return True
    return any(part in exclude_patterns for part in rel_path.parts)


def collect_files(
    base_path: Path,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    max_file_bytes: int = 120_000,
    max_total_bytes: int = 400_000,
) -> list[FileChunk]:
    include_patterns = include or DEFAULT_INCLUDES
    exclude_patterns = set(exclude or []) | DEFAULT_EXCLUDES

    collected: list[FileChunk] = []
    total_bytes = 0

    for path in base_path.rglob("*"):
        rel_path = path.relative_to(base_path)

        if _is_excluded(rel_path, exclude_patterns):
            continue

        if path.is_dir():
            continue

        if not _matches_any(rel_path, include_patterns):
            continue

        size = path.stat().st_size
        if size > max_file_bytes:
            continue

        if total_bytes + size > max_total_bytes:
            break

        content = path.read_text(encoding="utf-8", errors="ignore")
        collected.append(FileChunk(path=rel_path, content=content))
        total_bytes += size

    return collected


def build_prompt(files: list[FileChunk], focus: str | None) -> str:
    header = "You are an expert software reviewer. Analyze the following code."
    if focus:
        header += f" Focus on: {focus}."

    blocks = [header, "Provide findings with severity and suggested fixes."]
    for chunk in files:
        blocks.append(f"\nFile: {chunk.path}\n" + chunk.content)

    return "\n".join(blocks)


def ollama_generate(
    prompt: str,
    model: str,
    base_url: str,
    timeout: int = 120,
) -> str:
    url = base_url.rstrip("/") + "/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    if "response" not in data:
        raise ValueError(f"Unexpected Ollama response: {json.dumps(data)[:200]}")
    return data["response"]


def gemini_refine(
    analysis: str,
    command: str,
    args: list[str] | None = None,
    timeout: int = 120,
) -> str:
    cmd = shlex.split(command)
    if not cmd:
        raise ValueError("Gemini command is empty")

    executable = shutil.which(cmd[0])
    if not executable:
        raise FileNotFoundError(f"Gemini command not found: {cmd[0]}")

    final_cmd = [executable, *cmd[1:], *(args or [])]
    result = subprocess.run(
        final_cmd,
        input=analysis,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Gemini CLI failed with exit code "
            f"{result.returncode}: {result.stderr.strip()}"
        )

    if not result.stdout.strip():
        raise RuntimeError("Gemini CLI returned empty output")

    return result.stdout.strip()
