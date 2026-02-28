from __future__ import annotations

import hashlib
import json
import os
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

REFINE_SYSTEM_PROMPT = (
    "You are a senior software engineer. You receive a code analysis report "
    "produced by a local LLM. Your job is to refine it: remove false positives, "
    "improve clarity, add actionable suggestions, and re-rank findings by severity. "
    "Return your refined analysis as Markdown."
)


# ---------------------------------------------------------------------------
# File collection
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


def build_prompt(files: list[FileChunk], focus: str | None) -> str:
    header = "You are an expert software reviewer. Analyze the following code."
    if focus:
        header += f" Focus on: {focus}."

    blocks = [header, "Provide findings with severity and suggested fixes."]
    for chunk in files:
        blocks.append(f"\nFile: {chunk.path}\n" + chunk.content)

    return "\n".join(blocks)


# ---------------------------------------------------------------------------
# Result caching
# ---------------------------------------------------------------------------


class AnalysisCache:
    """Simple disk cache keyed by SHA-256 of the prompt string."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._cache_dir = cache_dir or Path(".ollama_cache")

    @staticmethod
    def _hash(prompt: str) -> str:
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    def get(self, prompt: str) -> str | None:
        cache_file = self._cache_dir / (self._hash(prompt) + ".json")
        if not cache_file.exists():
            return None
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            return data.get("response")
        except (json.JSONDecodeError, OSError):
            return None

    def put(self, prompt: str, response: str) -> None:
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self._cache_dir / (self._hash(prompt) + ".json")
        cache_file.write_text(
            json.dumps({"prompt_hash": self._hash(prompt), "response": response}),
            encoding="utf-8",
        )


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------


def ollama_generate(
    prompt: str,
    model: str,
    base_url: str,
    timeout: int = 120,
    cache: AnalysisCache | None = None,
) -> str:
    if cache is not None:
        cached = cache.get(prompt)
        if cached is not None:
            return cached

    url = base_url.rstrip("/") + "/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    if "response" not in data:
        raise ValueError(f"Unexpected Ollama response: {json.dumps(data)[:200]}")

    result = data["response"]

    if cache is not None:
        cache.put(prompt, result)

    return result


# ---------------------------------------------------------------------------
# Gemini (google-generativeai SDK)
# ---------------------------------------------------------------------------


class GeminiProvider:
    """Refine analysis results using the Google Generative AI Python SDK.

    Requires the ``GEMINI_API_KEY`` environment variable (or pass *api_key*
    explicitly).
    """

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        api_key: str | None = None,
        timeout: int = 120,
    ) -> None:
        resolved_key = api_key or os.getenv("GEMINI_API_KEY", "")
        if not resolved_key:
            raise ValueError(
                "GEMINI_API_KEY not set. Export it or pass --gemini-api-key."
            )

        # Import here so the dependency is optional at module-load time.
        import google.generativeai as genai  # noqa: WPS433

        genai.configure(api_key=resolved_key)
        self._model = genai.GenerativeModel(model)
        self._timeout = timeout

    def refine(self, analysis: str) -> str:
        """Send *analysis* to Gemini for refinement and return the result."""
        prompt = f"{REFINE_SYSTEM_PROMPT}\n\n---\n\n{analysis}"
        response = self._model.generate_content(
            prompt,
            request_options={"timeout": self._timeout},
        )
        text = response.text
        if not text or not text.strip():
            raise RuntimeError("Gemini returned empty output")
        return text.strip()


def gemini_refine(
    analysis: str,
    model: str = "gemini-2.0-flash",
    api_key: str | None = None,
    timeout: int = 120,
) -> str:
    """Convenience wrapper around :class:`GeminiProvider`."""
    provider = GeminiProvider(model=model, api_key=api_key, timeout=timeout)
    return provider.refine(analysis)
