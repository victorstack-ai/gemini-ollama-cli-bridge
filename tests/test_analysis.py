from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gemini_ollama_bridge.analysis import (
    AnalysisCache,
    FileChunk,
    GeminiProvider,
    build_prompt,
    collect_files,
    gemini_refine,
    ollama_generate,
)


# ---------------------------------------------------------------------------
# File collection
# ---------------------------------------------------------------------------


def test_collect_files_respects_include_exclude(tmp_path: Path) -> None:
    (tmp_path / "keep.py").write_text("print('ok')")
    (tmp_path / "skip.txt").write_text("nope")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "bad.js").write_text("alert(1)")

    files = collect_files(tmp_path, include=["**/*.py", "**/*.txt"], exclude=["*.txt"])
    assert [f.path.name for f in files] == ["keep.py"]


def test_collect_files_skips_large_files(tmp_path: Path) -> None:
    small = tmp_path / "small.py"
    small.write_text("x = 1")
    big = tmp_path / "big.py"
    big.write_text("x" * 200_000)

    files = collect_files(tmp_path, include=["**/*.py"], max_file_bytes=1_000)
    names = [f.path.name for f in files]
    assert "small.py" in names
    assert "big.py" not in names


def test_collect_files_respects_max_total_bytes(tmp_path: Path) -> None:
    for i in range(10):
        (tmp_path / f"file{i}.py").write_text("a" * 100)

    files = collect_files(tmp_path, include=["**/*.py"], max_total_bytes=350)
    # Each file is ~100 bytes, so only 3 should fit within 350 bytes
    assert len(files) <= 4


def test_collect_files_returns_empty_for_no_matches(tmp_path: Path) -> None:
    (tmp_path / "readme.txt").write_text("hello")
    files = collect_files(tmp_path, include=["**/*.rs"])
    assert files == []


def test_collect_files_excludes_default_dirs(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config.py").write_text("git stuff")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "mod.py").write_text("cached")
    (tmp_path / "real.py").write_text("print(1)")

    files = collect_files(tmp_path, include=["**/*.py"])
    names = [f.path.name for f in files]
    assert "real.py" in names
    assert "config.py" not in names
    assert "mod.py" not in names


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


def test_build_prompt_contains_focus_and_files(tmp_path: Path) -> None:
    file_path = tmp_path / "demo.py"
    file_path.write_text("x = 1")

    files = collect_files(tmp_path, include=["**/*.py"])
    prompt = build_prompt(files, focus="bugs, security")

    assert "Focus on: bugs, security" in prompt
    assert "File: demo.py" in prompt
    assert "x = 1" in prompt


def test_build_prompt_without_focus() -> None:
    chunks = [FileChunk(path=Path("a.py"), content="pass")]
    prompt = build_prompt(chunks, focus=None)
    assert "Focus on" not in prompt
    assert "File: a.py" in prompt
    assert "pass" in prompt


def test_build_prompt_multiple_files() -> None:
    chunks = [
        FileChunk(path=Path("a.py"), content="import os"),
        FileChunk(path=Path("b.js"), content="console.log(1)"),
    ]
    prompt = build_prompt(chunks, focus="performance")
    assert "File: a.py" in prompt
    assert "File: b.js" in prompt
    assert "import os" in prompt
    assert "console.log(1)" in prompt


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------


def test_ollama_generate_uses_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, str]:
            return {"response": "ok"}

    captured: dict = {}

    def fake_post(url: str, json: dict, timeout: int) -> DummyResponse:
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return DummyResponse()

    monkeypatch.setattr("requests.post", fake_post)

    result = ollama_generate("hello", "llama3.1", "http://localhost:11434")

    assert result == "ok"
    assert captured["url"].endswith("/api/generate")
    assert captured["json"]["model"] == "llama3.1"


def test_ollama_generate_raises_on_unexpected_response(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"error": "something went wrong"}

    def fake_post(url: str, json: dict, timeout: int) -> DummyResponse:
        return DummyResponse()

    monkeypatch.setattr("requests.post", fake_post)

    with pytest.raises(ValueError, match="Unexpected Ollama response"):
        ollama_generate("hello", "llama3.1", "http://localhost:11434")


def test_ollama_generate_with_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    call_count = 0

    class DummyResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, str]:
            return {"response": "generated"}

    def fake_post(url: str, json: dict, timeout: int) -> DummyResponse:
        nonlocal call_count
        call_count += 1
        return DummyResponse()

    monkeypatch.setattr("requests.post", fake_post)

    cache = AnalysisCache(cache_dir=tmp_path / "cache")

    # First call: goes to Ollama, result is cached
    result1 = ollama_generate("prompt1", "llama3.1", "http://localhost:11434", cache=cache)
    assert result1 == "generated"
    assert call_count == 1

    # Second call with same prompt: served from cache
    result2 = ollama_generate("prompt1", "llama3.1", "http://localhost:11434", cache=cache)
    assert result2 == "generated"
    assert call_count == 1  # no additional network call

    # Third call with different prompt: goes to Ollama
    result3 = ollama_generate("prompt2", "llama3.1", "http://localhost:11434", cache=cache)
    assert result3 == "generated"
    assert call_count == 2


def test_ollama_generate_without_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, str]:
            return {"response": "no-cache"}

    def fake_post(url: str, json: dict, timeout: int) -> DummyResponse:
        return DummyResponse()

    monkeypatch.setattr("requests.post", fake_post)

    # Passing cache=None should still work fine
    result = ollama_generate("hello", "llama3.1", "http://localhost:11434", cache=None)
    assert result == "no-cache"


# ---------------------------------------------------------------------------
# AnalysisCache
# ---------------------------------------------------------------------------


def test_cache_get_returns_none_when_empty(tmp_path: Path) -> None:
    cache = AnalysisCache(cache_dir=tmp_path / "empty_cache")
    assert cache.get("nonexistent") is None


def test_cache_put_and_get(tmp_path: Path) -> None:
    cache = AnalysisCache(cache_dir=tmp_path / "cache")
    cache.put("my-prompt", "my-response")
    assert cache.get("my-prompt") == "my-response"


def test_cache_different_prompts(tmp_path: Path) -> None:
    cache = AnalysisCache(cache_dir=tmp_path / "cache")
    cache.put("prompt-a", "response-a")
    cache.put("prompt-b", "response-b")
    assert cache.get("prompt-a") == "response-a"
    assert cache.get("prompt-b") == "response-b"


def test_cache_handles_corrupt_file(tmp_path: Path) -> None:
    cache = AnalysisCache(cache_dir=tmp_path / "cache")
    cache.put("prompt", "response")

    # Corrupt the cache file
    import hashlib

    h = hashlib.sha256("prompt".encode()).hexdigest()
    cache_file = tmp_path / "cache" / (h + ".json")
    cache_file.write_text("not valid json", encoding="utf-8")

    assert cache.get("prompt") is None


# ---------------------------------------------------------------------------
# GeminiProvider
# ---------------------------------------------------------------------------


def test_gemini_provider_raises_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GEMINI_API_KEY not set"):
        GeminiProvider(api_key="")


def test_gemini_provider_raises_without_api_key_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GEMINI_API_KEY not set"):
        GeminiProvider()


def test_gemini_refine_calls_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    mock_response = MagicMock()
    mock_response.text = "refined analysis"

    mock_model_instance = MagicMock()
    mock_model_instance.generate_content.return_value = mock_response

    mock_genai = MagicMock()
    mock_genai.GenerativeModel.return_value = mock_model_instance

    import sys

    saved = sys.modules.get("google.generativeai")
    sys.modules["google.generativeai"] = mock_genai
    try:
        result = gemini_refine(
            analysis="raw analysis",
            model="gemini-2.0-flash",
            api_key="test-key",
        )
    finally:
        if saved is not None:
            sys.modules["google.generativeai"] = saved
        else:
            sys.modules.pop("google.generativeai", None)

    assert result == "refined analysis"
    mock_genai.configure.assert_called_once_with(api_key="test-key")
    mock_genai.GenerativeModel.assert_called_once_with("gemini-2.0-flash")
    mock_model_instance.generate_content.assert_called_once()


def test_gemini_refine_raises_on_empty_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    mock_response = MagicMock()
    mock_response.text = ""

    mock_model_instance = MagicMock()
    mock_model_instance.generate_content.return_value = mock_response

    mock_genai = MagicMock()
    mock_genai.GenerativeModel.return_value = mock_model_instance

    import sys

    saved = sys.modules.get("google.generativeai")
    sys.modules["google.generativeai"] = mock_genai
    try:
        with pytest.raises(RuntimeError, match="Gemini returned empty output"):
            gemini_refine(
                analysis="raw analysis",
                model="gemini-2.0-flash",
                api_key="test-key",
            )
    finally:
        if saved is not None:
            sys.modules["google.generativeai"] = saved
        else:
            sys.modules.pop("google.generativeai", None)


# ---------------------------------------------------------------------------
# End-to-end pipeline (mocked)
# ---------------------------------------------------------------------------


def test_full_pipeline_without_gemini(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Simulate the full analysis pipeline: collect -> build prompt -> ollama."""
    (tmp_path / "app.py").write_text("def main():\n    pass\n")

    class DummyResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, str]:
            return {"response": "## No issues found\nCode looks clean."}

    def fake_post(url: str, json: dict, timeout: int) -> DummyResponse:
        return DummyResponse()

    monkeypatch.setattr("requests.post", fake_post)

    files = collect_files(tmp_path, include=["**/*.py"])
    assert len(files) == 1

    prompt = build_prompt(files, focus="bugs")
    assert "Focus on: bugs" in prompt
    assert "def main():" in prompt

    result = ollama_generate(prompt, "llama3.1", "http://localhost:11434")
    assert "No issues found" in result


def test_full_pipeline_with_caching(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify that caching prevents duplicate Ollama calls in the pipeline."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "lib.py").write_text("x = 42")

    call_count = 0

    class DummyResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, str]:
            return {"response": "analysis result"}

    def fake_post(url: str, json: dict, timeout: int) -> DummyResponse:
        nonlocal call_count
        call_count += 1
        return DummyResponse()

    monkeypatch.setattr("requests.post", fake_post)

    cache = AnalysisCache(cache_dir=tmp_path / ".cache")
    files = collect_files(tmp_path / "src", include=["**/*.py"])
    prompt = build_prompt(files, focus=None)

    # First run
    r1 = ollama_generate(prompt, "llama3.1", "http://localhost:11434", cache=cache)
    assert r1 == "analysis result"
    assert call_count == 1

    # Second run -- same files, same prompt => cache hit
    r2 = ollama_generate(prompt, "llama3.1", "http://localhost:11434", cache=cache)
    assert r2 == "analysis result"
    assert call_count == 1  # still 1
