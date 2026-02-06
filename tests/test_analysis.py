from __future__ import annotations

from pathlib import Path

import pytest

from gemini_ollama_bridge.analysis import (
    build_prompt,
    collect_files,
    gemini_refine,
    ollama_generate,
)


def test_collect_files_respects_include_exclude(tmp_path: Path) -> None:
    (tmp_path / "keep.py").write_text("print('ok')")
    (tmp_path / "skip.txt").write_text("nope")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "bad.js").write_text("alert(1)")

    files = collect_files(tmp_path, include=["**/*.py", "**/*.txt"], exclude=["*.txt"])
    assert [f.path.name for f in files] == ["keep.py"]


def test_build_prompt_contains_focus_and_files(tmp_path: Path) -> None:
    file_path = tmp_path / "demo.py"
    file_path.write_text("x = 1")

    files = collect_files(tmp_path, include=["**/*.py"])
    prompt = build_prompt(files, focus="bugs, security")

    assert "Focus on: bugs, security" in prompt
    assert "File: demo.py" in prompt
    assert "x = 1" in prompt


def test_ollama_generate_uses_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyResponse:
        def __init__(self) -> None:
            self.called = False

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, str]:
            return {"response": "ok"}

    captured = {}

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


def test_gemini_refine_invokes_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_which(cmd: str) -> str:
        return "/usr/bin/gemini"

    class DummyResult:
        returncode = 0
        stdout = "refined"
        stderr = ""

    def fake_run(*args, **kwargs):
        return DummyResult()

    monkeypatch.setattr("shutil.which", fake_which)
    monkeypatch.setattr("subprocess.run", fake_run)

    result = gemini_refine("analysis", command="gemini", args=["--model", "x"])
    assert result == "refined"
