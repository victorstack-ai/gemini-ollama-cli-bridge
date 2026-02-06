from __future__ import annotations

import argparse
import os
from pathlib import Path

from .analysis import build_prompt, collect_files, gemini_refine, ollama_generate


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="gemini-ollama-bridge",
        description="Offline code analysis with Ollama + optional Gemini refinement.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="Analyze code with Ollama")
    analyze.add_argument("--path", default=".", help="Base path to scan")
    analyze.add_argument(
        "--include",
        action="append",
        default=[],
        help="Glob pattern to include (repeatable)",
    )
    analyze.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Glob pattern or folder name to exclude (repeatable)",
    )
    analyze.add_argument(
        "--model",
        default=os.getenv("OLLAMA_MODEL", "llama3.1"),
        help="Ollama model name",
    )
    analyze.add_argument(
        "--ollama-url",
        default=os.getenv("OLLAMA_URL", "http://localhost:11434"),
        help="Ollama base URL",
    )
    analyze.add_argument(
        "--focus",
        default="",
        help="Comma-separated focus areas (bugs, security, performance)",
    )
    analyze.add_argument(
        "--max-file-bytes",
        type=int,
        default=120_000,
        help="Skip files larger than this",
    )
    analyze.add_argument(
        "--max-total-bytes",
        type=int,
        default=400_000,
        help="Stop after this many bytes of files",
    )
    analyze.add_argument(
        "--gemini-refine",
        action="store_true",
        help="Run Gemini CLI to refine the offline analysis",
    )
    analyze.add_argument(
        "--gemini-command",
        default=os.getenv("GEMINI_CMD", "gemini"),
        help="Command to invoke Gemini CLI",
    )
    analyze.add_argument(
        "--gemini-args",
        action="append",
        default=[],
        help="Extra args for Gemini CLI (repeatable)",
    )

    return parser.parse_args()


def run() -> int:
    args = _parse_args()

    if args.command == "analyze":
        base_path = Path(args.path).resolve()
        include = args.include or None
        exclude = args.exclude or None

        files = collect_files(
            base_path,
            include=include,
            exclude=exclude,
            max_file_bytes=args.max_file_bytes,
            max_total_bytes=args.max_total_bytes,
        )

        if not files:
            raise SystemExit("No files matched. Adjust --include/--exclude.")

        prompt = build_prompt(files, args.focus or None)
        analysis = ollama_generate(prompt, args.model, args.ollama_url)

        if args.gemini_refine:
            analysis = gemini_refine(
                analysis=analysis,
                command=args.gemini_command,
                args=args.gemini_args,
            )

        print(analysis)
        return 0

    raise SystemExit(f"Unknown command: {args.command}")
