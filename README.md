# Gemini + Ollama CLI Bridge

A small CLI that runs offline code analysis with a local Ollama model and (optionally) refines the results with the Gemini CLI. The goal is to keep analysis local-first while still letting you opt into Gemini for polish or cross-checking when you choose.

## Why
- **Offline by default**: Use your local Ollama instance for quick code review.
- **Gemini integration**: If the Gemini CLI is installed, refine the offline analysis with a second pass.

## Requirements
- Python 3.10+
- Ollama running locally (`ollama serve`) with a model pulled
- Optional: Gemini CLI installed and available on PATH

## Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
```

## Usage
```bash
python -m gemini_ollama_bridge analyze \
  --path . \
  --model llama3.1 \
  --focus "bugs, security, performance" \
  --gemini-refine
```

### Common flags
- `--path`: folder to scan (default: current directory)
- `--include`: glob patterns (repeatable) to include
- `--exclude`: glob patterns (repeatable) to exclude
- `--model`: Ollama model name (or set `OLLAMA_MODEL`)
- `--ollama-url`: base URL for Ollama (default: `http://localhost:11434`)
- `--gemini-refine`: run Gemini CLI on the offline result
- `--gemini-command`: command to invoke Gemini (default: `gemini`)
- `--gemini-args`: extra args passed to Gemini (repeatable)

## Notes on Gemini CLI
The bridge sends the refinement prompt via **stdin** to the Gemini CLI. If your CLI requires flags for model selection, pass them using `--gemini-args`.

Example:
```bash
python -m gemini_ollama_bridge analyze --gemini-refine \
  --gemini-command gemini \
  --gemini-args --model \
  --gemini-args gemini-2.0-flash
```

## Output
Results are printed to stdout as Markdown so you can pipe into a file or another tool.
