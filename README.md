# Gemini + Ollama CLI Bridge

A local-first CLI tool that runs offline code analysis with a local [Ollama](https://ollama.com/) model and optionally refines the results through the [Google Gemini](https://ai.google.dev/) API. Keep your code review workflow private and fast by default, with the option to leverage Gemini for a second-pass refinement when you choose.

## Features

- **Offline by default** -- all analysis runs against your local Ollama instance.
- **Gemini refinement** -- optionally send the local analysis to Google Gemini for cross-checking or polish via the official Python SDK.
- **Result caching** -- Ollama analysis results are cached based on file content hashes, so unchanged files are not re-processed.
- **Flexible file selection** -- glob-based include/exclude patterns with sensible defaults for common project layouts.
- **Markdown output** -- results print as Markdown for easy piping into files, viewers, or other tools.

## Architecture

```
                +-----------+
  source files  |  collect  |   glob include/exclude
  on disk ----->|  files    |----> FileChunk list
                +-----------+
                      |
                      v
                +-----------+
                |  build    |   system prompt + file contents
                |  prompt   |----> single prompt string
                +-----------+
                      |
                      v
                +-----------+
                |  Ollama   |   local HTTP API  (cache-aware)
                |  generate |----> raw analysis text
                +-----------+
                      |
              (optional)
                      v
                +-----------+
                |  Gemini   |   google-generativeai SDK
                |  refine   |----> refined analysis text
                +-----------+
                      |
                      v
                   stdout
```

The pipeline has three main stages:

1. **File collection** (`collect_files`) -- walks the target directory, filters by include/exclude globs, and respects byte-size limits.
2. **Ollama analysis** (`ollama_generate`) -- sends the assembled prompt to a locally-running Ollama model. Results are cached on disk keyed by a SHA-256 hash of the prompt, so identical inputs skip the network call.
3. **Gemini refinement** (`GeminiProvider.refine`) -- an optional second pass that sends the Ollama output to Google Gemini using the `google-generativeai` Python SDK. Requires a `GEMINI_API_KEY` environment variable.

## Installation

### Prerequisites

- Python 3.10 or later
- [Ollama](https://ollama.com/) installed and running (`ollama serve`)
- At least one Ollama model pulled (e.g. `ollama pull llama3.1`)
- (Optional) A Google Gemini API key for the refinement step

### Steps

```bash
# Clone the repository
git clone https://github.com/yourusername/gemini-ollama-cli-bridge.git
cd gemini-ollama-cli-bridge

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
# .venv\Scripts\activate    # Windows

# Install runtime dependencies
pip install -r requirements.txt

# Install development dependencies (pytest, ruff)
pip install -r requirements-dev.txt
```

## Configuration

### Ollama

| Setting | Default | Environment Variable |
|---------|---------|---------------------|
| Model name | `llama3.1` | `OLLAMA_MODEL` |
| Base URL | `http://localhost:11434` | `OLLAMA_URL` |

Make sure Ollama is running before you use the tool:

```bash
ollama serve          # start the Ollama server
ollama pull llama3.1  # pull a model (one-time)
```

### Gemini

| Setting | Default | Environment Variable |
|---------|---------|---------------------|
| API key | (none -- required for `--gemini-refine`) | `GEMINI_API_KEY` |
| Model | `gemini-2.0-flash` | `GEMINI_MODEL` |

To use Gemini refinement, set your API key:

```bash
export GEMINI_API_KEY="your-api-key-here"
```

You can obtain an API key from [Google AI Studio](https://aistudio.google.com/apikey).

## Usage

### Basic local analysis

```bash
python -m gemini_ollama_bridge analyze --path ./my-project
```

### Specify a model and focus areas

```bash
python -m gemini_ollama_bridge analyze \
  --path ./my-project \
  --model codellama \
  --focus "bugs, security, performance"
```

### With Gemini refinement

```bash
export GEMINI_API_KEY="your-key"
python -m gemini_ollama_bridge analyze \
  --path ./my-project \
  --gemini-refine \
  --gemini-model gemini-2.0-flash
```

### Custom file selection

```bash
python -m gemini_ollama_bridge analyze \
  --path ./my-project \
  --include "**/*.py" \
  --include "**/*.ts" \
  --exclude "tests" \
  --exclude "*.generated.*"
```

### Sample output

```
## Code Analysis Results

### File: src/main.py

**[HIGH] Potential SQL injection on line 42**
The query string is built via f-string interpolation with user input.
Suggested fix: use parameterised queries with `cursor.execute(sql, params)`.

**[MEDIUM] Unused import `os` on line 1**
The `os` module is imported but never referenced.
Suggested fix: remove the import.

### File: src/config.py

**[LOW] Hard-coded timeout value on line 18**
The timeout of 30 seconds is embedded directly in the code.
Suggested fix: extract to a configuration constant or environment variable.
```

### CLI reference

| Flag | Description | Default |
|------|-------------|---------|
| `--path` | Folder to scan | `.` (current directory) |
| `--include` | Glob patterns to include (repeatable) | common source extensions |
| `--exclude` | Glob patterns / folder names to exclude (repeatable) | `.git`, `node_modules`, etc. |
| `--model` | Ollama model name | `llama3.1` / `$OLLAMA_MODEL` |
| `--ollama-url` | Base URL for Ollama | `http://localhost:11434` / `$OLLAMA_URL` |
| `--focus` | Comma-separated focus areas | (none) |
| `--max-file-bytes` | Skip files larger than this | `120000` |
| `--max-total-bytes` | Stop collecting after this many bytes | `400000` |
| `--gemini-refine` | Enable Gemini refinement pass | off |
| `--gemini-model` | Gemini model to use | `gemini-2.0-flash` / `$GEMINI_MODEL` |
| `--no-cache` | Disable result caching | caching on |

## Troubleshooting

### "Connection refused" when calling Ollama

Make sure the Ollama server is running:

```bash
ollama serve
```

If you changed the default port, set `--ollama-url` or `OLLAMA_URL` accordingly.

### "No files matched. Adjust --include/--exclude."

The scanner did not find any files that match the include patterns. Check that:

1. `--path` points to the correct directory.
2. Your project contains files with supported extensions (`.py`, `.js`, `.ts`, `.php`, `.md`, `.json`, `.yaml`, `.yml`).
3. Custom `--exclude` patterns are not too broad.

### Gemini refinement fails with "GEMINI_API_KEY not set"

You need to export the API key before running the command:

```bash
export GEMINI_API_KEY="your-key"
```

### Gemini refinement returns an API error

- Verify your API key is valid at [Google AI Studio](https://aistudio.google.com/apikey).
- Check that the model name is correct (default: `gemini-2.0-flash`).
- Ensure you have not exceeded your API quota.

### Cached results seem stale

The cache is keyed on the SHA-256 hash of the full prompt (which includes file contents). If files change, the cache key changes automatically. To force a fresh analysis, pass `--no-cache`:

```bash
python -m gemini_ollama_bridge analyze --path . --no-cache
```

You can also clear the cache directory manually:

```bash
rm -rf .ollama_cache/
```

## Running tests

```bash
pytest tests/ -v
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
