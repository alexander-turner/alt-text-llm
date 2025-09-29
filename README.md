# alt-text-llm

AI-powered alt text generation and labeling tools for markdown content.

## Installation

```bash
cd ~/Downloads
git clone https://github.com/alexander-turner/alt-text-llm.git
cd alt-text-llm
sh ./setup.sh
```

## Usage

The tool provides three main commands: `scan`, `generate`, and `label`.

### 1. Scan for missing alt text

Scan your markdown files to find images without meaningful alt text:

```bash
python -m alt_text_llm.main scan --root /path/to/markdown/files
```

This creates `asset_queue.json` with all assets needing alt text.

### 2. Generate AI suggestions

Generate alt text suggestions using an LLM:

```bash
python -m alt_text_llm.main generate \
  --root /path/to/markdown/files \
  --model gemini-2.5-flash \
  --suggestions-file suggested_alts.json
```

**Available options:**

- `--model` (required) - LLM model to use (e.g., `gemini-2.5-flash`, `gemini-2.5-pro`)
- `--max-chars` - Maximum characters for alt text (default: 300)
- `--timeout` - LLM timeout in seconds (default: 120)
- `--estimate-only` - Only show cost estimate without generating
- `--process-existing` - Also process assets that already have captions

**Cost estimation:**

```bash
python -m alt_text_llm.main generate \
  --root /path/to/markdown/files \
  --model gemini-2.5-flash \
  --estimate-only
```

### 3. Label and approve suggestions

Interactively review and approve the AI-generated suggestions:

```bash
python -m alt_text_llm.main label \
  --suggestions-file suggested_alts.json \
  --output asset_captions.json
```

**Interactive commands:**

- Edit the suggested alt text (vim keybindings enabled)
- Press Enter to accept the suggestion as-is
- Type `undo` or `u` to go back to the previous item
- Images display in your terminal (requires `imgcat`)

## Complete workflow example

```bash
# 1. Scan markdown files for missing alt text
python -m alt_text_llm.main scan --root ./content

# 2. Estimate the cost
python -m alt_text_llm.main generate \
  --root ./content \
  --model gemini-2.5-flash \
  --estimate-only

# 3. Generate suggestions (if cost is acceptable)
python -m alt_text_llm.main generate \
  --root ./content \
  --model gemini-2.5-flash

# 4. Review and approve suggestions
python -m alt_text_llm.main label
```

## Configuration

### Setting up LLM

The `generate` command requires the `llm` tool to be configured with API keys:

```bash
# For Gemini models
llm keys set gemini
# Enter your API key when prompted

# Test it works
llm -m gemini-2.5-flash "Hello, world!"
```

See the [llm documentation](https://llm.datasette.io/) for more details on configuration and available models.

## Output files

- `asset_queue.json` - Queue of assets needing alt text (from `scan`)
- `suggested_alts.json` - AI-generated suggestions (from `generate`)
- `asset_captions.json` - Approved final captions (from `label`)

## Requirements

- Python 3.11+
- External executables: `git`, `llm`, `magick`, `ffmpeg`, `imgcat`

## License

MIT

## Author

Alexander Turner (TurnTrout)
