# alt-text-llm

AI-powered alt text generation and labeling tools for markdown content. Originally developed for [my website](https://turntrout.com/design) ([repo](https://github.com/alexander-turner/TurnTrout.com)).

## Installation

### Quick install from GitHub

```bash
pip install git+https://github.com/alexander-turner/alt-text-llm.git
```

### Automated setup (includes system dependencies)

```bash
git clone https://github.com/alexander-turner/alt-text-llm.git
cd alt-text-llm
./setup.sh
```

## Prerequisites

The following command-line tools must be installed:

- **`llm`** - LLM interface ([install instructions](https://llm.datasette.io/))
- **`git`** - Version control
- **`magick`** (ImageMagick) - Image processing
- **`ffmpeg`** - Video processing
- **`imgcat`** - Terminal image display

**macOS:**

```bash
brew install imagemagick ffmpeg imgcat
pip install llm
```

**Linux:**

```bash
sudo apt-get install imagemagick ffmpeg
pip install llm
# imgcat: curl -sL https://iterm2.com/utilities/imgcat -o ~/.local/bin/imgcat && chmod +x ~/.local/bin/imgcat
```

## Usage

The tool provides three main commands: `scan`, `generate`, and `label`.

### 1. Scan for missing alt text

Scan your markdown files to find images without meaningful alt text:

```bash
alt-text-llm scan --root /path/to/markdown/files
```

This creates `asset_queue.json` with all assets needing alt text.

### 2. Generate AI suggestions

Generate alt text suggestions using an LLM:

```bash
alt-text-llm generate \
  --root /path/to/markdown/files \
  --model gemini-2.5-flash \
  --suggestions-file suggested_alts.json
```

**Available options:**

- `--model` (required) - LLM model to use (e.g., `gemini-2.5-flash`, `gpt-4o-mini`, `claude-3-5-sonnet`)
- `--max-chars` - Maximum characters for alt text (default: 300)
- `--timeout` - LLM timeout in seconds (default: 120)
- `--estimate-only` - Only show cost estimate without generating
- `--process-existing` - Also process assets that already have captions

**Cost estimation:**

```bash
alt-text-llm generate \
  --root /path/to/markdown/files \
  --model gemini-2.5-flash \
  --estimate-only
```

### 3. Label and approve suggestions

Interactively review and approve the AI-generated suggestions:

```bash
alt-text-llm label \
  --suggestions-file suggested_alts.json \
  --output asset_captions.json
```

**Interactive commands:**

- Edit the suggested alt text (vim keybindings enabled)
- Press Enter to accept the suggestion as-is
- Submit `undo` or `u` to go back to the previous item
- Images display in your terminal (requires `imgcat`)

## Example workflow

```bash
# 1. Scan markdown files for missing alt text
alt-text-llm scan --root ./content

# 2. Estimate the cost
alt-text-llm generate \
  --root ./content \
  --model gemini-2.5-flash \
  --estimate-only

# 3. Generate suggestions (if cost is acceptable)
alt-text-llm generate \
  --root ./content \
  --model gemini-2.5-flash

# 4. Review and approve suggestions
alt-text-llm label
```

## Configuration

### LLM Integration

This tool uses the [`llm` CLI tool](https://llm.datasette.io/) to generate alt text. This provides access to many different AI models including:

- **Gemini** (Google) via the [llm-gemini plugin](https://github.com/simonw/llm-gemini)
- **Claude** (Anthropic) via the [llm-claude-3 plugin](https://github.com/tomviner/llm-claude-3)
- And [many more via plugins](https://llm.datasette.io/en/stable/plugins/directory.html)

### Setting up your model

**For Gemini models (default):**

```bash
llm install llm-gemini
llm keys set gemini # enter API key
llm -m gemini-2.5-flash "Hello, world!"
```

**For other models:**

1. Install the appropriate llm plugin (e.g., `llm install llm-openai`)
2. Configure your API key (e.g., `llm keys set openai`)
3. Use the model name with `--model` flag (e.g., `--model gpt-4o-mini`)

See the [llm documentation](https://llm.datasette.io/en/stable/setup.html) for setup instructions and the [plugin directory](https://llm.datasette.io/en/stable/plugins/directory.html) for available models.

## Output files

- `asset_queue.json` - Queue of assets needing alt text (from `scan`)
- `suggested_alts.json` - AI-generated suggestions (from `generate`)
- `asset_captions.json` - Approved final captions (from `label`)
