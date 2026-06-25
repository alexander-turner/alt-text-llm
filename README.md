# alt-text-llm

AI-powered alt text generation and labeling tools for markdown content. Originally developed for [my website](https://turntrout.com/design) ([repo](https://github.com/alexander-turner/TurnTrout.com)).

## Features

- **Intelligent scanning** - Detects images/videos missing meaningful alt text (ignores empty `alt=""`)
- **AI-powered generation** - Uses LLM of your choice to create context-aware alt text suggestions
- **Interactive labeling** - Manually review and edit LLM suggestions. Images display directly in your terminal
- **Automatic application** - Apply approved captions back to your markdown files

![A labeled example of the labeling pipeline: 1) view the context for an image, 2) view the image itself, while 3) editing the AI-generated label suggestion.](image.png)

## Installation

### From PyPI

```bash
pip install alt-text-llm
```

### Automated setup (includes system dependencies)

```bash
git clone https://github.com/alexander-turner/alt-text-llm.git
cd alt-text-llm
./setup.sh
```

## Prerequisites

**macOS:**

```bash
brew install imagemagick ffmpeg imgcat
```

**Linux:**

```bash
sudo apt-get install imagemagick ffmpeg
# imgcat: curl -sL https://iterm2.com/utilities/imgcat -o ~/.local/bin/imgcat && chmod +x ~/.local/bin/imgcat
```

Alt text is generated through [OpenRouter](https://openrouter.ai). Get an API
key at https://openrouter.ai/keys and export it before running `generate`:

```bash
export OPENROUTER_API_KEY=sk-or-...
```

## Usage

The tool provides four main commands: `scan`, `generate`, `label`, and `apply`.

### 1. Scan for missing alt text

Scan your markdown files to find images without meaningful alt text:

```bash
alt-text-llm scan --root /path/to/markdown/files
```

This creates `asset_queue.json` with all assets needing alt text. `--root`
defaults to the current directory.

### 2. Generate AI suggestions

Generate alt text suggestions using an LLM:

```bash
alt-text-llm generate \
  --root /path/to/markdown/files \
  --model google/gemini-2.5-flash \
  --suggestions-file suggested_alts.json
```

**Available options:**

- `--model` (required) - OpenRouter model id of the form `provider/model-slug` (e.g., `google/gemini-2.5-flash`, `google/gemini-2.5-flash-lite`, `google/gemini-2.5-pro`, `anthropic/claude-sonnet-4.5`). Browse all available ids at https://openrouter.ai/models.
- `--max-chars` - Maximum characters for alt text (default: 300)
- `--timeout` - LLM timeout in seconds (default: 120)
- `--estimate-only` - Only show cost estimate without generating
- `--process-existing` - Also process assets that already have captions

> **Note:** Video alt text generation requires a model that accepts video
> input (the Google Gemini models).

**Cost estimation:**

Cost estimates are pulled live from OpenRouter's model catalogue
(`https://openrouter.ai/api/v1/models`), so `--estimate-only` works for any
model OpenRouter prices. After a real run, the tool also prints the actual
total cost reported by OpenRouter per request.

```bash
alt-text-llm generate \
  --root /path/to/markdown/files \
  --model google/gemini-2.5-flash \
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

- Edit the suggested alt text (pass `--vi-mode` for vim keybindings)
- Press Enter to accept the suggestion as-is
- Submit `undo` or `u` to go back to the previous item
- Images display in your terminal (requires `imgcat`)
- Pass `--no-skip-existing` to relabel assets already present in the output file

### 4. Apply approved captions

Apply the approved captions back to your markdown files:

```bash
alt-text-llm apply \
  --captions-file asset_captions.json
```

**Available options:**

- `--captions-file` - Path to the captions JSON file with `final_alt` populated (default: `asset_captions.json`)
- `--dry-run` - Preview changes without modifying files

**What it does:**

- Reads approved captions from the captions file
- Locates corresponding images/videos in markdown files
- Updates alt text for all supported formats:
  - Markdown images: `![alt](path)`
  - HTML img tags: `<img src="path" alt="alt">`
  - Wikilink images: `![[path|alt]]`
- Preserves file formatting and handles special characters

## Example workflow

```bash
# 1. Scan markdown files for missing alt text
alt-text-llm scan --root ./content

# 2. Estimate the cost
alt-text-llm generate \
  --root ./content \
  --model google/gemini-2.5-flash \
  --estimate-only

# 3. Generate suggestions (if cost is acceptable)
alt-text-llm generate \
  --root ./content \
  --model google/gemini-2.5-flash

# 4. Review and approve suggestions
alt-text-llm label

# 5. Apply approved captions to markdown files
alt-text-llm apply
```

## Configuration

### LLM Integration

This tool calls [OpenRouter](https://openrouter.ai) directly over HTTPS to
generate alt text, which provides access to many models from many providers
through a single API key.

### Setting up your model

1. Create an OpenRouter API key at https://openrouter.ai/keys.
2. Export it in your shell:

   ```bash
   export OPENROUTER_API_KEY=sk-or-...
   ```

3. Pass an OpenRouter model id of the form `provider/model-slug` to `--model`,
   for example:

   - `google/gemini-2.5-flash`
   - `google/gemini-2.5-flash-lite`
   - `google/gemini-2.5-pro`
   - `anthropic/claude-sonnet-4.5`

Browse the full catalogue of available model ids at
https://openrouter.ai/models. Note that only video-capable models (the Google
Gemini models) can generate alt text for videos.

### Shell completion

Tab-completion is provided through
[`argcomplete`](https://github.com/kislyuk/argcomplete). Completing the
`--model` flag suggests live OpenRouter model ids (this needs network access;
the public model list does not require an API key).

Enable it globally for bash (one-time, then restart your shell):

```bash
activate-global-python-argcomplete
```

Or enable it per-shell.

**bash:**

```bash
eval "$(register-python-argcomplete alt-text-llm)"
```

**zsh:**

```bash
autoload -U bashcompinit && bashcompinit
eval "$(register-python-argcomplete alt-text-llm)"
```

For zsh's native completion system you can instead use
`register-python-argcomplete --shell zsh alt-text-llm`.

## Output files

- `asset_queue.json` - Queue of assets needing alt text (from `scan`)
- `suggested_alts.json` - AI-generated suggestions (from `generate`)
- `asset_captions.json` - Approved final captions (from `label`)
