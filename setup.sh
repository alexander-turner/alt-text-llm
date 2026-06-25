#!/bin/bash
set -e
set -o pipefail

echo "Installing alt-text-llm and dependencies..."
pip install -e .

echo ""
echo "Installing system dependencies..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "macOS detected - using Homebrew"
    brew install imagemagick ffmpeg imgcat
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Linux detected - using apt"
    sudo apt-get update -qq && sudo apt-get install -y -qq imagemagick ffmpeg
    # Install imgcat
    mkdir -p ~/.local/bin
    if curl -sL https://iterm2.com/utilities/imgcat -o ~/.local/bin/imgcat && [[ -s ~/.local/bin/imgcat ]]; then
        chmod +x ~/.local/bin/imgcat
    else
        echo "⚠️  Failed to download imgcat; please install it manually."
        rm -f ~/.local/bin/imgcat
    fi
    echo ""
    echo "ℹ️  Ensure ~/.local/bin is on your PATH. Add the following line to your"
    echo "    shell profile (e.g. ~/.bashrc or ~/.zshrc) so it persists:"
    echo '        export PATH="$HOME/.local/bin:$PATH"'
else
    echo "⚠️  Unknown OS - please manually install: imagemagick, ffmpeg, imgcat"
fi

echo ""
echo "✅ Setup complete!"
echo "Next step: Set your OpenRouter API key (get one at https://openrouter.ai/keys):"
echo "    export OPENROUTER_API_KEY=sk-or-..."
echo ""
echo "Optional: enable shell tab-completion."
echo "    bash: activate-global-python-argcomplete   (one-time, then restart shell)"
echo "    or:   eval \"\$(register-python-argcomplete alt-text-llm)\""
