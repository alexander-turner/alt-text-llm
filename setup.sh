#!/bin/bash
set -e

echo "Installing alt-text-llm and dependencies..."
pip install -e .

echo "Installing llm tool..."
pip install llm

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
    curl -sL https://iterm2.com/utilities/imgcat -o ~/.local/bin/imgcat && chmod +x ~/.local/bin/imgcat
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "⚠️  Unknown OS - please manually install: imagemagick, ffmpeg, imgcat"
fi

echo ""
echo "✅ Setup complete!"
echo "Next step: Configure your LLM API key with 'llm keys set gemini'"