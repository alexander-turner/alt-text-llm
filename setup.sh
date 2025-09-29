#!/bin/bash
set -e

# Install Python dependencies
pip install -q -r requirements.txt
pip install -q llm

# Install system tools based on OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    brew install imagemagick ffmpeg imgcat
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    sudo apt-get update -qq && sudo apt-get install -y -qq imagemagick ffmpeg
    # Install imgcat
    mkdir -p ~/.local/bin
    curl -sL https://iterm2.com/utilities/imgcat -o ~/.local/bin/imgcat && chmod +x ~/.local/bin/imgcat
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "Setup complete. Configure llm: 'llm keys set gemini'"