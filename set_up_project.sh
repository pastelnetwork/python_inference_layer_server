#!/bin/bash

# Detect the shell type and source appropriate profile
if [[ "$SHELL" == *"zsh"* ]]; then
  SHELL_PROFILE=".zshrc"
else
  SHELL_PROFILE=".bashrc"
fi

# Install pyenv if not already installed
if ! command -v pyenv &> /dev/null; then
    sudo apt-get update
    sudo apt-get install -y build-essential libssl-dev zlib1g-dev libbz2-dev \
    libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
    xz-utils tk-dev libffi-dev liblzma-dev python3-openssl git

    git clone https://github.com/pyenv/pyenv.git ~/.pyenv
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/$SHELL_PROFILE
    echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/$SHELL_PROFILE
    echo 'eval "$(pyenv init --path)"' >> ~/$SHELL_PROFILE
    source ~/$SHELL_PROFILE
fi

# Set PYENV_ROOT and update PATH for this script
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"

# Update pyenv and install Python 3.12
cd ~/.pyenv && git pull && cd -
pyenv install -s 3.12

# Setup the project directory
PROJECT_DIR="$HOME/python_inference_layer_server"
if [ -d "$PROJECT_DIR" ]; then
    cd "$PROJECT_DIR"
    pyenv local 3.12
    python -m venv venv
    source venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install wheel
    pip install -r requirements.txt
else
    echo "Project directory does not exist."
fi
