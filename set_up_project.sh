# Install pyenv if not already installed
if ! command -v pyenv &> /dev/null; then
    sudo apt-get update
    sudo apt-get install -y build-essential libssl-dev zlib1g-dev libbz2-dev \
    libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
    xz-utils tk-dev libffi-dev liblzma-dev python3-openssl git

    git clone https://github.com/pyenv/pyenv.git ~/.pyenv
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
    echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
    echo 'eval "$(pyenv init --path)"' >> ~/.zshrc
fi

# Set PYENV_ROOT and update PATH for this script
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"

# Update pyenv
cd ~/.pyenv && git pull && cd -

# Install Python 3.12 using pyenv
pyenv install 3.12

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
