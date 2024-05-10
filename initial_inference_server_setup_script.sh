#!/bin/bash

# Determine the user shell
user_shell=$(echo $SHELL)
echo "Detected user shell: $user_shell"

# Set shell path and profile file based on user shell
if [[ $user_shell == *"/zsh"* ]]; then
  profile_file="zshrc"
  shell_cmd="zsh"
else
  profile_file="bashrc"
  shell_cmd="bash"
fi
echo "Using profile file: ~/.$profile_file with shell command: $shell_cmd"

# Update code block with appropriate shell execution
if [ -d ~/python_inference_layer_server ]; then
  echo "Directory exists. Stashing and pulling latest code..."
  cd ~/python_inference_layer_server
  git stash
  git pull
else
  echo "Directory does not exist. Cloning repository..."
  cd ~
  git clone https://github.com/pastelnetwork/python_inference_layer_server.git
  cd python_inference_layer_server
fi

# Get the name of the existing tmux session, create one if it doesn't exist
tmux_session_name=$(tmux list-sessions -F '#{session_name}' | head -1)
if [[ -z "$tmux_session_name" ]]; then
  echo "No tmux session found. Creating a new session..."
  tmux new-session -d -s default_session
  tmux_session_name="default_session"
else
  echo "Found existing tmux session: $tmux_session_name"
fi

# Check if 'supernode_script' window exists in tmux
window_exists=$(tmux list-windows -t $tmux_session_name -F '#{window_name}' | grep -q '^supernode_script$'; echo $?)
if [[ $window_exists -eq 0 ]]; then
  echo "Supernode_script window exists. Killing window..."
  tmux kill-window -t $tmux_session_name:supernode_script
else
  echo "No existing 'supernode_script' window found. Proceeding to create one..."
fi

# Create temporary script
echo "Creating temporary script..."
cat << EOF > ~/run_script.sh
#!/bin/$shell_cmd
source ~/.$profile_file
cd ~/python_inference_layer_server
pyenv local 3.12
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --upgrade setuptools
pip install wheel
pip install -r requirements.txt
python main.py
EOF
chmod 0755 ~/run_script.sh

# Launch script in new tmux window using the appropriate shell
echo "Launching script in new tmux window..."
tmux new-window -t $tmux_session_name: -n supernode_script -d "$shell_cmd ~/run_script.sh"
