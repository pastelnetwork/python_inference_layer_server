#!/bin/bash

# Determine the user shell
user_shell=$(echo $SHELL)

# Set shell path and profile file based on user shell
if [[ $user_shell == *"/zsh"* ]]; then
  profile_file="zshrc"
else
  profile_file="bashrc"
fi

# Source the appropriate profile file
source ~/.$profile_file

# Update code block
cd ~/python_inference_layer_server || {
  cd ~
  git clone https://github.com/pastelnetwork/python_inference_layer_server.git
  cd python_inference_layer_server
}
git stash
git pull

# Get the name of the existing tmux session, create one if it doesn't exist
tmux_session_name=$(tmux list-sessions -F '#{session_name}' | head -1)
if [[ -z "$tmux_session_name" ]]; then
  tmux new-session -d -s default_session
  tmux_session_name="default_session"
fi

# Check if 'supernode_script' window exists in tmux
window_exists=$(tmux list-windows -t $tmux_session_name -F '#{window_name}' | grep -q '^supernode_script$'; echo $?)
if [[ $window_exists -eq 0 ]]; then
  # Kill existing 'supernode_script' window
  tmux kill-window -t $tmux_session_name:supernode_script
fi

# Create temporary script
cat << EOF > ~/run_script.sh
#!/bin/$user_shell
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

# Launch script in new tmux window
tmux new-window -t $tmux_session_name: -n supernode_script -d "$SHELL -c '~/run_script.sh'"

# Remove the temporary script after execution in tmux window
rm ~/run_script.sh
