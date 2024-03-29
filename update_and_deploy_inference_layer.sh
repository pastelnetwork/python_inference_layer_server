#!/bin/bash

# Change to the project directory
cd /home/ubuntu/python_supernode_messaging_and_control_layer

# Pull the latest changes from the repository
git stash
git pull

# Set the Python version using pyenv
pyenv local 3.12

# Activate the virtual environment
source venv/bin/activate

# Install the required packages
pip install -r requirements.txt

# Launch the Python script
python main.py