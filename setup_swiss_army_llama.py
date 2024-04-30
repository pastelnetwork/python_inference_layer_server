import os
import subprocess
import httpx
import re
from logger_config import setup_logger
logger = setup_logger()

def get_external_ip_func():
    providers = [
        "https://ipinfo.io/ip",
        "https://api.ipify.org",
        "https://checkip.amazonaws.com",
        "https://icanhazip.com"
    ]
    for provider in providers:
        try:
            response = httpx.get(provider)
            response.raise_for_status()
            return response.text.strip()
        except httpx.RequestError as e:
            logger.warning(f"Failed to retrieve external IP address from {provider}: {e}")
    logger.warning("Failed to retrieve external IP address from all providers.")
    return "Unknown"

def is_port_available(port):
    try:
        result = subprocess.run(["lsof", "-i", f":{port}"], capture_output=True)
        return result.returncode != 0
    except:  # noqa: E722
        return True

def is_swiss_army_llama_responding(swiss_army_llama_port: int, security_token: str):
    try:
        url = f"http://localhost:{swiss_army_llama_port}/get_list_of_available_model_names/"
        params = {'token': security_token}  # Assuming the token should be passed as a query parameter
        response = httpx.get(url, params=params)
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False
    
def run_command(command, shell='/bin/bash', env=None):
    """Runs a command through subprocess.run with specified shell and environment."""
    subprocess.run(command, shell=True, executable=shell, env=env)

def is_pyenv_installed():
    """Checks if pyenv is installed by attempting to run 'pyenv --version' in a subshell."""
    shell_path = os.environ.get('SHELL', '/bin/bash')
    profile_file = "zshrc" if "/zsh" in shell_path else "bashrc"
    command = f"source ~/.{profile_file} && pyenv --version"
    try:
        subprocess.run(command, shell=True, executable=shell_path, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False
    
def is_python_3_12_installed():
    try:
        result = subprocess.run(["pyenv", "versions", "--bare"], capture_output=True, text=True)
        return "3.12" in result.stdout
    except:  # noqa: E722
        return False

def is_rust_installed():
    """Check if Rust is installed by attempting to run `rustc --version`."""
    try:
        subprocess.run(["rustc", "--version"], capture_output=True, check=True)
        return True
    except subprocess.CalledProcessError:
        return False
    
def set_timezone_utc():
    """
    Sets the timezone to UTC for the current Python session and permanently for the system.
    For permanent setting, it appends `export TZ=UTC` to the user's shell profile file.
    """
    # Set timezone to UTC for the current session
    os.environ['TZ'] = 'UTC'
    # Determine the user's shell profile file
    shell_profile_path = os.path.expanduser('~/.bashrc')  # Default to .bashrc
    if os.path.exists(os.path.expanduser('~/.zshrc')):
        shell_profile_path = os.path.expanduser('~/.zshrc')  # Use .zshrc if it exists
    # Command to append TZ=UTC export to the shell profile
    append_tz_command = f'echo "export TZ=UTC" >> {shell_profile_path}'
    # Check if TZ=UTC is already in the profile to avoid duplicates
    try:
        with open(shell_profile_path, 'r') as shell_profile:
            if 'export TZ=UTC' not in shell_profile.read():
                # Append the export command to the profile
                subprocess.run(append_tz_command, shell=True)
    except Exception as e:
        logger.error(f"An error occurred while setting timezone: {e}")    
    
def setup_swiss_army_llama(security_token):
    set_timezone_utc()
    swiss_army_llama_path = os.path.expanduser("~/swiss_army_llama")
    if not os.path.exists(swiss_army_llama_path):
        logger.info("Changing to the home directory.")
        os.chdir(os.path.expanduser("~"))
        logger.info("Cloning the Swiss Army Llama repository.")
        command = "git clone https://github.com/Dicklesworthstone/swiss_army_llama"
        logger.info(f"Now running command: {command}")
        subprocess.run(command, shell=True, executable="/bin/bash")
    else:
        logger.info("Swiss Army Llama repository already exists.")
        logger.info("Pulling the latest changes from the repository.")
        os.chdir(swiss_army_llama_path)
        command = "git stash && git pull"
        logger.info(f"Now running command: {command}")
        subprocess.run(command, shell=True, executable="/bin/bash")
    logger.info("Updating the Swiss Army Llama code file with the provided security token.")
    swiss_army_llama_file_path = os.path.join(swiss_army_llama_path, "swiss_army_llama.py")
    with open(swiss_army_llama_file_path, "r") as file:
        code_content = file.read()
    code_content = re.sub(r'SECURITY_TOKEN\s*=\s*"[^"]+"', f'SECURITY_TOKEN = "{security_token}"', code_content)
    code_content = re.sub(r'use_hardcoded_security_token\s*=\s*\d+', 'use_hardcoded_security_token = 1', code_content)
    with open(swiss_army_llama_file_path, "w") as file:
        file.write(code_content)
    logger.info("Checking for pyenv installation.")
    if not is_pyenv_installed():
        logger.info("pyenv is not installed. Installing pyenv.")
        apt_get_install = "sudo apt-get update && sudo apt-get install -y build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python3-openssl git redis redis-tools"
        run_command(apt_get_install)
        logger.info("Installing pyenv.")
        pyenv_clone = "git clone https://github.com/pyenv/pyenv.git ~/.pyenv"
        pyenv_dir = os.path.expanduser("~/.pyenv")
        if not os.path.exists(pyenv_dir):
            run_command(pyenv_clone)
        else:
            logger.info("~/.pyenv already exists. Skipping clone.")
        shell_rc_path = os.path.expanduser("~/.zshrc") if os.path.exists(os.path.expanduser("~/.zshrc")) else os.path.expanduser("~/.bashrc")
        with open(shell_rc_path, "a") as shell_rc:
            shell_rc.write('\nexport PYENV_ROOT="$HOME/.pyenv"\n')
            shell_rc.write('export PATH="$PYENV_ROOT/bin:$PATH"\n')
            shell_rc.write('eval "$(pyenv init --path)"\n')
        # Manually update os.environ for subsequent commands in this script
        os.environ["PYENV_ROOT"] = os.path.expanduser("~/.pyenv")
        os.environ["PATH"] = f"{os.path.expanduser('~/.pyenv/bin')}:{os.environ.get('PATH', '')}"
        logger.info(f"Added pyenv initialization to {shell_rc_path}. Attempting to source it.")
    else:
        logger.info("pyenv is already installed.")
    if not is_python_3_12_installed():
        logger.info("Python 3.12 is not installed. Installing the latest version of Python 3.12.")
        commands = [
            "pyenv install 3.12",
            "pyenv local 3.12"
        ]
        for command in commands:
            logger.info(f"Now running command: {command}")
            subprocess.run(command, shell=True, executable="/bin/bash")
    else:
        logger.info("Python 3.12 is already installed.")
    logger.info("Creating and activating virtual environment in ~/swiss_army_llama.")
    os.chdir(swiss_army_llama_path)
    # Initialize the commands list with Python environment setup commands
    commands = [
        "python -m venv venv",
        "source venv/bin/activate",
        "pip install --upgrade pip",
        "pip install wheel",
        "pip install -r requirements.txt"
    ]
    if not is_rust_installed():
        logger.info("Rust is not installed. Installing Rust.")
        # Prepend Rust installation and setup commands to the commands list
        rust_commands = [
            "curl https://sh.rustup.rs -sSf | sh -s -- -y",  # Install Rust non-interactively
            "source $HOME/.cargo/env",  # Ensure Rust environment is set up for subsequent commands
            "rustup default nightly",
            "rustup update nightly"
        ]
        commands = rust_commands + commands  # Combine Rust commands with the rest
    for command in commands:
        logger.info(f"Now running command: {command}")
        if command.startswith("source"):
            command_to_run = command.replace("source", ".", 1)  # Replace 'source' with '.' for bash compatibility
            subprocess.run(command_to_run, shell=True, executable="/bin/bash", env={"PATH": f"{swiss_army_llama_path}/venv/bin:{os.environ['PATH']}"})
        else:
            subprocess.run(command, shell=True, executable="/bin/bash", env={"PATH": f"{swiss_army_llama_path}/venv/bin:{os.environ['PATH']}"})
    logger.info("Running Swiss Army Llama...")
    command = "python swiss_army_llama.py"
    logger.info(f"Now running command: {command}")
    process = subprocess.Popen(
        command,
        shell=True,
        executable="/bin/bash",
        env={"PATH": f"{swiss_army_llama_path}/venv/bin:{os.environ['PATH']}"},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    external_ip = get_external_ip_func()
    logger.info(f"Setup complete. Open a browser to {external_ip}:8089 to get to the FastAPI Swagger page. Note that your security token is {security_token}.")
    # Read and log the output of the Swiss Army Llama process
    for line in process.stdout:
        logger.info(f"Swiss Army Llama: {line.strip()}")
    # Read and log any error output of the Swiss Army Llama process
    for line in process.stderr:
        logger.error(f"Swiss Army Llama Error: {line.strip()}")
    # Wait for the Swiss Army Llama process to finish (if needed)
    process.wait()        

def check_and_setup_swiss_army_llama(security_token):
    swiss_army_llama_port = 8089
    if is_swiss_army_llama_responding(swiss_army_llama_port, security_token):
        logger.info("Swiss Army Llama is already set up and running.")
    else:
        if not is_port_available(swiss_army_llama_port):
            logger.error(f"Port {swiss_army_llama_port} is not available. Please ensure that Swiss Army Llama is not running on another process.")
            return
        setup_swiss_army_llama(security_token)