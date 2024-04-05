import os
import subprocess
import httpx
import re
from logger_config import setup_logger
logger = setup_logger()

def get_external_ip_func():
    response = httpx.get("https://ipinfo.io/ip")
    response.raise_for_status()
    return response.text.strip()

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
    
def is_pyenv_installed():
    try:
        result = subprocess.run(["pyenv", "--version"], capture_output=True)
        return result.returncode == 0
    except:  # noqa: E722
        return False

def is_python_3_12_installed():
    try:
        result = subprocess.run(["pyenv", "versions", "--bare"], capture_output=True, text=True)
        return "3.12" in result.stdout
    except:  # noqa: E722
        return False

def setup_swiss_army_llama(security_token):
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
        command = "git pull"
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

    if not is_pyenv_installed():
        logger.info("pyenv is not installed. Installing pyenv.")
        commands = [
            "sudo apt-get update",
            "sudo apt-get install -y build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python3-openssl git",
            "git clone https://github.com/pyenv/pyenv.git ~/.pyenv"
        ]
        shell_rc_path = os.path.expanduser("~/.zshrc") if os.path.exists(os.path.expanduser("~/.zshrc")) else os.path.expanduser("~/.bashrc")
        pyenv_init_commands = [
            'echo \'export PYENV_ROOT="$HOME/.pyenv"\' >> {}'.format(shell_rc_path),
            'echo \'export PATH="$PYENV_ROOT/bin:$PATH"\' >> {}'.format(shell_rc_path),
            'echo \'eval "$(pyenv init --path)"\' >> {}'.format(shell_rc_path),
            "source {}".format(shell_rc_path)
        ]
        commands += pyenv_init_commands
        for command in commands:
            logger.info(f"Now running command: {command}")
            subprocess.run(command, shell=True, executable="/bin/bash")
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
    commands = [
        "python -m venv venv",
        "source venv/bin/activate",
        "pip install --upgrade pip",
        "pip install wheel",
        "pip install -r requirements.txt"
    ]
    for command in commands:
        logger.info(f"Now running command: {command}")
        subprocess.run(command, shell=True, executable="/bin/bash", env={"PATH": f"{swiss_army_llama_path}/venv/bin:{os.environ['PATH']}"})

    logger.info("Running Swiss Army Llama.")
    command = "python swiss_army_llama.py"
    logger.info(f"Now running command: {command}")
    subprocess.run(command, shell=True, executable="/bin/bash", env={"PATH": f"{swiss_army_llama_path}/venv/bin:{os.environ['PATH']}"})

    external_ip = get_external_ip_func()
    logger.info(f"Setup complete. Open a browser to {external_ip}:8089 to get to the FastAPI Swagger page. Note that your security token is {security_token}.")

def check_and_setup_swiss_army_llama(security_token):
    swiss_army_llama_port = 8089
    if is_swiss_army_llama_responding(swiss_army_llama_port, security_token):
        logger.info("Swiss Army Llama is already set up and running.")
    else:
        if not is_port_available(swiss_army_llama_port):
            logger.error(f"Port {swiss_army_llama_port} is not available. Please ensure that Swiss Army Llama is not running on another process.")
            return

        setup_swiss_army_llama(security_token)