import os
import subprocess
import httpx
import re
from logger_config import logger

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
            logger.warning("Failed to retrieve external IP address from {}: {}".format(provider, e))
    logger.warning("Failed to retrieve external IP address from all providers.")
    return "Unknown"

def run_command(command, env=None, capture_output=False, check=False, timeout=None):
    shell = '/bin/zsh' if os.path.exists('/bin/zsh') else '/bin/bash'
    full_env = {**os.environ, **env} if env else os.environ.copy()
    command = ' '.join(command) if isinstance(command, list) else command
    try:
        result = subprocess.run(command, shell=True, env=full_env, capture_output=capture_output, text=True, executable=shell, check=check, timeout=timeout)
        if capture_output:
            if result.stdout:
                logger.info(result.stdout)
            if result.stderr:
                logger.error(result.stderr)
        return result
    except subprocess.TimeoutExpired:
        logger.warning("Command '{}' timed out after {} seconds".format(command, timeout))
    except subprocess.CalledProcessError as e:
        logger.error("Command '{}' failed with exit code {}".format(command, e.returncode))
        if capture_output:
            logger.error(e.output)
        raise

def is_port_available(port):
    result = run_command(["lsof", "-i", ":{}".format(port)], capture_output=True)
    return result.returncode != 0

def is_swiss_army_llama_responding(external_ip, port, security_token):
    try:
        url = "http://{}:{}/get_list_of_available_model_names/".format(external_ip, port)
        params = {'token': security_token}
        response = httpx.get(url, params=params)
        return response.status_code == 200
    except Exception as e:
        print("Error: {}".format(e))
        return False

def update_security_token(file_path, token):
    with open(file_path, 'r+') as file:
        content = file.read()
        file.seek(0)
        file.truncate()
        content = re.sub(r'SECURITY_TOKEN\s*=\s*"[^"]*"', 'SECURITY_TOKEN = "{}"'.format(token), content)
        file.write(content)

def is_pyenv_installed():
    result = run_command(["pyenv --version"], capture_output=True)
    return result.returncode == 0

def is_python_3_12_installed():
    result = run_command(["pyenv versions"], capture_output=True)
    return "3.12" in result.stdout

def is_rust_installed():
    try:
        result = run_command(["rustc", "--version"], capture_output=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def setup_virtual_environment(swiss_army_llama_path):
    venv_path = os.path.join(swiss_army_llama_path, 'venv')
    if not os.path.exists(venv_path):
        os.makedirs(venv_path)
        run_command(['python3 -m venv {}'.format(venv_path)], check=True)
    pip_executable = os.path.join(venv_path, 'bin', 'pip')
    run_command(['{} install --upgrade pip'.format(pip_executable)], check=True)
    run_command(['{} install wheel'.format(pip_executable)], check=True)
    run_command(['{} install -r {}/requirements.txt'.format(pip_executable, swiss_army_llama_path)], check=True)
    return os.path.join(venv_path, 'bin', 'python')

def set_timezone_utc():
    os.environ['TZ'] = 'UTC'
    shell_profile_path = os.path.expanduser('~/.zshrc') if os.path.exists(os.path.expanduser('~/.zshrc')) else os.path.expanduser('~/.bashrc')
    if 'export TZ=UTC' not in open(shell_profile_path, 'r').read():
        run_command(['echo "export TZ=UTC" >> {}'.format(shell_profile_path)])

def check_systemd_service_exists(service_name):
    result = run_command("systemctl is-enabled {}".format(service_name), capture_output=True)
    return result.returncode == 0 and 'enabled' in result.stdout

def create_systemd_service(service_name, user, working_directory, exec_start):
    service_content = """[Unit]
Description=Swiss Army Llama service
After=network.target

[Service]
Type=simple
User={}
WorkingDirectory={}
ExecStart={}
Restart=always

[Install]
WantedBy=multi-user.target
""".format(user, working_directory, exec_start)
    service_path = "/etc/systemd/system/{}.service".format(service_name)
    temp_service_path = "/tmp/{}.service".format(service_name)
    with open(temp_service_path, 'w') as file:
        file.write(service_content)
    run_command("sudo mv {} {}".format(temp_service_path, service_path), check=True)
    logger.info("Systemd service file created at {}".format(service_path))
    run_command("sudo systemctl daemon-reload", check=True)
    run_command("sudo systemctl enable {}".format(service_name), check=True)
    run_command("sudo systemctl start {}".format(service_name), check=True)
    status_output = run_command("sudo systemctl status {}".format(service_name), capture_output=True, timeout=5)
    logger.info("Status of {} service:\n{}".format(service_name, status_output.stdout))

def ensure_pyenv_setup():
    if not is_pyenv_installed():
        logger.info("Installing pyenv...")
        run_command(["sudo apt-get update && sudo apt-get install -y build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python3-openssl git redis redis-tools"])
        run_command(["curl https://pyenv.run | bash"])
    if not is_python_3_12_installed():
        logger.info("Installing Python 3.12 using pyenv...")
        run_command(["pyenv install 3.12"])
        run_command(["pyenv global 3.12"])

def configure_shell_for_pyenv():
    shell_rc_path = os.path.expanduser("~/.zshrc") if os.path.exists(os.path.expanduser("~/.zshrc")) else os.path.expanduser("~/.bashrc")
    pyenv_init_str = 'export PYENV_ROOT="$HOME/.pyenv"\nexport PATH="$PYENV_ROOT/bin:$PATH"\neval "$(pyenv init --path)"\n'
    if pyenv_init_str not in open(shell_rc_path).read():
        with open(shell_rc_path, "a") as shell_rc:
            shell_rc.write(pyenv_init_str)
    os.environ["PYENV_ROOT"] = os.path.expanduser("~/.pyenv")
    os.environ["PATH"] = "{}:{}/bin".format(os.environ['PYENV_ROOT'], os.environ.get('PATH', ''))

def has_repo_been_updated(repo_path):
    try:
        run_command("git -C {} fetch".format(repo_path), check=True)
        local_commit = run_command("git -C {} rev-parse HEAD".format(repo_path), capture_output=True).stdout.strip()
        remote_commit = run_command("git -C {} rev-parse origin/main".format(repo_path), capture_output=True).stdout.strip()
        return local_commit != remote_commit
    except Exception as e:
        logger.error("Error checking repository updates: {}".format(e))
        return False

def setup_swiss_army_llama(security_token):
    set_timezone_utc()
    swiss_army_llama_path = os.path.expanduser("~/swiss_army_llama")
    swiss_army_llama_script = os.path.join(swiss_army_llama_path, "swiss_army_llama.py")
    if not os.path.exists(swiss_army_llama_path):
        logger.info("Cloning the Swiss Army Llama repository.")
        run_command("git clone https://github.com/Dicklesworthstone/swiss_army_llama {}".format(swiss_army_llama_path), check=True)
    else:
        logger.info("Swiss Army Llama repository already exists.")
    logger.info("Checking for updates to the Swiss Army Llama repository.")
    run_command("git -C {} fetch".format(swiss_army_llama_path), check=True)
    local_commit = run_command("git -C {} rev-parse HEAD".format(swiss_army_llama_path), capture_output=True).stdout.strip()
    remote_commit = run_command("git -C {} rev-parse origin/main".format(swiss_army_llama_path), capture_output=True).stdout.strip()
    if local_commit != remote_commit:
        logger.info("Updates found. Stashing local changes and pulling new changes from the repository.")
        run_command("git -C {} stash".format(swiss_army_llama_path), check=True)
        run_command("git -C {} pull".format(swiss_army_llama_path), check=True)
        update_security_token(swiss_army_llama_script, security_token)
    configure_shell_for_pyenv()
    if not is_pyenv_installed():
        ensure_pyenv_setup()
    if not is_python_3_12_installed():
        logger.info("Python 3.12 is not installed. Installing Python 3.12 using pyenv.")
        run_command("pyenv install 3.12", check=True)
        run_command("pyenv global 3.12", check=True)
    venv_path = os.path.join(swiss_army_llama_path, 'venv')
    if not os.path.exists(venv_path):
        python_executable = setup_virtual_environment(swiss_army_llama_path)
    else:
        python_executable = os.path.join(venv_path, 'bin', 'python')
        run_command(['pip install --upgrade -r {}/requirements.txt'.format(swiss_army_llama_path)], check=True) 
    if not is_rust_installed():
        logger.info("Rust is not installed. Installing Rust.")
        run_command("curl https://sh.rustup.rs -sSf | sh -s -- -y", check=True)
        os.environ.update({
            'PATH': "{}/.cargo/bin:{}".format(os.environ.get('HOME'), os.environ.get('PATH'))
        })
        run_command("rustup default nightly && rustup update nightly", check=True)
    if not check_systemd_service_exists("swiss_army_llama"):
        create_systemd_service("swiss_army_llama", os.getlogin(), swiss_army_llama_path, "{} {}".format(python_executable, swiss_army_llama_script))
    else:
        logger.info("Swiss Army Llama systemd service already exists; skipping installation, reloading systemd, and starting/enabling the service.")
        run_command("sudo systemctl daemon-reload", check=True)
        run_command("sudo systemctl enable swiss_army_llama", check=True)
        run_command("sudo systemctl start swiss_army_llama", check=True)
        status_output = run_command("sudo systemctl status swiss_army_llama", capture_output=True, timeout=5)
        logger.info("Status of swiss_army_llama service:\n{}".format(status_output.stdout))

def kill_running_instances_of_swiss_army_llama():
    logger.info("Stopping Swiss Army Llama service...")
    run_command("sudo systemctl stop swiss_army_llama", check=False)
    logger.info("Killing any remaining Swiss Army Llama processes...")
    run_command("ps -ef | grep 'swiss_army' | grep -v grep | awk '{print $2}' | xargs -r kill -9", check=False)

def check_and_setup_swiss_army_llama(security_token):
    swiss_army_llama_port = 8089
    remote_swiss_army_llama_mapped_port = 8087  # Local port for remote SSH tunnel (not used here)   # noqa: F841
    external_ip = get_external_ip_func()
    if external_ip == "Unknown":
        logger.error("Unable to reach external network providers. Network may be unreachable.")
        return
    repo_path = os.path.expanduser("~/swiss_army_llama")
    repo_updated = has_repo_been_updated(repo_path)
    service_responding = is_swiss_army_llama_responding("localhost", swiss_army_llama_port, security_token)
    swiss_army_llama_port_available = is_port_available(swiss_army_llama_port)
    if service_responding:
        logger.info("Swiss Army Llama service is responding on port {}.".format(swiss_army_llama_port))
        if not repo_updated:
            logger.info("Swiss Army Llama is already set up and running, and the repository has not been updated.")
            return
    elif not swiss_army_llama_port_available:
        logger.error("Swiss Army Llama service is NOT responding on port {}.".format(swiss_army_llama_port))
        logger.error("Port {} is not available. It may be in use by another process.".format(swiss_army_llama_port))
    if repo_updated or not service_responding:
        if repo_updated:
            logger.info("Repository has updates. Running setup.")
        else:
            logger.info("Repository has not been updated, but the service is not responding. Running setup.")
        kill_running_instances_of_swiss_army_llama()  # Ensure service is stopped before setup
        setup_swiss_army_llama(security_token)
