import picologging as logging
import os
import shutil
import queue
from picologging.handlers import RotatingFileHandler, QueueHandler, QueueListener

logger = logging.getLogger("pastel_supernode_inference_layer")

def setup_logger():
    if logger.handlers:
        return logger
    old_logs_dir = 'old_logs'
    if not os.path.exists(old_logs_dir):
        os.makedirs(old_logs_dir)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    log_file_path = 'pastel_supernode_inference_layer.log'
    log_queue = queue.Queue(-1)
    fh = RotatingFileHandler(log_file_path, maxBytes=10*1024*1024, backupCount=5)
    fh.setFormatter(formatter)
    def namer(default_log_name):
        return os.path.join(old_logs_dir, os.path.basename(default_log_name))
    def rotator(source, dest):
        shutil.move(source, dest)
    fh.namer = namer
    fh.rotator = rotator
    queue_handler = QueueHandler(log_queue)
    logger.addHandler(queue_handler)
    listener = QueueListener(log_queue, fh)
    listener.start()
    logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)
    return logger
