import logging
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d_%H_%M_%S')}.log"
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)
os.makedirs(logs_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

logging.basicConfig(filename=LOG_FILE_PATH, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)


if __name__ == "__main__":
    logging.info("This is a test info message")
    logging.warning("This is a test warning message")
    logging.error("This is a test error message")
    logging.critical("This is a test critical message")
