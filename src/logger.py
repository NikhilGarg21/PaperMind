import logging
import os

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

def get_logger(name : str) -> logging.Logger:
    """Get a logger instance with the specified name."""
    try:

        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)

        file_handler = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
        file_handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        if not logger.handlers:
            logger.addHandler(console_handler)
            logger.addHandler(file_handler)

        return logger
    
    except Exception as e:
        print(f"Error setting up logger '{name}': {e}")
        raise