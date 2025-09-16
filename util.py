import numpy as np
import logging
from PIL import Image
import psutil

def setup_logging(log_file_path, logger_name):
    """
    Set up logging to file and attach it to the specified logger.
    
    Args:
        log_file_path (str): Path to the log file where logs will be written.
        logger_name (str): The name of the logger to configure (e.g., 'werkzeug').
    
    Returns:
        Logger: Configured logger with file logging enabled.
    """
    logger = logging.getLogger(logger_name)
    
    logger.handlers = []
    
    logger.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(log_file_path)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    
    return logger

def get_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss  # Resident Set Size (memory used by process)


def visual_encrypted_data(binary_data_str):
    try:
        binary_data = bytes.fromhex(binary_data_str.replace("\\x", "").replace("b'", "").replace("'", ""))

        data_length = len(binary_data)

        width = int(data_length ** 0.5)  # Calculate width
        height = (data_length // 3) // width  # Calculate height based on RGB

        required_length = width * height * 3

        if len(binary_data) < required_length:
            padded_data = binary_data + b'\x00' * (required_length - len(binary_data))
        else:
            padded_data = binary_data[:required_length]

        image_array = np.frombuffer(padded_data, dtype=np.uint8)

        image_array = image_array.reshape((height, width, 3))

        image = Image.fromarray(image_array, 'RGB')
        image.save("encrypted_visualization.png")  # Save as PNG

    except Exception as e:
        print("An error occurred:", e)


