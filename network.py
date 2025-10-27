import time
import requests
import os
from dotenv import load_dotenv

load_dotenv()

class Dir:
    def __init__(self, name: str):
        self.name = os.path.join(os.getcwd(), name.lstrip("/"))
        if not os.path.exists(self.name):
            os.makedirs(self.name, exist_ok=True)

class Network:
    """Simulate a network on disk."""

    def __init__(self):
        # Directories for network simulation
        self.dev_dir = Dir("/dev")
        self.server_dir = Dir("/server")
        self.log_dir = Dir("/logs")

        # Endpoints loaded from environment variables
        self.local_kms_endpoint = os.getenv("LOCAL_SERVER_ENDPOINT", "http://127.0.0.1:5000")
        self.remote_kms_endpoint = os.getenv("REMOTE_SERVER_ENDPOINT")

