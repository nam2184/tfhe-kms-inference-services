import time
import requests
import os

headers = {
    'Content-Type': 'application/json; charset=utf-8'
}

LOCAL_SERVER_ENDPOINT = "http://127.0.0.1:5000"
REMOTE_SERVER_ENDPOINT = "https://kms.netty"

# Define the cookies to be sent (like session or auth tokens)
cookies = {
    'session_id': 'your_session_id'  # Replace with actual session cookie
}

class Dir:
    def __init__(self, name):
        self.name = os.getcwd() + name
        if not os.path.exists(self.name):
            os.makedirs(self.name)

class Network:
    """Simulate a network on disk."""

    def __init__(self):
        self.dev_dir = Dir("/dev") # pylint: disable=consider-using-with
        self.server_dir = Dir("/server") # pylint: disable=consider-using-with
        self.log_dir = Dir("/logs") # pylint: disable=consider-using-with
        self.local_kms_endpoint = LOCAL_SERVER_ENDPOINT
        self.remote_kms_endpoint = REMOTE_SERVER_ENDPOINT
