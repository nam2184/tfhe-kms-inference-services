import os
import time
import base64
import json
import logging
import tempfile
from multiprocessing import Process
from dotenv import load_dotenv
import requests
import shutil

from flask import Flask, jsonify, send_file
from flask_cors import CORS, cross_origin
from flask_smorest import Api, Blueprint
from flask_smorest.blueprint import MethodView

from concrete.ml.deployment import FHEModelServer
from db_server import HomomorphicDBService, HomomorphicKeyModel
from models import (
    EncryptedBodyMessageSchema,
    EncryptedMessageSchema,
    ErrorTypeSchema,
    PostKeyBodySchema,
)
from network import Network
import util


class InferenceService:
    def __init__(self, host="127.0.0.1", port=5001, debug=True, local=False):
        load_dotenv()
        self.host = host
        self.port = port
        self.debug = debug
        self.local = local
        self.network = Network()

        self.app = Flask("inference")
        self.app.config.update(
            API_TITLE="Model Inference API",
            API_VERSION="1.0",
            OPENAPI_VERSION="3.0.0",
        )

        self.api = Api(self.app)
        CORS(self.app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)
        self._setup_logging()
        self._setup_routes()
        self._setup_signal_handlers()

    # ------------------ Setup Methods ------------------ #
    def _setup_logging(self):
        log_path = os.path.join(self.network.log_dir.name, "server.log")
        util.setup_logging(log_path, "werkzeug")
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
        )
        self.logger = logging.getLogger("InferenceService")
        self.logger.info("Logging initialized.")

    def _setup_signal_handlers(self):
        import signal, sys

        def _graceful_exit(signum, frame):
            self.logger.info(f"Received signal {signum}, shutting down gracefully...")
            sys.exit(0)

        signal.signal(signal.SIGINT, _graceful_exit)
        signal.signal(signal.SIGTERM, _graceful_exit)

    def _setup_routes(self):
        blp = Blueprint("model", "model", url_prefix="/", description="Model Inference Endpoints")

        # ---- /key ---- #
        @cross_origin(supports_credentials=True)
        @blp.route("/key")
        class Key(MethodView):
            @blp.doc(description="Upload keys for a user.")
            @blp.arguments(PostKeyBodySchema)
            @blp.response(200)
            @blp.alt_response(status_code=400, schema=ErrorTypeSchema)
            def post(inner_self, data):
                return self._handle_key_upload(data)

        # ---- /key/<id> ---- #
        @cross_origin(supports_credentials=True)
        @blp.route("/key/<int:chat_id>")
        class KeyCheck(MethodView):
            @blp.doc(description="Check if a user's key exists.")
            @blp.response(200)
            @blp.alt_response(status_code=400, schema=ErrorTypeSchema)
            def get(inner_self, chat_id):
                return self._handle_key_check(chat_id)

        # ---- /predict ---- #
        @cross_origin(supports_credentials=True)
        @blp.route("/predict")
        class Predict(MethodView):
            @blp.doc(description="Run encrypted prediction.")
            @blp.arguments(EncryptedBodyMessageSchema)
            @blp.response(200, EncryptedMessageSchema)
            @blp.alt_response(status_code=204)
            @blp.alt_response(status_code=400, schema=ErrorTypeSchema)
            def post(inner_self, data):
                return self._handle_predict(data)

        self.api.register_blueprint(blp)

    # ------------------ Utility Methods ------------------ #
    def _download_server_zip(self, endpoint_url: str):
        server_dir_path = self.network.server_dir.name
        if any(os.scandir(server_dir_path)):
            self.logger.info("Server directory already contains files. Skipping download.")
            return

        self.logger.info(f"Downloading server.zip from {endpoint_url}")
        response = requests.get(
            endpoint_url,
            headers={"Accept": "application/octet-stream"},
            cookies={"session_id": "1"},
            stream=True  
        )        
        if response.status_code == 200:
            file_path = os.path.join(server_dir_path, "server.zip")
            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            self.logger.info(f"Saved server.zip to {file_path}")
        else:
            self.logger.error(f"Failed to download server.zip ({response.status_code})")
            raise RuntimeError(f"Download failed: {response.text}")
        time.sleep(5)

    def generate_server_configs(self):
        endpoint = self.network.local_kms_endpoint if self.local else self.network.remote_kms_endpoint
        self._download_server_zip(f"{endpoint}/model")

    # ------------------ Endpoint Handlers ------------------ #
    def _handle_key_upload(self, data):
        try:
            encoded_file = data["file"]
            base64.b64decode(encoded_file)
        except Exception:
            return jsonify({"status": "error", "message": "Invalid base64"}), 400

        db = HomomorphicDBService()
        chat_id = data["chat_id"]

        if db.homomorphic_key_exists(chat_id):
            self.logger.info(f"Key already exists for chat_id={chat_id}")
            return "", 200

        key_record = HomomorphicKeyModel(file=encoded_file, chat_id=chat_id)
        db.insert_homomorphic_key(key_record)
        self.logger.info(f"Inserted new homomorphic key for chat_id={chat_id}")
        return "", 200

    def _handle_key_check(self, chat_id):
        db = HomomorphicDBService()
        if not db.homomorphic_key_exists(chat_id):
            return jsonify({"section": "error", "message": "No key found"}), 400
        return "", 200

    def _handle_predict(self, data):
        try:
            chat_id = data.get("chat_id")
            self.logger.info(f"/predict called for chat_id={chat_id}")
            key_service = HomomorphicDBService()
            key = key_service.get_homomorphic_key_by_chat_id(chat_id)

            if key is None:
                self.logger.warning(f"No homomorphic key found for chat_id={chat_id}")
                return "", 204

            serialized_evaluation_keys = base64.b64decode(key.file)
            image_to_classify = base64.b64decode(data.get("image_to_classify", ""))

            start_time = time.time()
            self.logger.info(f"Starting FHE prediction for chat_id={chat_id}")

            fhe_server = FHEModelServer(self.network.server_dir.name)
            encrypted_prediction = fhe_server.run(image_to_classify, serialized_evaluation_keys)

            duration = time.time() - start_time
            self.logger.info(
                f"FHE prediction completed for chat_id={chat_id} "
                f"(duration={duration:.3f}s, output={len(encrypted_prediction)} bytes)"
            )

            encoded_prediction = base64.b64encode(encrypted_prediction).decode("utf-8")
            data["classification_result"] = encoded_prediction
            return data, 200

        except Exception as e:
            self.logger.error(f"Exception in /predict: {e}")
            return jsonify({"section": "predict", "message": str(e)}), 400

    # ------------------ Runtime ------------------ #
    def save_openapi_spec(self, path="inference-api.json"):
        with self.app.app_context():
            spec_dict = self.api.spec.to_dict()
            with open(path, "w") as f:
                json.dump(spec_dict, f, indent=2)
            self.logger.info(f"OpenAPI spec saved to {path}")

    def run(self):
        self.logger.info(f"Starting InferenceService on {self.host}:{self.port}")
        HomomorphicDBService()  # Initialize DB
        self.generate_server_configs()
        self.app.run(host=self.host, port=self.port, debug=self.debug, use_reloader=False)

    def run_background(self):
        p = Process(target=self.run)
        p.start()
        self.logger.info(f"InferenceService running in background (pid={p.pid})")
        return p


if __name__ == "__main__":
    service = InferenceService(local=False)
    service.save_openapi_spec()
    service.run()

