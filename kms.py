import os
import base64
import json
import zipfile
import signal
import sys
import tempfile
import logging
import shutil
from multiprocessing import Process

from dotenv import load_dotenv
from flask import Flask, jsonify, send_file
from flask_cors import CORS, cross_origin
from flask_smorest import Api, Blueprint
from flask_smorest.blueprint import MethodView
from Crypto.Random import get_random_bytes
import numpy as np

# Concrete ML imports
from concrete.ml.deployment import FHEModelDev, FHEModelClient
from concrete.ml.torch.compile import compile_brevitas_qat_model, torch

# Internal modules
from db_kms import KeysDBService, KeyModel, ClientHEModel
from ml.ml import CNN
from ml import data
from network import Network
from models import ErrorTypeSchema, GetClientSchema


class KMSService:
    def __init__(self, host="127.0.0.1", port=5000, debug=True):
        load_dotenv()
        self.host = host
        self.port = port
        self.debug = debug

        self.network = Network()
        self.q_module = None

        self.app = Flask("kms")
        self.app.config.update(
            API_TITLE="Key Manager API",
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
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
        )
        self.logger = logging.getLogger("KMSService")

    def _setup_signal_handlers(self):
        signal.signal(signal.SIGINT, self._graceful_exit)
        signal.signal(signal.SIGTERM, self._graceful_exit)

    def _graceful_exit(self, signum, frame):
        self.logger.info(f"Received signal {signum}, shutting down gracefully...")
        sys.exit(0)

    def _setup_routes(self):
        blp = Blueprint("kms", "kms", url_prefix="/", description="KMS Endpoints")

        # ---- /client/<id> ---- #
        @cross_origin(supports_credentials=True)
        @blp.route("/client/<int:chat_id>")
        class Client(MethodView):
            @blp.doc(description="Get FHE client.")
            @blp.response(status_code=200, schema=GetClientSchema)
            @blp.alt_response(status_code=400, schema=ErrorTypeSchema)
            def get(inner_self, chat_id):
                return self._handle_client_request(chat_id)

        # ---- /keys/<id> ---- #
        @cross_origin(supports_credentials=True)
        @blp.route("/keys/<int:chat_id>")
        class Keys(MethodView):
            @blp.doc(description="Get normal keys.")
            @blp.alt_response(status_code=400, schema=ErrorTypeSchema)
            def get(inner_self, chat_id):
                return self._handle_keys_request(chat_id)

        # ---- /model ---- #
        @cross_origin(supports_credentials=True)
        @blp.route("/model")
        class Model(MethodView):
            @blp.doc(description="Get FHE model.")
            @blp.alt_response(status_code=400, schema=ErrorTypeSchema)
            def get(inner_self):
                return self._handle_model_request()

        self.api.register_blueprint(blp)

    # ------------------ Core Logic ------------------ #
    def _generate_dev_configs(self, model):
        base_dir = self.network.dev_dir.name
        self.logger.info(f"Cleaning dev_dir: {base_dir}")

        for filename in os.listdir(base_dir):
            path = os.path.join(base_dir, filename)
            if os.path.isfile(path) or os.path.islink(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)

        fhe = FHEModelDev(base_dir, model=model)
        fhe.save()
        self.logger.info(f"Saved new FHEModelDev config for model={model}")

    def _generate_temp_keys(self, chat_id: int):
        temp_dir = tempfile.mkdtemp()
        aes_key_path = os.path.join(temp_dir, f"aes_key{chat_id}.bin")

        aes_key = get_random_bytes(32)
        with open(aes_key_path, "wb") as f:
            f.write(aes_key)

        zip_path = os.path.join(temp_dir, f"normal_keys{chat_id}.zip")
        with zipfile.ZipFile(zip_path, "w") as zipf:
            zipf.write(aes_key_path, arcname=f"aes_key{chat_id}.bin")

        return zip_path, temp_dir

    # ------------------ Endpoint Handlers ------------------ #
    def _handle_client_request(self, chat_id):
        self.logger.info(f"Handling /client/{chat_id}")
        db = KeysDBService()
        record = db.get_heclient_key_by_chat_id(chat_id)

        base_dir = self.network.dev_dir.name
        resp = {}

        if record is None:
            self._generate_dev_configs(self.q_module)
            client_zip_path = os.path.join(base_dir, "client.zip")

            with open(client_zip_path, "rb") as f:
                encoded_client_zip = base64.b64encode(f.read()).decode("utf-8")
            resp["client_specs"] = encoded_client_zip

            client_keys_dir = os.path.join(base_dir, f"keys{chat_id}.bin")
            client = FHEModelClient(base_dir)
            client.generate_private_and_evaluation_keys(True)
            client.client.keys.save(client_keys_dir)

            with open(client_keys_dir, "rb") as f:
                encoded_zip = base64.b64encode(f.read()).decode("utf-8")
            resp["keys"] = encoded_zip

            client_specs = ClientHEModel(file=encoded_zip, client_specs=encoded_client_zip, chat_id=chat_id)
            db.insert_heclient_key(client_specs)

        else:
            resp = {
                "keys": record.file,
                "client_specs": record.client_specs,
            }

        return resp

    def _handle_keys_request(self, chat_id):
        db = KeysDBService()
        record = db.get_key_by_chat_id(chat_id)

        if record is None:
            zip_path, temp_dir = self._generate_temp_keys(chat_id)
            with open(zip_path, "rb") as f:
                encoded_zip = base64.b64encode(f.read()).decode("utf-8")
            db.insert_key(KeyModel(file=encoded_zip, chat_id=chat_id))
        else:
            encoded_zip = record.file
            decoded = base64.b64decode(encoded_zip)
            temp_dir = tempfile.mkdtemp()
            zip_path = os.path.join(temp_dir, "keys.zip")
            with open(zip_path, "wb") as f:
                f.write(decoded)

        response = send_file(
            zip_path,
            mimetype="application/zip",
            as_attachment=True,
            download_name=f"normal_keys{chat_id}.zip"
        )

        @response.call_on_close
        def cleanup():
            shutil.rmtree(temp_dir, ignore_errors=True)
            self.logger.info("Temporary keys deleted.")

        return response

    def _handle_model_request(self):
        file_path = os.path.join(self.network.dev_dir.name, "server.zip")
        if os.path.exists(file_path):
            return send_file(
                file_path,
                mimetype="application/zip",
                as_attachment=True,
                download_name="server.zip"
            )
        return {"section": "model", "message": "File not found"}, 404

    # ------------------ FHE/ML Setup ------------------ #
    def setup_he_module(self):
        self.logger.info("Setting up homomorphic encryption module...")
        image_size = 16
        net = CNN(n_classes=2, in_channels=3, image_size=image_size)

        calibration_data = data.split_and_preprocess_calibration(
            os.getcwd() + "/dataset", n_samples=10, size=(image_size, image_size)
        )
        calibration_data = np.transpose(calibration_data, (0, 3, 1, 2))
        net.forward(torch.from_numpy(calibration_data))

        checkpoint = torch.load(os.getcwd() + "/ml/models/resnet_cnn_best4bits.pth")
        net.load_state_dict(checkpoint["classifier_state_dict"])

        self.logger.info("Compiling model for FHE deployment...")
        self.q_module = compile_brevitas_qat_model(
            net, calibration_data, n_bits=3, rounding_threshold_bits=4, p_error=0.01
        )

    def save_openapi_spec(self, path="kms-api.json"):
        with self.app.app_context():
            spec_dict = self.api.spec.to_dict()
            with open(path, "w") as f:
                json.dump(spec_dict, f, indent=2)
            self.logger.info(f"OpenAPI spec saved to {path}")

    def run(self):
        self.logger.info(f"Starting KMSService on {self.host}:{self.port}")
        self.app.run(host=self.host, port=self.port, debug=self.debug, use_reloader=False)

    def run_background(self):
        p = Process(target=self.run)
        p.start()
        self.logger.info(f"KMSService running in background process (pid={p.pid})")
        return p


if __name__ == "__main__":
    service = KMSService()
    service.save_openapi_spec()
    service.setup_he_module()
    KeysDBService()  # Ensure DB initialized
    service.run()

