import base64
from io import BytesIO
import json
from multiprocessing import Process
import time
from concrete.fhe.compilation.configuration import np
from flask import Flask, request, jsonify
import os
from sqlmodel import Session
from db_server import HomomorphicDBService, HomomorphicKeyModel 
from models import EncryptedBodyMessageSchema, EncryptedMessageSchema, ErrorTypeSchema, PostKeySchema, PostKeyBodySchema
import util
from flask_smorest import Api, Blueprint
from flask_smorest.blueprint import MethodView

from pandas.core.generic import pickle
from network import Network
import requests
from dotenv import load_dotenv
from flask_cors import CORS, cross_origin
from werkzeug.datastructures import FileStorage
from concrete.ml.deployment import FHEModelServer

load_dotenv()
network = Network()

app = Flask("model")
app.config["API_TITLE"] = "Model Inference API"
app.config["API_VERSION"] = "1.0"
app.config["OPENAPI_VERSION"] = "3.0.0"

api = Api(app)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)
blp = Blueprint("model", "model", url_prefix="/", description="Model Inference Endpoints")

headers = {
    'Content-Type': 'application/json; charset=utf-8'
}

cookies = {
    'session_id': '1'  # Replace with actual session cookie
}

def generate_server_configs_local(network : Network):
    print("configuring server")
    server_dir_path = network.server_dir.name
    if any(os.scandir(server_dir_path)):
        print("Server directory already contains files. Skipping download.")
        return

    url = f"{network.local_kms_endpoint}/model"
    response = requests.get(url, headers= headers, cookies = cookies)
    if response.status_code == 200:
        with open(os.path.join(server_dir_path, "server.zip"), "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    else:
        print("Download failed:", response.status_code)
    time.sleep(10)

def generate_server_configs(network : Network):
    print("configuring server")
    server_dir_path = network.server_dir.name
    if any(os.scandir(server_dir_path)):
        print("Server directory already contains files. Skipping download.")
        return

    url = f"{network.remote_kms_endpoint}/model"
    response = requests.get(url, headers= headers, cookies = cookies)
    if response.status_code == 200:
        with open(os.path.join(server_dir_path, "server.zip"), "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    else:
        print("Download failed:", response.status_code)
    time.sleep(10)



#Server
@cross_origin(supports_credentials=True) 
@blp.route('/key')
class Key(MethodView):
    @blp.doc(description="Post up keys for the user")
    @blp.arguments(PostKeyBodySchema)
    @blp.response(200, PostKeySchema)
    @blp.alt_response(status_code=400, schema=ErrorTypeSchema)
    def post(self, data):
        encoded_file = data["file"]
        try:
            base64.b64decode(encoded_file)
        except Exception:
            return jsonify({'status': 'error', 'message': "Invalid base64"}), 400
        
        key_record = HomomorphicKeyModel(
            file=encoded_file,
            chat_id=data["chat_id"]
        )
        key = HomomorphicDBService().insert_homomorphic_key(key_record)
        return key   

@cross_origin(supports_credentials=True) 
@blp.route('/key/<int:chat_id>')
class KeyCheck(MethodView):
    @blp.doc(description="Check for keys")
    @blp.response(200, PostKeySchema)
    @blp.alt_response(status_code=400, schema=ErrorTypeSchema)
    def get(self, chat_id):
        key = HomomorphicDBService().get_homomorphic_key_by_chat_id(chat_id)
        if not key:
            return jsonify({'status': 'error', 'message': "No key found"}), 400
        return key

@cross_origin(supports_credentials=True) 
@blp.route('/predict')
class Predict(MethodView):
    @blp.doc(description="Predict classification nsfw")
    @blp.arguments(EncryptedBodyMessageSchema)
    @blp.response(200, EncryptedMessageSchema)
    @blp.alt_response(status_code=204)
    @blp.alt_response(status_code=400, schema=ErrorTypeSchema)
    def post(self, data):
        # --- Fetch key from DB ---
        try :
            key = HomomorphicDBService().get_homomorphic_key_by_chat_id(data["chat_id"])
            if key is not None :
                serialized_evaluation_keys = base64.b64decode(key.file)
                image_to_classify = base64.b64decode(data["image_to_classify"])
                time_begin = time.time()
                encrypted_prediction = FHEModelServer(network.server_dir.name).run(
                    image_to_classify, serialized_evaluation_keys
                )
                time_end = time.time()
                print(f"Time take : {time_end - time_begin} seconds")
                encoded_str = base64.b64encode(encrypted_prediction).decode("utf-8")
                data["classification_result"] = encoded_str
                with open(network.server_dir.name + "/encrypted_prediction.enc", "wb") as f:
                    f.write(encrypted_prediction)
                return data, 200
            else:
                return "", 204
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 400

# --- Register endpoints ---
api.register_blueprint(blp)

# --- Save OpenAPI spec ---
def save_openapi_spec(app, output_path="inference-api.json"):
    with app.app_context():
        spec_dict = api.spec.to_dict()
        with open(output_path, "w") as f:
            json.dump(spec_dict, f, indent=2)
        print(f"OpenAPI spec written to {output_path}")

def run_flask_app(app, port):
    app.run(host='127.0.0.1', port=port, debug=True)

if __name__ == '__main__':
    save_openapi_spec(app)
    HomomorphicDBService()    
    log_file_path = os.path.join(network.log_dir.name, "server.log")
    util.setup_logging(log_file_path, 'werkzeug')
    generate_server_configs_local(network)
    run_flask_app(app, 5001)

