import base64
from logging import debug
from multiprocessing import Process
import re
from concrete.fhe.compilation.artifacts import shutil
from flask import Flask, json, request, jsonify, send_file, Response
import os
from flask_wtf.csrf import logging
from werkzeug.wrappers import response
from db import DBService, HomomorphicKeyModel, KeyModel, ClientHEModel
import util
from network import Network
from concrete.ml.torch.compile import compile_brevitas_qat_model, tempfile, torch
from dotenv import load_dotenv
from flask_cors import CORS, cross_origin
from flask_smorest import Api, Blueprint
from flask_smorest.blueprint import MethodView
from models import  ErrorTypeSchema, GetClientSchema
from ml import ml, data
import numpy as np
from Crypto.Random import get_random_bytes
import zipfile

load_dotenv()
network = Network()

app = Flask("kms")
app.config["API_TITLE"] = "Key Manager API"
app.config["API_VERSION"] = "1.0"
app.config["OPENAPI_VERSION"] = "3.0.0"

api = Api(app)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

from concrete.ml.deployment import FHEModelDev, FHEModelClient 

headers = {
    'Content-Type': 'application/json; charset=utf-8'
}

cookies = {
    'session_id': '1'  # Replace with actual session cookie
}

blp = Blueprint("kms", "kms", url_prefix="/", description="KMS Endpoints")

def generate_dev_configs(network, model):
    base_dir = network.dev_dir.name
    print(f"[INFO] Cleaning dev_dir: {base_dir}")

    for filename in os.listdir(base_dir):
        file_path = os.path.join(base_dir, filename)

        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.remove(file_path)
            print(f"[DEBUG] Removed file: {file_path}")
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
            print(f"[DEBUG] Removed directory: {file_path}")

    fhe = FHEModelDev(base_dir, model=model)
    fhe.save()
    print(f"[INFO] Saved new FHEModelDev config for model={model}")
   
def generate_temp_keys(chat_id : int):
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    aes_key_path = os.path.join(temp_dir, f"aes_key{chat_id}.bin")

    # Generate AES key
    aes_key = get_random_bytes(32)
    with open(aes_key_path, "wb") as f:
        f.write(aes_key)

    # Zip the key
    zip_path = os.path.join(temp_dir, f"normal_keys{chat_id}.zip")
    with zipfile.ZipFile(zip_path, "w") as zipf:
        zipf.write(aes_key_path, arcname=f"aes_key{chat_id}.bin")
    return zip_path, temp_dir

@cross_origin(supports_credentials=True)
@blp.route('/client/<int:chat_id>')
class Client(MethodView):
    @blp.doc(description="Get FHE client.")
    @blp.response(status_code=200, schema=GetClientSchema)
    @blp.alt_response(status_code=400, schema=ErrorTypeSchema)
    def get(self, chat_id):
        print(f"[INFO] Handling GET /client/{chat_id}")

        db = DBService()
        record = db.get_heclient_key_by_chat_id(chat_id)
        print(f"[DEBUG] DB lookup for chat_id={chat_id}: {'found record' if record else 'no record'}")

        base_dir = network.dev_dir.name
        print(f"[DEBUG] base_dir={base_dir}")

        client = FHEModelClient(base_dir, key_dir=base_dir)
        print("[INFO] Created FHEModelClient")

        zip_path = os.path.join(base_dir, "client.zip")
        print(f"[DEBUG] Expecting client.zip at: {zip_path}")

        with open(zip_path, "rb") as f:
            zip_bytes = f.read()
            encoded_client_zip = base64.b64encode(zip_bytes).decode("utf-8")
        print("[INFO] Encoded base client.zip")

        resp = {"client_specs": encoded_client_zip}
        if record is None:
            print("[INFO] No record in DB -> generating fresh keys/zip")

            client = FHEModelClient(base_dir, key_dir=base_dir)
            client.generate_private_and_evaluation_keys()
            print("[INFO] Re-created FHEModelClient for fresh keys")

            # 1. Find numeric directory
            numeric_dirs = [
                d for d in os.listdir(base_dir)
                if os.path.isdir(os.path.join(base_dir, d)) and re.fullmatch(r"\d+", d)
            ]
            print(f"[DEBUG] numeric_dirs={numeric_dirs}")
            if not numeric_dirs:
                print("[ERROR] No numeric client folder found")
                return {"error": "No numeric client folder found"}, 400

            client_dir = os.path.join(base_dir, numeric_dirs[0])
            print(f"[INFO] Using client_dir={client_dir}")

            # 2. Create zip file
            zip_filename = f"client{chat_id}.zip"
            zip_keys_path = os.path.join(base_dir, zip_filename)

            # numeric_dirs[0] is the folder you want preserved inside the zip
            numeric_dir_path = os.path.join(base_dir, numeric_dirs[0])

            # Make the archive
            shutil.make_archive(
                base_name=os.path.join(base_dir, f"client{chat_id}"),  # name of zip without .zip
                format="zip",
                root_dir=base_dir,          # parent folder
                base_dir=numeric_dirs[0]    # folder inside base_dir to include
            )

            print(f"[INFO] Created zip archive at {zip_keys_path}")

            # 3. Read + encode before storing in DB
            with open(zip_keys_path, "rb") as f:
                zip_bytes = f.read()
                encoded_zip = base64.b64encode(zip_bytes).decode("utf-8")
                resp["keys"] = encoded_zip
            print("[INFO] Encoded generated client keys")

            client_specs = ClientHEModel(file=encoded_zip, chat_id=chat_id)
            db.insert_heclient_key(client_specs)
            print(f"[INFO] Inserted new client keys into DB for chat_id={chat_id}")

        else:
            encoded_key_zip = record.file
            resp["keys"] = encoded_key_zip
            print(f"[INFO] Using existing DB record for chat_id={chat_id}")
        
        # 5. Cleanup after response is sent
        def cleanup():
            print("[INFO] Running cleanup")
            if "zip_keys_path" in locals() and zip_keys_path is not None:
                if os.path.exists(zip_keys_path):
                    os.remove(zip_keys_path)
                    print(f"[DEBUG] Removed zip file: {zip_keys_path}")
            if "client_dir" in locals() and client_dir is not None:
                if os.path.exists(client_dir):
                    os.remove(client_dir)
                    print(f"[DEBUG] Removed client dir: {client_dir}")

        response = Response()
        response.call_on_close(cleanup)

        print(f"[INFO] Returning response for chat_id={chat_id}")
        return resp

@cross_origin(supports_credentials=True)
@blp.route('/keys/<int:chat_id>')
class Keys(MethodView):
    @blp.doc(
        description="Get Normal keys.",
        responses={
            200: {
                "description": "ZIP file containing the keys.",
                "content": {
                    "application/zip": {
                        "schema": {
                            "type": "string",
                            "format": "binary"
                        }
                    }
                }
            }
        }
    )
    @blp.alt_response(status_code=400, schema=ErrorTypeSchema)
    def get(self, chat_id):
        try:
            db = DBService()
            record = db.get_key_by_chat_id(chat_id)

            if record is None:
                # Generate fresh keys + zip
                zip_path, temp_dir = generate_temp_keys(chat_id)

                # Read zip as base64 string
                with open(zip_path, "rb") as f:
                    encoded_zip = base64.b64encode(f.read()).decode("utf-8")

                # Save to DB
                key = KeyModel(file=encoded_zip, chat_id=chat_id)
                db.insert_key(key)

            else:
                # Row exists → Reconstruct zip file from stored base64
                encoded_zip = record.file  # or aes_file if needed
                decoded_bytes = base64.b64decode(encoded_zip)

                temp_dir = tempfile.mkdtemp()
                zip_path = os.path.join(temp_dir, "keys.zip")

                with open(zip_path, "wb") as f:
                    f.write(decoded_bytes)

            # Send the ZIP file
            response = send_file(
                zip_path,
                mimetype="application/zip",
                as_attachment=True,
                download_name=f"normal_keys{chat_id}.zip"
            )

            # Clean up temp files after response
            @response.call_on_close
            def cleanup():
                shutil.rmtree(temp_dir, ignore_errors=True)
                print("Temporary keys deleted.")
            return response
        
        except Exception as e:
            error = {"section" : "keys", "message" : str(e)}
            return error, 400

@cross_origin(supports_credentials=True) 
@blp.route('/model')
class Model(MethodView):
    @blp.doc(description="Get FHE model.")
    @blp.doc(
        description="Get FHE model.",
        responses={
            200: {
                "description": "ZIP file containing the server.",
                "content": {
                    "application/zip": {
                        "schema": {
                            "type": "string",
                            "format": "binary"
                        }
                    }
                }
            }
        }
    )
    @blp.alt_response(status_code=400, schema=ErrorTypeSchema)
    def get(self):
        file_path = os.path.join(network.dev_dir.name, 'server.zip')
        if os.path.exists(file_path):
            return send_file(
                    file_path,
                    mimetype='application/zip',
                    as_attachment=True,
                    download_name='server.zip'
                    )
        else:
            print(f"File not found at path: {file_path}")
            return

# --- Register endpoints ---
api.register_blueprint(blp)

# --- Save OpenAPI spec ---
def save_openapi_spec(app, output_path="kms-api.json"):
    with app.app_context():
        spec_dict = api.spec.to_dict()
        with open(output_path, "w") as f:
            json.dump(spec_dict, f, indent=2)
        print(f"OpenAPI spec written to {output_path}")

def run_flask_app(app, port):
    app.run(host='127.0.0.1', port=port, debug=True)

if __name__ == '__main__':
    save_openapi_spec(app)
    image_size = 15
    net = ml.CNN(2, 1, image_size)
    dummy_input = torch.randn(1, 1, image_size, image_size)  # (N, C, H, W) 
    net(dummy_input)
    db = DBService()
    cwd = os.getcwd()
    x_train, x_test, y_train, y_test = data.split_and_preprocess_placeholder(cwd + "/dataset", size = (image_size, image_size)) 
    x_train = np.transpose(x_train, (0, 3, 1, 2))
    checkpoint = torch.load(cwd + f'/ml/models/cnn{image_size}_best.pth')
    #print(checkpoint)
    #for key in list(checkpoint.keys()):
     #   if 'model.' in key:
      #      checkpoint[key.replace('model.', '')] = checkpoint[key]
       #     del checkpoint[key]
    net.load_state_dict(checkpoint)
    print("Compiling model for deployment")
    q_module = compile_brevitas_qat_model(net, x_train, rounding_threshold_bits=4, p_error=0.01)        
    log_file_path = os.path.join(network.dev_dir.name, "dev.log")
    util.setup_logging(log_file_path, 'werkzeug')
                                                                                                                                                                
    generate_dev_configs(network, q_module)
    p1 = Process(target=run_flask_app, args=(app, 5000))
    p1.start()  
