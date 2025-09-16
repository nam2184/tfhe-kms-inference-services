import base64
import time
from multiprocessing import Process
from flask import Flask, json, jsonify, request
import os
import logging

from db import DBService
import util
from network import Network
from dotenv import load_dotenv
from flask_cors import CORS, cross_origin
from flask_smorest import Api, Blueprint
from flask_smorest.blueprint import MethodView
from models import ErrorTypeSchema, GetContextSchema, PostBodyModelSchema, PostModelSchema, PostSecretSchema, PostBodySecretSchema
from ibmhe import ibmhe, model
import pyhelayers
from tensorflow.keras.models import model_from_json

load_dotenv()
network = Network()

app = Flask("kms")
app.config["API_TITLE"] = "Key Manager API"
app.config["API_VERSION"] = "1.0"
app.config["OPENAPI_VERSION"] = "3.0.0"

api = Api(app)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

blp = Blueprint("kms", "kms", url_prefix="/", description="KMS Endpoints")

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
)

# ---------- Model Loading ----------
def load_tf_model(model_json_path, model_h5_path):
    with open(model_json_path, "r") as json_file:
        loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    model.load_weights(model_h5_path)
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

def load_he_model(model_json_path, model_h5_path):
    he_service = ibmhe.HEService()
    he_service.set_context_inner_no_config()
    logging.info("Loading encrypted model...")
    model = he_service.load_encrypted_model(
        os.getcwd() + model_json_path,
        os.getcwd() + model_h5_path
    )
    logging.info("Encrypted model loaded.")
    return model

@cross_origin(supports_credentials=True)
@blp.route("/context")
class Context(MethodView):
    @blp.doc(description="Get FHE client.")
    @blp.response(200, GetContextSchema)
    @blp.alt_response(status_code=400, schema=ErrorTypeSchema)
    def get(self):
        start_time = time.time()
        logging.info("Starting client generation...")

        # Load encrypted model
        model_80x80 = load_he_model(
            "/models/model80x80_architecture.json",
            "/models/model80x80_weights.h5"
        )
        logging.info("Model loaded successfully.")

        # Extract buffers
        ctx = model_80x80.get_created_he_context()

        logging.info("Got Context")        
        context_buffr = ctx.save_to_buffer()
        logging.info("Saved Context to buffer")        

        # Encode as base64
        context_b64 = base64.b64encode(context_buffr).decode("utf-8")

        elapsed_time = time.time() - start_time
        logging.info(f"Client generated in {elapsed_time:.2f} seconds.")
       
        del model_80x80
        del ctx
        print(context_b64) 
        return {
            "context": context_b64,
        }

@cross_origin(supports_credentials=True)
@blp.route("/model")
class Model(MethodView):
    @blp.arguments(PostBodyModelSchema)
    @blp.response(200, PostModelSchema)
    @blp.alt_response(status_code=400, schema=ErrorTypeSchema)
    def post(self):
        """Given context, return model buffer"""
        try:
            body = request.json
            context_b64 = body.get("context")
            if not context_b64:
                raise ValueError("Missing context")
            decoded_ctx = base64.b64decode(context_b64)
            ctx = pyhelayers.HeContext()
            ctx.load_from_buffer(decoded_ctx)

            model_80x80 = load_he_model(
                "/models/model80x80_architecture.json",
                "/models/model80x80_weights.h5"
            )
            model_bffr = model_80x80.save_to_buffer()
            model_b64 = base64.b64encode(model_bffr).decode("utf-8")

            return {"model": model_b64}
        except Exception as e:
            logging.error(f"/model error: {e}")
            return jsonify({"status": "error", "message": str(e)}), 400


@cross_origin(supports_credentials=True)
@blp.route("/secret")
class Secret(MethodView):
    @blp.arguments(PostBodySecretSchema)
    @blp.response(200, PostSecretSchema)
    @blp.alt_response(status_code=400, schema=ErrorTypeSchema)
    def post(self):
        """Given context, return secret key buffer"""
        try:
            body = request.json
            context_b64 = body.get("context")
            if not context_b64:
                raise ValueError("Missing context")

            decoded_ctx = base64.b64decode(context_b64)
            ctx = pyhelayers.DefaultContext()
            ctx.load_from_buffer(decoded_ctx)

            secret_bffr = ctx.save_secret_key()
            secret_b64 = base64.b64encode(secret_bffr).decode("utf-8")

            return {"secret_key": secret_b64}
        except Exception as e:
            logging.error(f"/secret error: {e}")
            return jsonify({"status": "error", "message": str(e)}), 400

# --- Register endpoints ---
api.register_blueprint(blp)

# --- Save OpenAPI spec ---
def save_openapi_spec(app, output_path="kms-ibm-api.json"):
    with app.app_context():
        spec_dict = api.spec.to_dict()
        with open(output_path, "w") as f:
            json.dump(spec_dict, f, indent=2)
        logging.info(f"OpenAPI spec written to {output_path}")

def run_flask_app(app, port):
    app.run(host='0.0.0.0', port=port)


if __name__ == '__main__':
    # Save OpenAPI spec
    save_openapi_spec(app)

    p1 = Process(target=run_flask_app, args=(app, 5000))
    p1.start()
