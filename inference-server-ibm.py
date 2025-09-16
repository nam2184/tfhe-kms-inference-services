import base64
import json
import os
import time
from multiprocessing import Process
from sqlalchemy.orm import context

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()  # TF 2.1.0 compatibility mode

from tensorflow.keras.models import model_from_json
from flask import Flask, request, jsonify
from flask_smorest import Api, Blueprint
from flask_smorest.blueprint import MethodView
from flask_cors import CORS, cross_origin

from sqlmodel import Session
from db import ContextModel, DBService, KeyModel
from models import EncryptedBodyMessageSchema, EncryptedMessageSchema, ErrorTypeSchema, PostContextBodySchema, PostContextSchema, PostKeySchema, PostKeyBodySchema
from network import Network
import util
from ibmhe import ibmhe, model, dataset
import pyhelayers

# IBM HELayers

network = Network()
app = Flask("ibmhe-server")
app.config["API_TITLE"] = "IBM HELayers Inference API"
app.config["API_VERSION"] = "1.0"
app.config["OPENAPI_VERSION"] = "3.0.0"

api = Api(app)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)
blp = Blueprint("model", "model", url_prefix="/", description="Model Inference Endpoints")

headers = {
    "Content-Type": "application/json; charset=utf-8"
}

cookies = {
    "session_id": "1"  # Replace with actual session cookie
}

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
    he_service.set_context_from_file("context.bin")
    print("now loading model") 
    model = he_service.load_encrypted_model_from_context_file(modelpath="model.bin")
    print("done loading model")
    return model

#model_8x8 = load_he_model("/models/model8x8_architecture.json", "/models/model8x8_weights.h5", config)
model_80x80 = load_he_model("/models/model80x80_architecture.json", "/models/model80x80_weights.h5")
#model_160x160 = load_he_model("models/model_160x160.json", "ibmhe/model_160x160.h5",config)


# ---------- Endpoints ----------
@cross_origin(supports_credentials=True)
@blp.route("/context")
class Key(MethodView):
    @blp.arguments(PostContextBodySchema)
    @blp.response(200, PostContextSchema)
    @blp.alt_response(status_code=400, schema=ErrorTypeSchema)
    def post(self, data):
        encoded_file = data["context"]
        encoded_model = data["model"]
        try:
            base64.b64decode(encoded_file)
        except Exception:
            return {"error": "Invalid base64 string"}, 400
        
        he_service = ibmhe.HEService()
        decoded_context = base64.b64decode(encoded_file)
        decoded_model = base64.b64decode(encoded_model)
        
        #he_service.set_context_from_buffer(decoded_context)
        #model = he_service.load_encrypted_model_from_context(decoded_model)
        
        context_record = ContextModel(
            id=int(data["id"]),
            context=encoded_file,
            model=encoded_model,
            chat_id=data["chat_id"],
        )
        context = DBService().insert_context(context_record)
        return context 

@cross_origin(supports_credentials=True)
@blp.route("/predict")
class Predict(MethodView):
    @blp.arguments(EncryptedBodyMessageSchema)
    @blp.response(200, EncryptedMessageSchema)
    @blp.alt_response(status_code=400, schema=ErrorTypeSchema)
    def post(self, data):
        try:
            context = DBService().get_context_by_chat_id(data["chat_id"])
            if context is None:
                raise ValueError("No context found from id")
            if context.model is None:
                raise ValueError("No model found in row")
            if context.context is None:
                raise ValueError("No context found in row")
                
            decoded_context = base64.b64decode(context.context)
            decoded_model = base64.b64decode(context.model)
            
            he_service = ibmhe.HEService()
            he_service.set_context_from_buffer(decoded_context)

            # load the encrypted model bound to this context
            model = he_service.load_encrypted_model_from_context(decoded_model)

            image_b64 = data["image"]
            image_buf = base64.b64decode(image_b64)

            encrypted_image = pyhelayers.load_encrypted_data(
                decoded_context,                  
                image_buf
            )

            time_begin = time.time()

            # Perform inference directly on encrypted data
            predictions = pyhelayers.EncryptedData(context)
            model.predict(predictions, encrypted_image)

            time_end = time.time()
            print(f"Inference time: {time_end - time_begin} sec")

            # prediction here will also be EncryptedData
            # save it to buffer then base64 encode for response
            pred_buf = predictions.save_to_buffer()
            encoded_str = base64.b64encode(pred_buf).decode("utf-8")

            data["classification_result"] = encoded_str
            return data

        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 400



# --- Register endpoints ---
api.register_blueprint(blp)

# --- Save OpenAPI spec ---
def save_openapi_spec(app, output_path="ibmhe-inference-api.json"):
    with app.app_context():
        spec_dict = api.spec.to_dict()
        with open(output_path, "w") as f:
            json.dump(spec_dict, f, indent=2)
        print(f"OpenAPI spec written to {output_path}")


def run_flask_app(app, port):
    app.run(host="127.0.0.1", port=port)


if __name__ == "__main__":
    save_openapi_spec(app)
    log_file_path = os.path.join(network.server_dir.name, "ibmhe-server.log")
    util.setup_logging(log_file_path, "werkzeug")
    p1 = Process(target=run_flask_app, args=(app, 5001))
    p1.start()
