import pyhelayers
import tensorflow.compat.v1 as tf
import time

tf.disable_eager_execution()


class HEService:
    def __init__(self):
        self.he_run_req = pyhelayers.HeRunRequirements()
        self.context = pyhelayers.DefaultContext()
        self.he_run_req.set_model_encrypted(False)
        self.plain_model = None   # NeuralNetPlain
        self.he_model = None      # Encrypted model

    def get_context(self) -> pyhelayers.HeContext:
        return self.context

    def set_context_from_file(self, filepath : str):
        """Use an externally provided HeContext."""
        self.context.load_from_file(filepath)
        self.he_run_req.set_he_context_options([self.context])

    def set_context_from_buffer(self, buffer : bytes):
        """Use an externally provided HeContext."""
        self.context.load_from_buffer(buffer)
        self.he_run_req.set_he_context_options([self.context])

    def set_context_outer(self, context: pyhelayers.HeContext):
        """Use an externally provided HeContext."""
        self.context = context
        self.he_run_req.set_he_context_options([context])

    def set_context_inner(self, config: pyhelayers.HeConfigRequirement):
        """Initialize a new context from config."""
        self.context.init(config)
        self.he_run_req.set_he_context_options([self.context])

    def set_context_inner_no_config(self):
        """Use existing default context."""
        self.he_run_req.set_he_context_options([self.context])

    def save_to_file(self, filepath: str) -> None:
        self.context.save_to_file(filepath)

    def load_key(self, key_path: str):
        """Load secret key from file for decryption."""
        self.context.load_secret_key_from_file(key_path)

    # --- Load TF model into NeuralNetPlain ---
    def load_encrypted_model(self, json_path: str, h5_path: str):
        """Load a plain TF model (architecture + weights)."""
        nnp = pyhelayers.NeuralNet()
        print("now encoding model")
        nnp.encode([json_path, h5_path], self.he_run_req)
        print("✅ Loaded plain model")
        self.he_model = nnp
        return nnp

    def load_encrypted_model_from_plain(self, json_path: str, h5_path: str):
        """Load a plain TF model (architecture + weights)."""
        nnp = pyhelayers.NeuralNetPlain()
        nnp.init_from_files(pyhelayers.PlainModelHyperParams(), [json_path, h5_path])
        print("✅ Loaded plain model")
        self.plain_model = nnp
        he_model = pyhelayers.NeuralNet()
        he_model.compile(nnp, self.he_run_req)
        print("✅ Loadedencrypted model")
        return nnp

    def load_encrypted_model_from_context_file(self, modelpath):
        """Load a plain TF model (architecture + weights)."""
        with open(modelpath, "rb") as f:
            buf = f.read()
        nnp = pyhelayers.load_he_model(self.context, buf)
        print("✅ Loade dencrypted model")
        return nnp


    def load_encrypted_model_from_context(self, model_buffer):
        nnp = pyhelayers.load_he_model(self.context, model_buffer)
        print("✅ Loaded he model")
        return nnp


        # --- Run inference ---
    def run_inference(self, input_data):
        if self.he_model is None:
            raise RuntimeError("HE model not compiled. Call compile_he_model first.")

        time_begin = time.time()
        predictions=self.he_model.predict(input_data)
        time_end = time.time()
        print(f"Time take : {time_end - time_begin} seconds")

        return predictions 

