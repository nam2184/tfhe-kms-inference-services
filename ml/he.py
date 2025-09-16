import numpy as np
from tqdm import tqdm

class FHEBase:
    def __init__(self, nn, keycompiler, data, num_threads=10) -> None:
        self.nn = nn
        self.keycompiler = keycompiler
        self.data = data
        self.use_sim = True
        self.num_threads = num_threads

    def preprocessing(self):
        pass

    def compile_model(self):
        pass

    def test(self, return_logits=False):
        # Casting the inputs into int64 is recommended
        all_y_pred = np.zeros((len(self.data.dataset)), dtype=np.int64)
        all_targets = np.zeros((len(self.data.dataset)), dtype=np.int64)

        # Initialize list to hold logits if requested
        all_logits = []

        # Iterate over the test batches and accumulate predictions and ground truth labels in a vector
        idx = 0
        for data, target in tqdm(self.data, desc="Progress bar"):
            data = data.numpy()
            target = target.numpy()

            fhe_mode = "simulate" if self.use_sim else "execute"

            # Quantize the inputs and cast to appropriate self.data type
            y_pred = self.keycompiler.forward(data, fhe=fhe_mode)

            endidx = idx + target.shape[0]

            # Accumulate the ground truth labels
            all_targets[idx:endidx] = target

            # Store logits if needed
            if return_logits:
                all_logits.append(y_pred)  # Assuming y_pred contains the raw output logits

            # Get the predicted class id and accumulate the predictions
            y_pred = np.argmax(y_pred, axis=1)
            all_y_pred[idx:endidx] = y_pred

            # Update the index
            idx += target.shape[0]

        # Compute and report accuracy
        n_correct = np.sum(all_targets == all_y_pred)
        accuracy = n_correct / len(self.data.dataset)

        if return_logits:
            # Convert logits list to a NumPy array
            all_logits = np.vstack(all_logits)  # Stack the logits if they are in list form
            return accuracy, all_y_pred, all_targets, all_logits
        
        return accuracy, all_y_pred, all_targets
