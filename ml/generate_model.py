from numpy.lib.function_base import copy
import data
from ml import CNN28, CNN8
import torch
from tqdm import tqdm
import os
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from he import FHEBase
from concrete.ml.torch.compile import compile_brevitas_qat_model
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import time
import json
from sklearn.metrics import classification_report
from sklearn.preprocessing import label_binarize
import copy


def test(net, test_loader):
    # Cross Entropy loss for classification when not using a softmax layer in the network
    loss_fn = nn.CrossEntropyLoss()
    avg_loss = 0
    correct_predictions = 0
    total_samples = 0

    for data, target in test_loader:
        output = net(data)
        loss_net = loss_fn(output, target.long())
        predicted_classes = torch.argmax(output, dim=1)
        correct_predictions += (predicted_classes == target).sum().item()
        total_samples += target.size(0)
        avg_loss += loss_net.item()

    accuracy = correct_predictions / total_samples
    return accuracy, avg_loss / len(test_loader)


def train_one_epoch(net, optimizer, train_loader):
    loss_fn = nn.CrossEntropyLoss()
    avg_loss = 0
    correct_predictions = 0
    total_samples = 0
    net.train()

    for data, target in train_loader:
        optimizer.zero_grad()
        output = net(data)
        loss_net = loss_fn(output, target.long())
        predicted_classes = torch.argmax(output, dim=1)
        correct_predictions += (predicted_classes == target).sum().item()
        total_samples += target.size(0)
        loss_net.backward()
        optimizer.step()
        avg_loss += loss_net.item()

    accuracy = correct_predictions / total_samples
    return accuracy, avg_loss / len(train_loader)


# Training function without FHE
def test_training_no_fhe(batch_size=32, n_epochs=60, size=8):
    # Load dataset
    x_train, x_test, y_train, y_test = data.test_nsfw_dataset(os.getcwd() + "/nsfw_dataset_v1") 

    # Prepare training dataset and dataloader
    train_dataset = data.Dataset(x_train, y_train, quantize=False)
    train_dataloader = train_dataset.load_data(batch_size=batch_size)

    # Initialize the model
    net = CNN8(n_classes=2)

    # Use Adam optimizer and LR scheduler
    optimizer = Adam(net.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.1)

    losses = []
    best_accuracy = 0.0
    best_loss = float('inf')
    best_model_state = None

    # Training loop
    with tqdm(total=n_epochs, unit=" epochs") as pbar:
        for epoch in range(n_epochs):
            accuracy, loss = train_one_epoch(net, optimizer, train_dataloader)
            losses.append(loss)
            scheduler.step(loss)

            pbar.set_description(f"Epoch {epoch} - Accuracy: {accuracy:.4f} - Loss: {loss:.4f}")
            pbar.update(1)

            # Track best model based on accuracy
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_state = net.state_dict()

    # Restore best model and save
    if best_model_state is not None:
        torch.save(best_model_state, 'model_complete.pth')
    else :    
        torch.save(net.state_dict(), 'model_complete.pth')
    print(f"Best training accuracy: {best_accuracy:.4f}")    

    return net, losses

# Concrete FHE testing function with parallelism
def test_concrete(net, sample_size=None, batch_size=32, num_threads=16, size = 8):
    x_train, x_test, y_train, y_test = data.test_nsfw_dataset(os.getcwd() + "/nsfw_dataset_v1") 
    if sample_size is not None:
        indices = np.random.choice(len(x_test), size=sample_size, replace=False)
        x_test = x_test[indices]
        y_test = y_test[indices]

    test_dataset = data.Dataset(x_test, y_test, quantize=False)
    test_dataloader = test_dataset.load_data(batch_size=batch_size)
    # Compile Brevitas QAT model                                                                                                                                                                    
    q_module = compile_brevitas_qat_model(net, x_train, rounding_threshold_bits=4, p_error=0.01, )        
    t_start = time.time()

    q_module.fhe_circuit.keygen()
    keygen_time = time.time() - t_start
    print(f"Keygen time: {keygen_time:.2f}s")

    fhe_model = FHEBase(net, q_module, test_dataloader, num_threads=num_threads)

    print("Testing in progress...")
    t_start_inference = time.time()

    # Perform inference
    accuracy_test, all_y_pred, all_targets, all_logits = fhe_model.test(return_logits=True)

    elapsed_time = time.time() - t_start_inference
    num_inferences = sample_size if sample_size is not None else len(y_test)
    time_per_inference = elapsed_time / num_inferences
    accuracy_percentage = 100 * accuracy_test

    # Convert logits to probabilities using softmax
    #Cant use loss entropy for AUC
    all_probs = torch.softmax(torch.tensor(all_logits), dim=1).numpy()

    # Binarize the target labels for AUC calculation
    y_one_hot = label_binarize(all_targets, classes=np.arange(all_probs.shape[1]))

    # Compute AUC for multi-class classification (One-vs-Rest approach)
    auc_score = roc_auc_score(y_one_hot, all_probs, multi_class='ovr')

    # Generate the classification report
    all_y_pred = np.argmax(all_probs, axis=1)
    report = classification_report(all_targets, all_y_pred, zero_division=1, output_dict=True)

    # Compute confusion matrix
    cm = confusion_matrix(all_targets, all_y_pred)

    # Collect results in a dictionary
    results = {
        "accuracy": accuracy_percentage,
        "time_per_inference": time_per_inference,
        "auc_score": auc_score,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "true_labels": all_targets.tolist(),
        "predicted_probabilities": all_probs.tolist(),
    }

    # Save the results to a JSON file
    #Change directory as needed
    results_dir = os.path.join(os.path.expanduser("~"), "homomorphic")
    os.makedirs(results_dir, exist_ok=True)


    results_path = os.path.join(results_dir, "results.json")
    with open(results_path, "w") as json_file:
        json.dump(results, json_file, indent=4)

    print(f"Results saved to {results_path}")

def print_classification_report(report):
    print("\nClassification Report:")
    for class_name, metrics in report.items():
        if isinstance(metrics, dict):
            if class_name.isdigit():
                print(f"\nClass {class_name}:")
            else:
                print(f"\n{class_name.capitalize()}:")
            for metric_name, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {metric_name}: {value:.2f}")
                else:
                    print(f"  {metric_name}: {value}")
        else:
            print(f"\n{class_name}: {metrics:.2f}")





if __name__ == "__main__":
    torch.manual_seed(42)

    # Train the model without FHE, using increased epochs and adjusted batch size
    net, losses_bits = test_training_no_fhe(batch_size=64, n_epochs=100, size = 8)

    # Test the FHE model with adjusted batch size and num_threads
    test_concrete(net, batch_size=64, num_threads=32, size = 8)

    # To test a certain number of samples, uncomment and adjust the following line:
    # test_concrete(net, sample_size=10, batch_size=32, num_threads=16)
