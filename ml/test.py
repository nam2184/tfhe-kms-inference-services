import os
from PIL import Image
import json
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import asyncio
import datetime
import base64
import random
import time
import logging

import ml
import data
from resnet import ResNet18

def preprocess_images_color(image_paths, size=(64, 64)):
    processed = []
    for path in image_paths:
        image = Image.open(path)                           # keep original color
        image = image.convert("RGB")  # ensure 3 channels
        image = image.resize(size, Image.LANCZOS)
        arr = np.array(image, dtype=np.float32) / 255.0
        # Convert to channel-first: (H, W, C) → (C, H, W)
        arr = np.transpose(arr, (2, 0, 1))
        processed.append(arr)
    return np.array(processed) 


# === Configure Logging ===
LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, f"evaluation_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def make_small_dataset(base_path, n_samples=100, seed=42):
    random.seed(seed)
    x_train, y_train = [], []
    
    classes = [c for c in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, c))]
    num_classes = len(classes)
    samples_per_class = n_samples // num_classes

    for class_name in classes:
        class_path = os.path.join(base_path, class_name)
        label = 0 if class_name.lower() in ["neutral", "drawings"] else 1
        images = [
            os.path.join(class_path, f)
            for f in os.listdir(class_path)
            if f.lower().endswith((".jpg", ".png"))
        ]
        for img in random.sample(images, min(len(images), samples_per_class)):
            x_train.append(img)
            y_train.append(label)

    logger.info(f"Created small dataset with {len(x_train)} samples across {num_classes} classes.")
    return x_train, y_train


def split_and_preprocess_calibration(base_path, n_samples=100, seed=42):
    calibration_data, classification = make_small_dataset(base_path, n_samples=n_samples, seed=seed)
    calibration_data_preprocessed = preprocess_images_color(calibration_data)
    logger.info("Preprocessed calibration dataset.")
    return calibration_data, calibration_data_preprocessed, classification


def test_resnet_cnn_accuracy(model_dir, base_path, n_samples=100, device=None, n_bits=4):
    """
    Evaluate a saved ResNet + CNN model on a random sample of images.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model_path = os.path.join(model_dir, f"resnet_cnn_best{n_bits}bits.pth")
    spec_path = os.path.join(model_dir, f"resnet_cnn_best_specs{n_bits}bits.json")

    if not os.path.exists(model_path) or not os.path.exists(spec_path):
        logger.error("Model or specs file not found in the specified directory.")
        raise FileNotFoundError("Model or specs file not found in the specified directory.")

    logger.info("Loading model specifications...")
    with open(spec_path, "r") as f:
        specs = json.load(f)

    n_classes = specs["n_classes"]
    backbone_target_channels = specs["backbone_target_channels"]
    backbone_target_size = specs["backbone_target_size"]

    # === Load random sample ===
    logger.info(f"Sampling {n_samples} images from dataset...")
    calibration_data, calibration_data_preprocessed, classification = split_and_preprocess_calibration(
        base_path, n_samples=n_samples
    )
    logger.info(calibration_data_preprocessed.shape)
    # === Prepare tensors ===
    x = torch.tensor(calibration_data_preprocessed, dtype=torch.float32)
    y = torch.tensor(classification, dtype=torch.long)
    loader = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=False)
    logger.info(f"Prepared DataLoader with {len(loader)} batches.")

    # === Load models ===
    logger.info("Loading models...")
    backbone = ResNet18(target_channels=backbone_target_channels, target_size=backbone_target_size).to(device)
    classifier = ml.CNN(n_classes=n_classes, in_channels=3, image_size=backbone_target_size, n_bits=n_bits).to(device)

    # Dummy forward for init (for quantized models)
    dummy_data = data.split_and_preprocess_calibration(
        os.getcwd() + "/dataset", n_samples=10, size=(16, 16)
    )
    dummy_data = np.transpose(dummy_data, (0, 3, 1, 2))
    classifier.forward(torch.from_numpy(dummy_data))

    state_dicts = torch.load(model_path, map_location=device)
    backbone.load_state_dict(state_dicts["backbone_state_dict"])
    classifier.load_state_dict(state_dicts["classifier_state_dict"])

    backbone.eval()
    classifier.eval()
    logger.info("Models loaded and set to evaluation mode.")

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for xb, yb in tqdm(loader, desc="Evaluating samples"):
            xb, yb = xb.to(device), yb.to(device)
            features = backbone(xb)
            features_min = features.min()
            features_max = features.max()
            features = (features - features_min) / (features_max - features_min + 1e-8)
            outputs = classifier(features)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(yb.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # === Compute metrics ===
    accuracy = np.mean(all_preds == all_targets)
    precision = precision_score(all_targets, all_preds, average="binary" if n_classes == 2 else "macro", zero_division=0)
    recall = recall_score(all_targets, all_preds, average="binary" if n_classes == 2 else "macro", zero_division=0)
    f1 = f1_score(all_targets, all_preds, average="binary" if n_classes == 2 else "macro", zero_division=0)

    logger.info(f"--- Random {n_samples} Sample Evaluation ---")
    logger.info(f"Accuracy:  {accuracy*100:.2f}%")
    logger.info(f"Precision: {precision:.3f}")
    logger.info(f"Recall:    {recall:.3f}")
    logger.info(f"F1 Score:  {f1:.3f}")

    return {
        "accuracy_percent": accuracy * 100,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }


if __name__ == "__main__":
    dataset_root = os.path.dirname(os.getcwd())
    try:
        results_3bits = test_resnet_cnn_accuracy(
            model_dir=os.path.join(os.getcwd(), "ml", "models"),
            base_path=os.path.join(dataset_root, "dataset"),
            n_bits=3
        )
        logger.info("Evaluation complete.")
        logger.info(json.dumps(results_3bits, indent=2))
        results_4bits = test_resnet_cnn_accuracy(
            model_dir=os.path.join(os.getcwd(), "ml", "models"),
            base_path=os.path.join(dataset_root, "dataset"),
            n_bits=4
        )
        logger.info("Evaluation complete.")
        logger.info(json.dumps(results_4bits, indent=2))
        result = {
            "3bits" : results_3bits,
            "4bits" : results_4bits
        }
        with open("plaintext_stats.json", "w") as out:
            json.dump(result, out, indent=4)
    except Exception as e:
        logger.exception(f"Error during evaluation: {e}")

