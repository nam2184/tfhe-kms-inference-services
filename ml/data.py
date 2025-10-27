import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2

def split_dataset(base_path, train_ratio=0.8, seed=42):
    """
    Split dataset into train/test with binary labels and preprocess images.
    Neutral = 0, everything else = 1.

    Args:
        base_path (str): Path to dataset containing class folders.
        train_ratio (float): Fraction of data for training.
        seed (int): Random seed for reproducibility.

    Returns:
        list: Train set as (image_array, label)
        list: Test set as (image_array, label)
    """
    random.seed(seed)
    train_set = []
    test_set = []
    for class_name in os.listdir(base_path):
        class_path = os.path.join(base_path, class_name)
        if not os.path.isdir(class_path):
            continue
        label = 0 if class_name.lower() == "neutral" else 1
        images = [os.path.join(class_path, f) for f in os.listdir(class_path) if f.lower().endswith((".jpg", ".png"))]
        random.shuffle(images)

        split_idx = int(len(images) * train_ratio)
        for img_path in images[:split_idx]:
            img = Image.open(img_path)
            img_array = preprocess_image(img)
            train_set.append((img_array, label))
        for img_path in images[split_idx:]:
            img = Image.open(img_path)
            img_array = preprocess_image(img)
            test_set.append((img_array, label))

    return train_set, test_set

def test_nsfw_dataset(data_dir):
    # Get train/test sets as tuples
    train_set, test_set = split_dataset(data_dir)

    # Split tuples into separate lists of paths and labels
    X_train, y_train = zip(*train_set) if train_set else ([], [])
    X_test, y_test = zip(*test_set) if test_set else ([], [])

    # Optional: further split training into train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, shuffle=True, random_state=42
    )

    return X_train, X_val, y_train, y_val

def split_dataset_xy(base_path, train_ratio=0.8, seed=42, sample=False):
    random.seed(seed)
    x_train, y_train, x_val, y_val = [], [], [], []

    for class_name in os.listdir(base_path):
        class_path = os.path.join(base_path, class_name)
        if not os.path.isdir(class_path):
            continue

        label = 0 if class_name.lower() == "neutral" or class_name.lower() == "drawings" else 1
        images = [os.path.join(class_path, f) 
                  for f in os.listdir(class_path) 
                  if f.lower().endswith((".jpg", ".png"))]
        random.shuffle(images)
        split_idx = int(len(images) * train_ratio)
        x_train.extend(images[:split_idx])
        y_train.extend([label] * split_idx)
        x_val.extend(images[split_idx:])
        y_val.extend([label] * (len(images) - split_idx))
        if sample :
            break

    return x_train, y_train, x_val, y_val

def make_small_dataset(base_path, n_samples=1000, seed=42):
    random.seed(seed)
    x_train, y_train = [], []

    all_images = []
    all_labels = []

    # Collect all images with their labels
    for class_name in os.listdir(base_path):
        class_path = os.path.join(base_path, class_name)
        if not os.path.isdir(class_path):
            continue

        # Label rule: 0 for "neutral"/"drawing", 1 for everything else
        label = 0 if class_name.lower() in ["neutral", "drawings"] else 1
        images = [
            os.path.join(class_path, f)
            for f in os.listdir(class_path)
            if f.lower().endswith((".jpg", ".png"))
        ]

        all_images.extend(images)
        all_labels.extend([label] * len(images))

    # Shuffle and sample
    combined = list(zip(all_images, all_labels))
    random.shuffle(combined)
    sampled = combined[:n_samples]

    # Unzip back into x_train and y_train
    x_train, y_train = zip(*sampled)

    return list(x_train), list(y_train)

def preprocess_images(image_paths, size=(8, 8)):
    processed = []
    for path in image_paths:
        img = Image.open(path).convert("L")              # force grayscale
        img_resized = img.resize(size, Image.LANCZOS)
        arr = np.array(img_resized, dtype=np.float32)    # shape (H, W)
        arr = arr / 255.0                                # normalize to [0,1]
        arr = np.expand_dims(arr, axis=-1)               # add channel -> (H, W, 1)
        processed.append(arr)
    return np.array(processed)                           # shape (N, H, W, 1)


def preprocess_images_color(image_paths, size=(8, 8)):
    processed = []
    for path in image_paths:
        img = Image.open(path)                           # keep original color
        img_resized = img.resize(size, Image.LANCZOS)
        arr = np.array(img_resized, dtype=np.float32)    # shape (H, W, C)
        arr = arr / 255.0                                # normalize to [0,1]
        processed.append(arr)
    return np.array(processed) 

def split_and_preprocess(base_path, train_ratio=0.8, seed=42, size=(8, 8)):
    """
    Splits the dataset and preprocesses both x_train and x_val automatically.

    Returns:
        x_train (np.ndarray): shape (N, H, W, 1)
        y_train (np.ndarray): shape (N,)
        x_val (np.ndarray): shape (M, H, W, 1)
        y_val (np.ndarray): shape (M,)
    """
    x_train_paths, y_train, x_val_paths, y_val = split_dataset_xy(base_path, train_ratio, seed)
    x_train = preprocess_images_color(x_train_paths, size)
    x_val = preprocess_images_color(x_val_paths, size)
    print("x_train shape:", x_train.shape)
    print("x_val shape:", x_val.shape)
    return x_train, np.array(y_train), x_val, np.array(y_val)

def split_and_preprocess_calibration(base_path, n_samples=100, seed=42, size=(8, 8)):
    """
    Splits the dataset and preprocesses both x_train and x_val automatically.

    Returns:
        x_train (np.ndarray): shape (N, H, W, 1)
        y_train (np.ndarray): shape (N,)
        x_val (np.ndarray): shape (M, H, W, 1)
        y_val (np.ndarray): shape (M,)
    """
    calibration_data, calibration_y = make_small_dataset(base_path=base_path, n_samples=n_samples, seed=seed)
    calibration_data = preprocess_images_color(calibration_data, size)
    return calibration_data

#https://huggingface.co/datasets/deepghs/nsfw_detect
