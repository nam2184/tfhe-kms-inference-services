import os
import random
from PIL import Image
import numpy as np

def split_dataset_xy(base_path, train_ratio=0.8, seed=42, sample=False):
    random.seed(seed)
    x_train, y_train, x_val, y_val = [], [], [], []

    for class_name in os.listdir(base_path):
        class_path = os.path.join(base_path, class_name)
        if not os.path.isdir(class_path):
            continue

        label = 0 if class_name.lower() == "neutral" else 1
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
    x_train = preprocess_images(x_train_paths, size)
    x_val = preprocess_images(x_val_paths, size)
    print("x_train shape:", x_train.shape)
    print("x_val shape:", x_val.shape)
    return x_train, np.array(y_train), x_val, np.array(y_val)


def split_and_preprocess_placeholder(base_path, train_ratio=0.8, seed=42, size=(8, 8)):
    """
    Splits the dataset and preprocesses both x_train and x_val automatically.

    Returns:
        x_train (np.ndarray): shape (N, H, W, 1)
        y_train (np.ndarray): shape (N,)
        x_val (np.ndarray): shape (M, H, W, 1)
        y_val (np.ndarray): shape (M,)
    """
    x_train_paths, y_train, x_val_paths, y_val = split_dataset_xy(base_path, train_ratio, seed, True)
    x_train = preprocess_images(x_train_paths, size)
    x_val = preprocess_images(x_val_paths, size)
    print("x_train shape:", x_train.shape)
    print("x_val shape:", x_val.shape)
    return x_train, np.array(y_train), x_val, np.array(y_val)

