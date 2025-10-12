import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json
import os
import ml
from data import split_dataset_xy, split_and_preprocess, preprocess_image
from resnet import ResNet18
from tqdm import tqdm

def train_and_save(
    base_path,
    save_dir="model_out",
    n_classes=2,
    epochs=10,
    size=8,
    batch_size=32,
    lr=1e-3,
    device=None,
):
    # Ensure device
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # === Load & preprocess dataset ===
    x_train, y_train, x_val, y_val = split_and_preprocess(base_path, size=(size, size))

    # PyTorch wants (N, C, H, W) not (N, H, W, 1)
    x_train = torch.tensor(x_train, dtype=torch.float32).permute(0, 3, 1, 2)
    x_val = torch.tensor(x_val, dtype=torch.float32).permute(0, 3, 1, 2)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)

    train_ds = TensorDataset(x_train, y_train)
    val_ds = TensorDataset(x_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # === Model, optimizer, loss ===
    model = ml.CNN(n_classes, 3, size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # === Training loop ===
    print("Starting training")
    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

        train_acc = correct / total
        train_loss = running_loss / total

        # === Validation ===
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                outputs = model(xb)
                loss = criterion(outputs, yb)
                val_loss += loss.item() * xb.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == yb).sum().item()
                val_total += yb.size(0)

        val_acc = val_correct / val_total
        val_loss /= val_total

        print(f"Epoch {epoch+1}/{epochs} "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # === Save best model ===
        if val_acc > best_val_acc:
            best_val_acc = val_acc

            os.makedirs(save_dir, exist_ok=True)
            model_path = os.path.join(save_dir, f"cnn{size}_best.pth")
            spec_path = os.path.join(save_dir, f"model{size}_best_specs.json")

            torch.save(model.state_dict(), model_path)

            specs = {
                "n_classes": n_classes,
                "input_shape": [1, size, size],   # C, H, W
                "epochs_trained": epoch + 1,
                "batch_size": batch_size,
                "learning_rate": lr,
                "best_val_acc": best_val_acc,
            }
            with open(spec_path, "w") as f:
                json.dump(specs, f, indent=2)

            print(f"New best model saved (Val Acc: {best_val_acc:.4f}) to {model_path}")

def train_and_save_resnet_cnn(
    base_path,
    save_dir="model_out",
    n_classes=2,
    epochs=10,
    input_size=None,       # optional, can be None for adaptive ResNet
    batch_size=32,
    lr=1e-3,
    device=None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # === Load & preprocess dataset ===
    x_train, y_train, x_val, y_val = split_and_preprocess(
        base_path,
        size=(input_size, input_size) if input_size else None
    )

    # Convert to (N, C, H, W)
    x_train = torch.tensor(x_train, dtype=torch.float32).permute(0, 3, 1, 2)
    x_val = torch.tensor(x_val, dtype=torch.float32).permute(0, 3, 1, 2)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False)

    # === Model setup ===
    backbone = ResNet18(target_channels=3, target_size=15).to(device)  # outputs (B,3,15,15)
    classifier = ml.CNN(n_classes=n_classes, in_channels=3, image_size=15).to(device)  # consumes ResNet features

    # Combine parameters for joint optimization
    optimizer = optim.Adam(list(backbone.parameters()) + list(classifier.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    print("Starting training...")

    for epoch in range(epochs):
        backbone.train()
        classifier.train()
        running_loss, correct, total = 0.0, 0, 0

        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()

            features = backbone(xb)              # ResNet feature extractor
            features_min = features.min()
            features_max = features.max()
            features = (features - features_min) / (features_max - features_min + 1e-8)
            outputs = classifier(features)       # CNN classifier

            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # === Validation ===
        backbone.eval()
        classifier.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                features = backbone(xb)
                # normalize to 0–1 range
                features_min = features.min()
                features_max = features.max()
                features = (features - features_min) / (features_max - features_min + 1e-8)
                outputs = classifier(features)
                loss = criterion(outputs, yb)
                val_loss += loss.item() * xb.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == yb).sum().item()
                val_total += yb.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        print(f"Epoch {epoch+1}/{epochs} "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # === Save best model ===
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(save_dir, exist_ok=True)

            model_path = os.path.join(save_dir, f"resnet_cnn_best.pth")
            spec_path = os.path.join(save_dir, f"resnet_cnn_best_specs.json")

            torch.save({
                "backbone_state_dict": backbone.state_dict(),
                "classifier_state_dict": classifier.state_dict(),
            }, model_path)

            specs = {
                "n_classes": n_classes,
                "backbone_target_channels": 3,
                "backbone_target_size": 15,
                "epochs_trained": epoch + 1,
                "batch_size": batch_size,
                "learning_rate": lr,
                "best_val_acc": best_val_acc,
            }
            with open(spec_path, "w") as f:
                json.dump(specs, f, indent=2)

            print(f"New best model saved (Val Acc: {best_val_acc:.4f}) to {model_path}")


if __name__ == "__main__":
    base_path = os.path.dirname(os.path.dirname(os.getcwd()))       # path to your dataset
    save_dir = "models"
    n_classes = 2
    epochs = 20
    input_size = 64               # or None for adaptive input
    batch_size = 32
    lr = 1e-3
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Training on device: {device}")

    train_and_save_resnet_cnn(
        base_path=base_path + "/dataset",
        save_dir=save_dir,
        n_classes=n_classes,
        epochs=epochs,
        input_size=input_size,
        batch_size=batch_size,
        lr=lr,
        device=device
    )
