import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json
import os
import ml
from data import split_dataset_xy, split_and_preprocess, preprocess_image

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
    model = ml.CNN(n_classes, 1, size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # === Training loop ===
    print("Starting training")
    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for xb, yb in train_loader:
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

            print(f"🔥 New best model saved (Val Acc: {best_val_acc:.4f}) to {model_path}")

project_root = os.path.dirname(os.getcwd())
train_and_save(
    base_path=project_root + "/dataset",
    save_dir="models",
    n_classes=2,   # neutral vs other
    size=15,
    epochs=50,
    batch_size=32,
    lr=1e-3
)


