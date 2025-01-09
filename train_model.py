import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import ExerciseDataset, ExerciseClassifier, ANGLE_NAMES
import json
import numpy as np

TRAIN_PATH = "data/processed/train.json"
VAL_PATH = "data/processed/val.json"
MODEL_PATH = "models/exercise_classifier.pth"
LABEL_MAPPING_PATH = "data/processed/label_mapping.json"
REFERENCE_STATS_PATH = "data/processed/reference_stats.json"

def compute_reference_stats(train_path):
    with open(train_path, "r") as f:
        data = json.load(f)

    angle_data_by_label = {}
    for entry in data:
        label = entry["label"]
        angles = entry["angles"]
        if "correct" in label:
            if label not in angle_data_by_label:
                angle_data_by_label[label] = {a: [] for a in ANGLE_NAMES}
            for a in ANGLE_NAMES:
                val = angles.get(a, 0)
                if val is None:
                    val = 0
                angle_data_by_label[label][a].append(val)

    reference_stats = {}
    for label, angle_dict in angle_data_by_label.items():
        reference_stats[label] = {}
        for a, vals in angle_dict.items():
            if len(vals) > 0:
                mean_val = float(np.mean(vals))
                std_val = float(np.std(vals))
                reference_stats[label][a] = {
                    "mean": mean_val,
                    "std": std_val
                }
            else:
                reference_stats[label][a] = {"mean": 0.0, "std": 0.0}

    with open(REFERENCE_STATS_PATH, "w") as f:
        json.dump(reference_stats, f, indent=4)
    print(f"Reference stats saved at {REFERENCE_STATS_PATH}")

def train_model():
    train_dataset = ExerciseDataset(TRAIN_PATH)
    val_dataset = ExerciseDataset(VAL_PATH)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    input_size = len(ANGLE_NAMES)
    num_classes = len(train_dataset.label_mapping)
    with open(LABEL_MAPPING_PATH, "w") as f:
        json.dump({str(v): k for k, v in train_dataset.label_mapping.items()}, f, indent=4)

    model = ExerciseClassifier(input_size=input_size, num_classes=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    model_saved = False

    for epoch in range(10):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        for inputs, labels in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_accuracy = 100 * correct / total
        print(f"Epoch {epoch + 1}: Train Loss: {epoch_loss:.4f}, Accuracy: {train_accuracy:.2f}%")

        val_loss = 0
        val_correct = 0
        val_total = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_accuracy = 100 * val_correct / val_total
        print(f"Epoch {epoch + 1}: Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)
            model_saved = True
            print("Best model saved.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    if not model_saved:
        print("Saving the model as no best model was saved during training.")
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        torch.save(model.state_dict(), MODEL_PATH)

    # Compute reference stats
    compute_reference_stats(TRAIN_PATH)

if __name__ == "__main__":
    train_model()
