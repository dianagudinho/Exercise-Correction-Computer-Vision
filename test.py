import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import json
import numpy as np
from model import ExerciseClassifier, ExerciseDataset, ANGLE_NAMES

# This test file is  showing accuracy for overall classification combining all frames classification which is not the case and 
# accuracy do not reflect the overall techniques we applied post on the model results made our results more accurate.
# We applied the Major vote Technique which brought us great results.  
TEST_PATH = "data/processed/test.json"
MODEL_PATH = "models/exercise_classifier.pth"
LABEL_MAPPING_PATH = "data/processed/label_mapping.json"

def load_label_mapping(path):
    with open(path, "r") as f:
        return {int(k): v for k, v in json.load(f).items()}
    
def evaluate_model():
    test_dataset = ExerciseDataset(TEST_PATH)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    label_mapping = load_label_mapping(LABEL_MAPPING_PATH)
    input_size = len(ANGLE_NAMES)
    num_classes = len(label_mapping)
    model = ExerciseClassifier(input_size=input_size, num_classes=num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()
    criterion = nn.CrossEntropyLoss()
    all_labels = []
    all_predictions = []
    total_loss = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.numpy())
            all_predictions.extend(predicted.numpy())
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    print("\n--- Model Evaluation Metrics ---")
    print(f"Test Loss: {total_loss / len(test_loader):.4f}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nLabel Mapping:")
    for idx, label in label_mapping.items():
        print(f"{idx}: {label}")

if __name__ == "__main__":
    evaluate_model()
