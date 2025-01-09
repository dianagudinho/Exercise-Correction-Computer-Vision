import torch
import torch.nn as nn
from torch.utils.data import Dataset
import json

ANGLE_NAMES = [
    "knee_angle", "elbow_angle", "hip_angle", "spine_angle",
    "right_knee_angle", "right_elbow_angle", "right_hip_angle", "right_spine_angle",
    "shoulder_angle", "pelvis_angle"
]

class ExerciseDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, "r") as f:
            data = json.load(f)

        self.samples = []
        self.labels = []
        self.label_mapping = {}
        label_counter = 0
        for entry in data:
            label = entry["label"]
            if label not in self.label_mapping:
                self.label_mapping[label] = label_counter
                label_counter += 1
            angles = entry["angles"]
            feature_vector = []
            for angle_name in ANGLE_NAMES:
                val = angles.get(angle_name, 0)
                if val is None:
                    val = 0
                feature_vector.append(val)

            self.samples.append(feature_vector)
            self.labels.append(self.label_mapping[label])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)


class ExerciseClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ExerciseClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.dropout = nn.Dropout(0.5)
        self.output = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(self.relu(self.bn2(self.fc2(x))))
        x = self.output(self.fc3(x))
        return x
