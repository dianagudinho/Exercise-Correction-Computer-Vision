import cv2
import mediapipe as mp
import torch
import json
import os
import numpy as np
from collections import deque
from angle_calculation import calculate_all_angles
from model import ExerciseClassifier, ANGLE_NAMES

MODEL_PATH = "models/exercise_classifier.pth"
LABEL_MAPPING_PATH = "data/processed/label_mapping.json"

if not os.path.exists(LABEL_MAPPING_PATH):
    print("Label mapping file not found. Please retrain the model.")
    exit()

with open(LABEL_MAPPING_PATH, "r") as f:
    LABEL_MAPPING = json.load(f)

required_features = len(ANGLE_NAMES)
num_classes = len(LABEL_MAPPING)
model = ExerciseClassifier(input_size=required_features, num_classes=num_classes)

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
except Exception as e:
    print(f"Error loading the model: {e}")
    exit()

def process_frame(frame, model, label_mapping):
    """
    Process a single frame to calculate angles, classify the pose, and return feedback.
    """
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=False, model_complexity=1) as pose:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            raw_landmarks = results.pose_landmarks.landmark
            landmarks = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in raw_landmarks]

            if len(landmarks) >= 33:
                angles = calculate_all_angles(landmarks)
                for angle in ANGLE_NAMES:
                    if angles.get(angle) is None:
                        angles[angle] = 0
                angle_values = [angles[angle] for angle in ANGLE_NAMES]
                input_tensor = torch.tensor([angle_values], dtype=torch.float32).to("cpu")
                with torch.no_grad():
                    prediction = model(input_tensor)
                    label_idx = prediction.argmax().item()
                    label = label_mapping.get(str(label_idx), "Unknown")
                    accuracy = torch.softmax(prediction, dim=1)[0, label_idx].item() * 100

                return {
                    "label": label,
                    "accuracy": accuracy,
                    "angles": angles
                }

    return None

def run_webcam():
    """
    Main function to access the webcam and provide real-time feedback.
    """
    print("Initializing webcam...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Unable to access the webcam.")
        exit()

    print("Webcam accessed successfully. Press 'q' to exit.")
    recent_labels = deque(maxlen=10)

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Unable to read frame from webcam.")
            break
        frame_resized = cv2.resize(frame, (640, 480))
        feedback = process_frame(frame_resized, model, LABEL_MAPPING)
        if feedback:
            label = feedback.get("label", "Unknown")
            accuracy = feedback.get("accuracy", 0)
            recent_labels.append(label)
            label_counts = {lbl: recent_labels.count(lbl) for lbl in set(recent_labels)}
            final_label = max(label_counts, key=label_counts.get)
            color = (0, 255, 0) if "correct" in final_label else (0, 0, 255)
            cv2.putText(frame_resized, f"Label: {final_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame_resized, f"Accuracy: {accuracy:.2f}%", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow("Exercise Posture Feedback", frame_resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam feed closed.")
