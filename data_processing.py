import os
import json
import cv2
import numpy as np
import time
from sklearn.model_selection import train_test_split
import mediapipe as mp
from angle_calculation import calculate_all_angles
from model import ANGLE_NAMES

VIDEO_ROOT = "data/annotations"
OUTPUT_PATH = "data/processed"

mp_pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False)

def extract_frame_data_from_video(video_path, label):
    cap = cv2.VideoCapture(video_path)
    frame_data = []

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_pose.process(frame_rgb)

        if results.pose_landmarks:
            raw_landmarks = results.pose_landmarks.landmark
            landmarks = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in raw_landmarks]
            if len(landmarks) < 33:
                continue

            angles = calculate_all_angles(landmarks)
            for a in ANGLE_NAMES:
                if angles.get(a) is None:
                    angles[a] = 0

            frame_data.append({"label": label, "angles": angles})

    cap.release()
    return frame_data

def process_videos(video_root):
    dataset = []
    valid_extensions = (".mp4", ".avi", ".mov")
    start_time = time.time()
    for category in os.listdir(video_root):
        cat_path = os.path.join(video_root, category)
        if not os.path.isdir(cat_path):
            continue
        label = category
        for video in os.listdir(cat_path):
            video_path = os.path.join(cat_path, video)
            if video_path.lower().endswith(valid_extensions):
                print(f"Processing: {video_path}")
                frames = extract_frame_data_from_video(video_path, label)
                dataset.extend(frames)

    elapsed_time = time.time() - start_time
    print(f"Processed all videos in {elapsed_time:.2f} seconds")
    return dataset

def save_datasets(dataset, output_path):
    if not dataset:
        print("No data to save!")
        return

    train_val, test = train_test_split(dataset, test_size=0.1, random_state=42)
    train, val = train_test_split(train_val, test_size=0.11, random_state=42)

    os.makedirs(output_path, exist_ok=True)

    with open(os.path.join(output_path, "train.json"), "w") as f:
        json.dump(train, f, indent=4)
    with open(os.path.join(output_path, "val.json"), "w") as f:
        json.dump(val, f, indent=4)
    with open(os.path.join(output_path, "test.json"), "w") as f:
        json.dump(test, f, indent=4)
    labels = set([entry["label"] for entry in dataset])
    label_mapping = {i: l for i, l in enumerate(sorted(labels))}
    rev_mapping = {str(i): l for i, l in label_mapping.items()}

    with open(os.path.join(output_path, "label_mapping.json"), "w") as f:
        json.dump(rev_mapping, f, indent=4)

    print(f"Train size: {len(train)}, Validation size: {len(val)}, Test size: {len(test)}")

if __name__ == "__main__":
    dataset = process_videos(VIDEO_ROOT)
    save_datasets(dataset, OUTPUT_PATH)
