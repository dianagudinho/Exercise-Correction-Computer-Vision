
import cv2
import mediapipe as mp
import torch
import numpy as np
from collections import deque
from angle_calculation import calculate_all_angles
from model import ANGLE_NAMES

def process_uploaded_video(
    video_path,
    model,
    label_mapping,
    output_path="data/processed/processed_video.mp4",
    frame_skip=1,
    window_size=10,
    line_thickness=4,    
    circle_radius=6,     
    font_scale=2,        
    font_thickness=3     
):
    """
    Process a video for classification and gather feedback.
    Draw landmarks and label frames.
    Collect angle data for correct and incorrect frames.
    """

    if video_path == "webcam":
        cap = cv2.VideoCapture(0)
        save_output = False
        print("[INFO] Using webcam mode. Press 'q' to quit.")
    else:
        cap = cv2.VideoCapture(video_path)
        save_output = True
        print(f"[INFO] Processing video from file: {video_path}")

    if save_output:
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps == 0:
            fps = 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'X264')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    else:
        out = None

    feedback_results = {
        "total_frames": 0,
        "processed_frames": 0,
        "correct_frames": 0,
        "incorrect_frames": 0,
        "frame_details": [],
        "accuracy": 0.0,
        "correct_angles": {a: [] for a in ANGLE_NAMES},
        "incorrect_angles": {a: [] for a in ANGLE_NAMES}
    }

    frame_count = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    recent_labels = deque(maxlen=window_size)

    with mp_pose.Pose(static_image_mode=False, model_complexity=1) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_count += 1
            if frame_skip > 1 and frame_count % frame_skip != 0:
                if save_output and out is not None:
                    out.write(frame)
                continue

            feedback_results["total_frames"] += 1
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            current_label = None
            angles = {}

            if results.pose_landmarks:
                raw_landmarks = results.pose_landmarks.landmark
                landmarks = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in raw_landmarks]

                if len(landmarks) >= 33:
                    angles = calculate_all_angles(landmarks)
                    for a in ANGLE_NAMES:
                        if angles.get(a) is None:
                            angles[a] = 0

                    angle_values = [angles[a] for a in ANGLE_NAMES]
                    input_tensor = torch.tensor([angle_values], dtype=torch.float32).to(device)
                    try:
                        prediction = model(input_tensor)
                        label_idx = prediction.argmax().item()
                        if str(label_idx) in label_mapping:
                            current_label = label_mapping[str(label_idx)]
                        else:
                            current_label = list(label_mapping.values())[label_idx]

                        feedback_results["processed_frames"] += 1
                        if "correct" in current_label:
                            feedback_results["correct_frames"] += 1
                            for a in ANGLE_NAMES:
                                feedback_results["correct_angles"][a].append(angles[a])
                        else:
                            feedback_results["incorrect_frames"] += 1
                            for a in ANGLE_NAMES:
                                feedback_results["incorrect_angles"][a].append(angles[a])

                        feedback_results["frame_details"].append({
                            "frame": frame_count,
                            "angles": angles,
                            "label": current_label
                        })
                    except Exception as e:
                        print(f"Error during model inference: {e}")

                mp_drawing.draw_landmarks(
                    frame, 
                    results.pose_landmarks, 
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0,255,0), thickness=line_thickness, circle_radius=circle_radius),
                    mp_drawing.DrawingSpec(color=(0,0,255), thickness=line_thickness, circle_radius=circle_radius)
                )

            if current_label is not None:
                recent_labels.append(current_label)
            final_label = None
            if len(recent_labels) > 0:
                label_counts = {}
                for lbl in recent_labels:
                    label_counts[lbl] = label_counts.get(lbl, 0) + 1
                final_label = max(label_counts, key=label_counts.get)
            else:
                final_label = current_label

            if final_label is not None:
                if "correct" in final_label:
                    color = (0, 255, 0)  
                else:
                    color = (0, 0, 255)
                cv2.putText(frame, final_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)

            if save_output and out is not None:
                out.write(frame)
            else:
                cv2.imshow("Live Feedback", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cap.release()
    if save_output and out is not None:
        out.release()
        print("[INFO] Processed video saved at:", output_path)
    if video_path == "webcam":
        cv2.destroyAllWindows()

    if feedback_results["processed_frames"] > 0:
        feedback_results["accuracy"] = (
            feedback_results["correct_frames"] / feedback_results["processed_frames"] * 100
        )
    else:
        print("[WARNING] No frames with pose were processed. No meaningful output video.")

    return output_path if save_output else None, feedback_results
