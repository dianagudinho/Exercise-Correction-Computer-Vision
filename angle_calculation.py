import math
import numpy as np

def calculate_angle(p1, p2, p3):
    try:
        p1_arr = np.array([p1['x'], p1['y'], p1['z']])
        p2_arr = np.array([p2['x'], p2['y'], p2['z']])
        p3_arr = np.array([p3['x'], p3['y'], p3['z']])
        v1 = p1_arr - p2_arr
        v2 = p3_arr - p2_arr
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        if v1_norm == 0 or v2_norm == 0:
            return None
        v1 = v1 / v1_norm
        v2 = v2 / v2_norm
        dot_product = np.dot(v1, v2)
        angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
        return math.degrees(angle)
    except Exception as e:
        print(f"Error calculating angle: {e}")
        return None

def calculate_all_angles(landmarks):
    if not landmarks or len(landmarks) < 33:
        return {}

    try:
        hip_l, knee_l, ankle_l = landmarks[23], landmarks[25], landmarks[27]
        shoulder_l, elbow_l, wrist_l = landmarks[11], landmarks[13], landmarks[15]
        hip_r, knee_r, ankle_r = landmarks[24], landmarks[26], landmarks[28]
        shoulder_r, elbow_r, wrist_r = landmarks[12], landmarks[14], landmarks[16]
        spine, neck = landmarks[24], landmarks[0]

        angles = {
            "knee_angle": calculate_angle(hip_l, knee_l, ankle_l),
            "elbow_angle": calculate_angle(shoulder_l, elbow_l, wrist_l),
            "hip_angle": calculate_angle(shoulder_l, hip_l, knee_l),
            "spine_angle": calculate_angle(neck, spine, hip_l),
            "right_knee_angle": calculate_angle(hip_r, knee_r, ankle_r),
            "right_elbow_angle": calculate_angle(shoulder_r, elbow_r, wrist_r),
            "right_hip_angle": calculate_angle(shoulder_r, hip_r, knee_r),
            "right_spine_angle": calculate_angle(neck, spine, hip_r),
            "shoulder_angle": calculate_angle(shoulder_l, spine, shoulder_r),
            "pelvis_angle": calculate_angle(hip_l, spine, hip_r),
        }

        return angles
    except Exception as e:
        print(f"Unexpected error while calculating angles: {e}")
        return {}
