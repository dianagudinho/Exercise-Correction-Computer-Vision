import json
import os

def load_reference_stats(path="data/processed/reference_stats.json"):
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)

def get_angle_stats_for_exercise(reference_stats, exercise_label):
    return reference_stats.get(exercise_label, {})
