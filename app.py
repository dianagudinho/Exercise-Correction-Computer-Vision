import os
import streamlit as st
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from feedback import process_uploaded_video
from model import ExerciseClassifier, ANGLE_NAMES
from webcam import run_webcam
from reference_stats import load_reference_stats
from streamlit_lottie import st_lottie

# --- ENVIRONMENT VARIABLES ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Intellifit: AI Exercise Feedback",
    layout="wide",
    page_icon="🏋️"
)

# --- PATHS ---
MODEL_PATH = "models/exercise_classifier.pth"
LABEL_MAPPING_PATH = "data/processed/label_mapping.json"
REFERENCE_STATS_PATH = "data/processed/reference_stats.json"

# --- LOAD MODEL AND LABEL MAPPING ---
if not os.path.exists(LABEL_MAPPING_PATH):
    st.error("Label mapping file not found. Please retrain the model.")
    st.stop()

with open(LABEL_MAPPING_PATH, "r") as f:
    LABEL_MAPPING = json.load(f)

required_features = len(ANGLE_NAMES)
num_classes = len(LABEL_MAPPING)
model = ExerciseClassifier(input_size=required_features, num_classes=num_classes)

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

reference_stats = load_reference_stats(REFERENCE_STATS_PATH)

# --- STYLING ---
st.markdown("""
<style>
body {
    font-family: 'Roboto', sans-serif !important;
    background: linear-gradient(to bottom, #e3f2fd, #e1bee7) !important;
    color: #333333 !important;
}
.stApp {
    background: linear-gradient(to right, #e3f2fd, #e1bee7) !important;
}
h1, h2, h3 {
    color: #333333 !important;
    font-weight: 600 !important;
}
.section-title {
    font-size: 1.6rem;
    font-weight: bold;
    margin-bottom: 1rem;
    color: #4e4e4e;
}
.feedback-box {
    background-color: #ffffff;
    padding: 15px;
    border-radius: 8px;
    box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
    margin-bottom: 1rem;
}
.feedback-list {
    list-style: none;
    padding-left: 0;
}
.feedback-list li {
    margin-bottom: 0.5rem;
    font-size: 1rem;
}
.angle-table {
    border-collapse: collapse;
    width: 100%;
    font-size: 0.9rem;
}
.angle-table th, .angle-table td {
    padding: 8px;
    text-align: left;
    border-bottom: 1px solid #ddd;
}
.angle-table th {
    background-color: #f5f5f5;
    font-weight: bold;
}
.success { color: #388e3c; }
.warning { color: #f57c00; }
.error { color: #d32f2f; }
</style>
""", unsafe_allow_html=True)

# --- LOTTIE ANIMATION ---
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_url = "https://assets9.lottiefiles.com/packages/lf20_9wpyhdzo.json"
lottie_json = load_lottieurl(lottie_url)

def generate_feedback(exercise_label, user_means):
    """Generate a list of feedback messages based on deviation from reference stats."""
    if exercise_label not in reference_stats:
        return ["No reference data available for this exercise."]
    msgs = []
    stats = reference_stats[exercise_label]
    for angle in ANGLE_NAMES:
        if angle in stats:
            mean_val = stats[angle]["mean"]
            std_val = stats[angle]["std"]
            user_val = user_means.get(angle, 0)
            if std_val == 0:
                continue
            diff = abs(user_val - mean_val)
            z_score = diff / std_val
            if z_score < 1:
                msgs.append(f"✅ <b>{angle}</b>: Good alignment")
            elif z_score < 2:
                msgs.append(f"🟡 <b>{angle}</b>: Slightly off")
            else:
                msgs.append(f"🔴 <b>{angle}</b>: Needs significant improvement")
    return msgs or ["All angles look great!"]

def plot_angle_chart(user_means, exercise_label):
    """Plot a side-by-side bar chart comparing user's angles to reference means."""
    if exercise_label not in reference_stats:
        st.warning("No reference data available.")
        return

    stats = reference_stats[exercise_label]
    angles, ref_vals, user_vals = [], [], []

    for a in ANGLE_NAMES:
        if a in stats:
            angles.append(a)
            ref_vals.append(stats[a]["mean"])
            user_vals.append(user_means.get(a, 0.0))

    if not angles:
        st.write("No comparable angles found.")
        return

    x = np.arange(len(angles))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8,4))
    ax.barh(x - width/2, ref_vals, width, label="Reference", color="#64b5f6")
    ax.barh(x + width/2, user_vals, width, label="You", color="#ffa726")

    ax.set_yticks(x)
    ax.set_yticklabels(angles, fontsize=11)
    ax.invert_yaxis() 
    ax.set_xlabel("Angle (°)", fontsize=11)
    ax.set_title("Angle Comparison", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

def compute_exercise_label_and_avgs(feedback):
    """Determine the exercise label from frames and compute average angles."""
    labels_count = {}
    for fd in feedback["frame_details"]:
        lbl = fd["label"]
        labels_count[lbl] = labels_count.get(lbl, 0) + 1

    if labels_count:
        exercise_label = max(labels_count, key=labels_count.get)
    else:
        exercise_label = "unknown_exercise"
    def avg_dict(d):
        return {a: float(np.mean(vals)) if vals else 0.0 for a, vals in d.items()}
    user_correct_avg = avg_dict(feedback["correct_angles"])
    user_incorrect_avg = avg_dict(feedback["incorrect_angles"])

    return exercise_label, user_correct_avg, user_incorrect_avg

def upload_mode():
    st.markdown("<h2 class='section-title'>Upload Your Exercise Video</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload your video (mp4, avi, mov)", type=["mp4", "avi", "mov"])
    frame_skip = 5

    if uploaded_file:
        video_path = os.path.join("data", "processed", "uploaded_video.mp4")
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.info("Processing your video... Please wait.")
        with st.spinner("Analyzing..."):
            processed_path, feedback = process_uploaded_video(
                video_path,
                model,
                LABEL_MAPPING,
                output_path="data/processed/processed_video.mp4",
                frame_skip=frame_skip,
            )

        if processed_path and os.path.exists(processed_path):
            st.video(processed_path)
        else:
            st.warning("No processed video generated. Possibly no pose detected.")

        exercise_label, user_correct_avg, user_incorrect_avg = compute_exercise_label_and_avgs(feedback)
        if feedback["correct_frames"] > 0:
            chosen_avg = user_correct_avg
        else:
            chosen_avg = user_incorrect_avg

        feedback_msgs = generate_feedback(exercise_label, chosen_avg)

        st.markdown("<h3 class='section-title'>Feedback</h3>", unsafe_allow_html=True)
        st.markdown("<div class='feedback-box'>", unsafe_allow_html=True)
        st.markdown("<ul class='feedback-list'>", unsafe_allow_html=True)
        for msg in feedback_msgs:
            st.markdown(f"<li>{msg}</li>", unsafe_allow_html=True)
        st.markdown("</ul></div>", unsafe_allow_html=True)

        st.markdown("<h3 class='section-title'>Angle Comparison</h3>", unsafe_allow_html=True)
        plot_angle_chart(chosen_avg, exercise_label)

        with st.expander("Advanced Details"):
            st.write("**Average Angles:**")
            st.write("<table class='angle-table'><tr><th>Angle</th><th>Value (°)</th></tr>", unsafe_allow_html=True)
            for k, v in chosen_avg.items():
                st.write(f"<tr><td>{k}</td><td>{v:.2f}</td></tr>", unsafe_allow_html=True)
            st.write("</table>", unsafe_allow_html=True)

            st.write("**Frame-by-Frame Details:**")
            for fd in feedback.get("frame_details", []):
                label_class = "good" if "correct" in fd['label'] else "error"
                st.markdown(f"**Frame {fd['frame']} - <span class='{label_class}'>{fd['label']}</span>**", unsafe_allow_html=True)
                st.json(fd["angles"])

        st.markdown("### Next Steps")
        st.write("- Adjust your form based on the suggestions above.")
        st.write("- Re-record and see if your highlighted angles improve.")
        st.success("Keep practicing! 💪")

def webcam_mode():
    st.markdown("<h2 class='section-title'>Real-Time Feedback</h2>", unsafe_allow_html=True)
    st.write("Start live feedback with your webcam. Press 'q' to quit when done.")
    if st.button("Start Webcam"):
        run_webcam()

def main():
    st_lottie(lottie_json, height=200, key="animation")
    st.title("🏋️ Intellifit: AI-Powered Exercise Evaluation")

    tabs = st.tabs(["📤 Upload Video", "🎥 Live Feedback"])
    with tabs[0]:
        upload_mode()
    with tabs[1]:
        webcam_mode()

if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)
    main()
