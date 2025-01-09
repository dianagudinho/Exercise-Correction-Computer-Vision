# Exercise-Correction-Computer-Vision

**Run the Application**:  
pip install -r requirements.txt

streamlit run src/app.py

browse the files in code/data/test folder select any video for processing

# Intellifit: AI-Powered Exercise Feedback 🏋️

**Intellifit** is an advanced AI-powered application that provides real-time and post-analysis feedback on exercise posture using deep learning and pose estimation techniques. The user-friendly interface is designed to help users identify and correct their workout form, enhancing performance and reducing injury risks.

---
## Features 🚀

- **Upload Video for Analysis**:
  - Upload your exercise video to receive feedback on posture.
  - Compare your form with ideal angles.
- **Real-Time Webcam Feedback**:
  - Use your webcam for live analysis and instant posture feedback.
- **Feedback Metrics**:
  - ✅ **Good Alignment**: Close to ideal angles.
  - 🟡 **Slightly Off**: Minor deviations to adjust.
  - 🔴 **Needs Improvement**: Significant deviation needing correction.
- **Interactive UI**:
  - Modern and responsive interface with animations.
  - Visual charts and comparison tables for better understanding.

---

## Tech Stack 🛠️

**Backend**:  
- **PyTorch**: Deep learning model for inference.  
- **MediaPipe**: Pose estimation to detect joint angles.  
- **OpenCV**: Video processing for real-time analysis.  

**Frontend**:  
- **Streamlit**: For creating an interactive and user-friendly interface.  
- **Lottie Animations**: To make the UI engaging.  
- **Matplotlib & Seaborn**: Visualizations for angle comparisons.

---

## Prerequisites 📋

Ensure the following are installed:

- **Python 3.8 or above**
- `pip` for package management
- A compatible GPU setup (optional, improves performance)

---

## Installation 🛠️

1. **Clone the Repository**:
git clone 
cd intellifit
2. **Install Dependencies**:
pip install -r requirements.txt

3. **Run the Application**:  
streamlit run src/app.py

4. **Project Structure 📂**:

intellifit/
├── data/
│   ├── processed/                  # Processed video outputs
│   ├── reference_stats.json        # Reference statistics for exercises
│   └── label_mapping.json          # Label mappings for exercises
├── logs/                           # Logs for debugging and analysis
├── models/
│   └── exercise_classifier.pth     # Pre-trained PyTorch model
├── project_sample_images/          # Sample images for demonstration
├── rds/                            # Any RDS or database integration files
├── reports/                        # Generated reports or analysis
├── src/
│   ├── __init__.py                 # Initialization for src package
│   ├── angle_calculation.py        # Calculate joint angles from pose landmarks
│   ├── data_processing.py          # Preprocessing video and pose data
│   ├── feedback.py                 # Generate feedback based on analysis
│   ├── model.py                    # PyTorch model definition
│   ├── reference_stats.py          # Load reference statistics
│   ├── train_model.py              # Script to train the model
│   └── webcam.py                   # Real-time webcam feedback
├── app.py                          # Main application code
├── requirements.txt                # Dependencies list
├── README.md                       # Documentation
└── .gitignore                      # Ignore unnecessary files

5.**Usage Guide 📖**:
1. Upload Video Mode:
Navigate to the Upload Video tab.
Upload an exercise video (.mp4, .avi, .mov).
Receive:
A processed video with detected pose landmarks.
Feedback messages based on your joint angles.
Bar charts comparing your angles with ideal posture.
2. Real-Time Feedback Mode:
Navigate to the Live Feedback tab.
Click "Start Webcam".
Perform exercises and receive instant, color-coded feedback on your posture.
Feedback Metrics 📊
✅ Good Alignment: Angles within acceptable deviation.
🟡 Slightly Off: Minor adjustments recommended.
🔴 Needs Improvement: Significant deviation from ideal posture.

Angle Comparison:
Visual charts to compare your angles against reference statistics.

Screenshots 📸
Home Page

Upload Video Analysis

Real-Time Webcam Feedback

Dependencies 📦
The required dependencies are listed in requirements.txt:

Install using:
pip install -r requirements.txt

Future Enhancements 🌟
Support for additional exercises and complex movements.
Integration with wearable fitness devices.
Enhanced pose estimation using advanced neural networks.
User progress tracking and analytics dashboard.

Contribution 🤝
We welcome contributions! Here's how you can help:

Fork the repository.
Create a new feature branch:
git checkout -b feature-name

Commit your changes:
git commit -m "Add feature-name"

Push to your branch:
git push origin feature-name
Open a pull request.

License 📄
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments 🙏
MediaPipe: For accurate pose detection.
PyTorch: Building and training deep learning models.
Streamlit: For developing an interactive user interface.
LottieFiles: For animations that enhance the UI experience.

💡 Intellifit: Train smart, stay fit! 🏋️‍♂️
