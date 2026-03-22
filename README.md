# Sign Language Detection using MediaPipe

A real-time sign language alphabet recognition system using MediaPipe for hand tracking and machine learning for classification.

## 📋 Overview

This project uses MediaPipe's hand tracking solution to detect hand landmarks and a Random Forest classifier to recognize American Sign Language (ASL) alphabet gestures. The system can detect and classify hand signs in real-time through a webcam.

## ✨ Features

- **Real-time hand tracking** using MediaPipe
- **Gesture recognition** for ASL alphabet (A-Z)
- **Data collection tool** for creating custom training datasets
- **Model training pipeline** with evaluation metrics
- **Live prediction** with confidence scores
- **Prediction smoothing** for stable results
- **Visual feedback** with hand landmark overlay

## 🛠️ Requirements

- Python 3.8 or higher
- Webcam/Camera
- Windows/Linux/macOS

## 📦 Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd "Sign language detection"
   ```

2. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

   The following packages will be installed:
   - opencv-python (for video capture and display)
   - mediapipe (for hand tracking)
   - numpy (for numerical operations)
   - scikit-learn (for machine learning)
   - matplotlib (for visualization)
   - pandas (for data handling)

## 🚀 Usage

The system works in three stages: **Data Collection** → **Model Training** → **Real-time Detection**

### Step 1: Collect Training Data

Run the data collection script to capture hand gestures for each letter:

```bash
python collect_data.py
```

**Instructions:**
1. The script will prompt you to select which letters to collect (A-Z or custom selection)
2. Choose how many samples per letter (default: 100)
3. For each letter:
   - Position your hand in the ASL sign for that letter
   - Press **SPACE** to start collecting samples
   - Hold the gesture steady while samples are collected
   - Press **q** to quit current collection if needed
4. Data is automatically saved after each letter in `data/sign_language_data.pkl`

**Tips for good data collection:**
- Use good lighting
- Keep your hand clearly visible
- Vary the hand position slightly during collection (different angles, distances)
- Use a plain background if possible
- Collect data from different people for better generalization

### Step 2: Train the Model

After collecting data, train the classifier:

```bash
python train_model.py
```

**The script will:**
1. Load the collected data
2. Display dataset statistics
3. Split data into training (80%) and testing (20%) sets
4. Train a Random Forest classifier
5. Evaluate the model and show accuracy metrics
6. Optionally perform cross-validation
7. Optionally plot confusion matrix and feature importance
8. Save the trained model to `models/sign_language_model.pkl`

**Expected output:**
- Accuracy score
- Classification report (precision, recall, F1-score)
- Confusion matrix visualization (optional)
- Feature importance plot (optional)

### Step 3: Real-time Detection

Run the detection script to recognize signs in real-time:

```bash
python detect_sign.py
```

**Controls:**
- **q**: Quit the application
- **c**: Clear prediction history (useful if predictions get stuck)

**Display elements:**
- Hand landmarks overlay
- Current predicted letter
- Confidence percentage
- Large letter display in top-right corner
- Color-coded confidence (Green: high, Yellow: medium, Orange: low)

## 📁 Project Structure

```
Sign language detection/
├── collect_data.py          # Data collection script
├── train_model.py           # Model training script
├── detect_sign.py           # Real-time detection script
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── data/                   # Training data (created automatically)
│   └── sign_language_data.pkl
└── models/                 # Trained models (created automatically)
    ├── sign_language_model.pkl
    ├── confusion_matrix.png
    └── feature_importance.png
```

## 🎯 How It Works

### 1. Hand Landmark Detection
- MediaPipe detects 21 hand landmarks (fingertips, joints, wrist, etc.)
- Each landmark has 3D coordinates (x, y, z)
- Total: 63 features per hand gesture

### 2. Feature Normalization
- Landmarks are normalized relative to the wrist position
- Scaling is applied to make the features translation and scale-invariant
- This allows the model to recognize signs regardless of hand size or distance from camera

### 3. Classification
- Random Forest classifier learns patterns from normalized landmarks
- During detection, new gestures are classified into alphabet letters
- Prediction smoothing uses a sliding window to stabilize results

### 4. Real-time Inference
- Webcam feed is processed frame-by-frame
- Hand landmarks are extracted and normalized
- Model predicts the sign with confidence score
- Results are displayed with visual feedback

## 📊 Model Performance

The model's performance depends on:
- **Data quality**: More varied and accurate training data = better results
- **Number of samples**: 100+ samples per letter recommended
- **Lighting conditions**: Consistent lighting improves accuracy
- **Hand positioning**: Clear, unobstructed hand view is essential

Expected accuracy with good training data: **85-95%**

## 🔧 Customization

### Adjust Hand Detection Sensitivity

In `collect_data.py` and `detect_sign.py`, modify:
```python
self.hands = mp_hands.Hands(
    min_detection_confidence=0.7,  # Lower for easier detection
    min_tracking_confidence=0.5     # Lower for smoother tracking
)
```

### Change Model Parameters

In `train_model.py`, adjust Random Forest parameters:
```python
self.model = RandomForestClassifier(
    n_estimators=100,      # Number of trees
    max_depth=20,          # Maximum tree depth
    min_samples_split=5,   # Minimum samples to split
    min_samples_leaf=2     # Minimum samples per leaf
)
```

### Adjust Prediction Smoothing

In `detect_sign.py`, modify:
```python
self.history_size = 5  # Number of predictions to consider (increase for more smoothing)
```

## 🐛 Troubleshooting

### Camera Not Working
- Ensure no other application is using the camera
- Try changing camera index in code: `cv2.VideoCapture(0)` → `cv2.VideoCapture(1)`

### Low Accuracy
- Collect more training data (200+ samples per letter)
- Ensure consistent hand positioning during data collection
- Use better lighting conditions
- Train with data from multiple people

### Hand Not Detected
- Improve lighting
- Move hand closer to camera
- Lower `min_detection_confidence` parameter
- Ensure hand is fully visible and not obstructed

### Predictions Jittery
- Increase `history_size` for more smoothing
- Increase `min_tracking_confidence` parameter
- Collect more consistent training data

## 📚 ASL Alphabet Reference

For proper hand signs, refer to ASL alphabet charts:
- [ASL Alphabet Chart](https://www.lifeprint.com/asl101/fingerspelling/abc-sign-language.htm)
- Practice each letter before collecting data
- Note: Letters J and Z involve motion (this system works best with static gestures)

## 🤝 Contributing

Improvements and suggestions are welcome! Consider:
- Adding more sign language gestures (words, phrases)
- Implementing two-hand gesture recognition
- Adding different classification models (SVM, Neural Networks)
- Supporting different sign languages (BSL, ISL, etc.)

## 📝 License

This project is free to use for educational and personal purposes.

## 🙏 Acknowledgments

- **MediaPipe** by Google for hand tracking solution
- **Scikit-learn** for machine learning tools
- ASL community for gesture references

## 📧 Support

For issues or questions:
1. Check the Troubleshooting section
2. Verify all dependencies are installed correctly
3. Ensure you have followed all steps in order (collect → train → detect)

---

**Happy Sign Language Detection! 🤟**#   S i g n - L a n g u a g e - R e c o g n i t i o n  
 