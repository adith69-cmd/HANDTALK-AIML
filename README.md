# HandTalk — Real-time Hand Gesture Recognition

A real-time hand gesture recognition system built with Python, OpenCV, and MediaPipe. Uses a custom-collected image dataset to train a classifier for three gestures: **Hi**, **Thanks**, and **No**.

## Features

- Real-time hand gesture detection and tracking via webcam
- Three-class gesture classifier: Hi, Thanks, No
- Custom self-collected dataset captured through a keyboard-triggered workflow
- Full pipeline: data collection → preprocessing → training → live inference
- On-screen visualization overlay showing detected landmarks

## How It Works

Webcam frame
    ↓
MediaPipe hand landmark detection
    ↓
Landmark feature extraction
    ↓
Trained classifier
    ↓
Gesture prediction (Hi / Thanks / No)

## Tech Stack

- **Python**
- **OpenCV** — frame capture and image processing
- **MediaPipe** — hand landmark detection
- **NumPy** — feature handling
- **scikit-learn** — classifier training and evaluation

## Files

- `collectdata.py` — captures training images via webcam for each gesture class
- `trainmodel.py` — preprocesses collected data and trains the classifier
- Inference script — loads the trained model and predicts gestures in real time

## Setup

Install dependencies:

pip install opencv-python mediapipe numpy scikit-learn


## Usage

**Data collection:**

python collectdata.py
- Press `h` to collect the **Hi** gesture
- Press `t` to collect the **Thanks** gesture
- Press `n` to collect the **No** gesture
- Press `r` to restart the current sequence
- Press `q` to quit

**Training:**
python trainmodel.py

## Motivation

Gesture-based interaction is a practical accessibility tool. HandTalk explores whether a lightweight, low-dependency pipeline (OpenCV + MediaPipe + scikit-learn) can recognize a small vocabulary of gestures in real time without specialized hardware.

## Limitations

- Currently supports only three gestures
- Tested on a small self-collected dataset; out-of-distribution performance is untested
- Real-time accuracy depends on lighting and camera quality

## Future Improvements

- Expand gesture vocabulary to the full alphabet or common phrases
- Add temporal modeling (LSTM / 1D CNN over landmark sequences) for dynamic gestures
- Evaluate against a public dataset for comparability

---

**Built with Python + OpenCV + MediaPipe**
