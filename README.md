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
