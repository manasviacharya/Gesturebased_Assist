# 🎮 AI Rock Paper Scissors Game (YOLO + MediaPipe)

![Python](https://img.shields.io/badge/Python-3.10-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Object%20Detection-orange)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand%20Tracking-green)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-red)

---

## 📌 Overview

A real-time **Rock Paper Scissors game** powered by computer vision.  
Players use hand gestures in front of a webcam to compete against the computer.

The system combines:
- YOLOv8 → hand detection  
- MediaPipe → landmark-based gesture recognition  

---

## 🎯 Features

- 🎮 Play Rock-Paper-Scissors using hand gestures  
- 🤖 Computer opponent with random moves  
- ⏱ Countdown system before each round  
- 📊 Live score tracking  
- 🧠 Hybrid detection (YOLO + MediaPipe)  
- ⚡ Real-time performance  

---

## 🧠 Tech Stack

- Python  
- YOLOv8 (Object Detection)  
- MediaPipe (Hand Tracking)  
- OpenCV  

---

## ⚙️ How It Works

1. YOLO detects the hand region  
2. Bounding box is expanded for better accuracy  
3. MediaPipe extracts hand landmarks  
4. Gesture is classified:
   - Rock  
   - Paper  
   - Scissors  
5. Game logic determines winner  
6. Score updates in real-time  

---

## 🎮 Controls

| Key | Action |
|-----|--------|
| S | Start round |
| Q | Quit game |

---

## 🎥 Demo

Add your demo GIF here:

![Demo](RPS.gif)

---

## 🚀 Installation

pip install ultralytics opencv-python mediapipe

---

## ▶️ Run

python rps_game.py

---

## 🏆 Highlights

- Real-time AI-powered game  
- Hybrid detection pipeline  
- Interactive UI with computer vision  
- End-to-end ML + application integration  

---

## 📌 Future Improvements

- Multiplayer mode  
- Better gesture classification  
- UI using Tkinter / PyQt  
- Sound effects & animations  

---

## 👨‍💻 Author

Manasvi Acharya

---

