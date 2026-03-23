# ✋ Real-Time Gesture-Based Assistive Control System

![Python](https://img.shields.io/badge/Python-3.10-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Object%20Detection-orange)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand%20Tracking-green)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-red)
![PyAutoGUI](https://img.shields.io/badge/PyAutoGUI-Automation-yellow)
![Status](https://img.shields.io/badge/Status-Completed-success)

--

## 📌 Overview

This project implements a **real-time gesture-based assistive control system** using computer vision and deep learning.  
Users can control system functions such as **play/pause and volume adjustment** using simple hand gestures.

The system is designed with an **accessibility-first approach**, enabling interaction without physical input devices.

---

## 🎯 Features

- ✋ Real-time hand gesture recognition  
- 🎯 Hybrid approach using YOLOv8 + MediaPipe  
- ⚡ Low latency (<100 ms) real-time performance  
- 🔊 Audio feedback for interaction confirmation  
- 🧠 Gesture stabilization using temporal buffering  
- 🕒 Hold-based activation to prevent accidental triggers  
- 💻 System-level automation (play/pause, volume control)  
- 📊 Latency tracking and UI feedback  

---

## 🧠 Tech Stack

- Python  
- YOLOv8 (Object Detection)  
- MediaPipe (Hand Tracking)  
- OpenCV (Computer Vision)  
- PyAutoGUI (System Automation)  
- Pyttsx3 (Text-to-Speech)

---

## ⚙️ How It Works

1. YOLOv8 detects the hand region in real-time  
2. MediaPipe extracts hand landmarks  
3. Finger positions are analyzed to classify gestures:
   - Rock  
   - Paper  
   - Scissors  
4. Temporal buffering ensures stable gesture recognition  
5. Gesture must be held for a short duration to trigger action  
6. System executes mapped action (e.g., play/pause, volume)

---

## 🎮 Gesture Mapping

| Gesture | Action |
|--------|--------|
| Rock | Play / Pause |
| Paper | Volume Up |
| Scissors | Volume Down |

---

## 📊 Performance

- Average Latency: ~60–100 ms  
- Real-time detection on CPU  
- Effective range: ~0.5m – 1.5m  

---

## ♿ Accessibility Focus

This system is designed for users with upper limb mobility impairments, enabling hands-free interaction.

Key accessibility features:
- Gesture hold mechanism to reduce accidental triggers  
- Audio feedback for confirmation  
- Contactless interaction  
- Minimal physical effort required  

---

## 🧪 Robustness Testing

| Condition | Result |
|----------|--------|
| Bright lighting | High accuracy |
| Low lighting | Moderate performance |
| Background clutter | Minor noise |
| Distance variation | Stable within range |

---

## 🎥 Demo

Add your demo GIF here:

![Demo](demo.gif)

---

## 🚀 Installation

pip install ultralytics opencv-python mediapipe pyautogui pyttsx3

---

## ▶️ Run the Project

python rps_game.py

---

## 🏆 Key Highlights

- Real-time AI system integrating detection + tracking + automation  
- Designed with accessibility principles  
- Implements temporal smoothing for stability  
- End-to-end ML pipeline  

---

## 📌 Future Improvements

- Multi-hand gesture recognition  
- Custom gesture training  
- Mobile/web deployment  

---

## 👨‍💻 Author

Manasvi Acharya

---

⭐ If you like this project, give it a star!
