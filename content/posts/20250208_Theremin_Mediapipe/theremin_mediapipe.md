+++
title = "Building a Virtual Theremin with MediaPipe and Pure Data"
date = 2025-02-07T19:52:15+07:00
draft = false
summary = "Using hand tracking to create a virtual theremin with MediaPipe and Pure Data."
tags = ["MediaPipe", "Computer Vision", "Hand Tracking", "Gesture Control", "Pure Data", "OpenCV", "OSC", "Creative Coding", "Interactive Music", "Machine Learning", "Python"]
+++

I recently worked on a fun project where I used **MediaPipe** for finger tracking and interfaced it with **Pure Data** to create a simple virtual theremin. The idea was to control pitch and volume using hand movements, without touching any physical object.

This blog post provides an overview of the project, the steps I followed, and a few code snippets to illustrate key aspects of the implementation.

---

## Demo Video

Before diving into the details, check out a quick demo of the theremin in action:

<video width="100%" controls>
  <source src="/videos/theremin_puredata.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>



## Project Overview

A theremin is a touchless musical instrument that produces sound based on the position of the player's hands. For this project, I used:
- **MediaPipe Hand Tracking** to detect finger positions
- **OpenCV** to visualize the hand movements
- **OSC (Open Sound Control)** to send data to **Pure Data**, which handled the sound synthesis

The result is a simple but effective virtual instrument that lets you manipulate sound using only hand gestures.

---

## How It Works

1. **Track Hand Landmarks:** Using MediaPipe, we detect hands and extract the positions of key landmarks (fingertips, wrist, etc.).
2. **Define Control Areas:** We set up an **ON/OFF button** on the screen to enable or disable sound.
3. **Map Hand Movements to Sound Parameters:**
   - Left hand controls **volume** (vertical movement).
   - Right hand controls **pitch** (horizontal movement).
4. **Send Data to Pure Data:** We use OSC messages to send pitch and volume values to a Pure Data patch, where they are converted into sound. The patch takes these values, processes them, and routes them to an oscillator and an amplitude controller, translating hand gestures into musical notes. This setup mimics the behavior of a real theremin, producing pitch and volume variations.

---

## Key Code Snippets

### 1. Tracking Hand Landmarks with MediaPipe

```python
import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Process landmarks here
            pass
    cv2.imshow("Hand Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
```

This snippet initializes the webcam, processes frames, and detects hands in real time.

---

### 2. Detecting Button Presses for ON/OFF Controls

```python
def check_button_press(landmarks, center, radius, height, width):
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    index_coords = (int(index_tip.x * width), int(index_tip.y * height))
    middle_coords = (int(middle_tip.x * width), int(middle_tip.y * height))
    dist_index = np.linalg.norm(np.array(index_coords) - np.array(center))
    dist_middle = np.linalg.norm(np.array(middle_coords) - np.array(center))
    return dist_index <= radius and dist_middle <= radius
```

This function checks whether both the index and middle fingers are inside a circular region, acting as an ON/OFF switch.

---

### 3. Mapping Hand Movements to Sound Parameters

```python
def map_hand_to_pitch_or_volume(handedness, index_tip_coords, ring_tip_coords, height, width):
    if handedness == 'Right':  # Left hand for volume
        normalized_y = ring_tip_coords[1] / height
        volume = min(max((1 - normalized_y) ** 2, 0), 1)
        return None, volume
    elif handedness == 'Left':  # Right hand for pitch
        normalized_x = index_tip_coords[0] / width
        pitch = int(127 * (normalized_x))  # Map to MIDI range
        return pitch, None
    return None, None
```

This function maps **vertical movement** of the left hand to **volume** and **horizontal movement** of the right hand to **pitch**.

---

## Final Thoughts

This was a blast to build! There’s something very satisfying about making noise by just waving your hands around like some kind of musical wizard. I’m keeping the details of the Pure Data setup out of this post for brevity (and hey, a little mystery never hurt anyone), but the core idea is simple: detect hands, map motion to sound, and have fun.

If you have any thoughts, ideas, or want to collaborate to expand this project, reach out! I’d love to hear what you think and see where we can take this next.

---

