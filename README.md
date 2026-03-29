# Animal Detection on Roads
### Using YOLOe-26s Open-Vocabulary Object Detection

> A real-time road safety system that detects animal intrusions on highways, tracks individuals across frames, and triggers audio alerts — built with YOLOe-26s-seg and OpenCV.

---

## Overview

Animal intrusions on roads are a major cause of accidents, especially at night and in low-visibility conditions. This project implements an automated detection and alert system that:

- Detects **16 animal classes** from a live or recorded video feed
- **Tracks individual animals** across frames with persistent IDs
- Fires **distinct audio alerts** — one beep on detection, two beeps when the road is clear
- Produces a **fully annotated output video** with bounding boxes, class labels, confidence scores, track IDs, and a live stats panel

---

## Model: YOLOe-26s-seg

Unlike standard YOLO models (YOLOv8, YOLOv9) which use fixed COCO-80 vocabularies, **YOLOe** supports **open-vocabulary detection** via text prompts. Target classes are encoded into detection embeddings through a language-vision alignment mechanism at runtime — no retraining required.

**Animal classes used as prompts:**
```
cow, sheep, duck, goat, dog, cat, horse, deer,
elephant, fox, rabbit, bird, camel, buffalo, zebra, tiger
```

---

## System Architecture

```
Video Input (MP4)
      │
      ▼
Frame Extraction (OpenCV)
      │
      ▼
YOLOe-26s-seg  ←── Text Prompts (16 animal classes)
  model.track()
      │
      ▼
Bounding Box + Track ID Assignment
      │
      ▼
Majority-Vote Label Stabilizer  (prevents per-track label flicker)
      │
      ▼
Alert State Machine
  ├── CLEAR → ANIMAL DETECTED  →  🔔 1 Beep
  └── ANIMAL DETECTED → CLEAR  →  🔔🔔 2 Beeps  (after 75-frame grace period)
      │
      ▼
Annotated Frame Writer (OpenCV VideoWriter)
      │
      ▼
Audio + Video Mux (FFmpeg)
      │
      ▼
Output MP4 with Embedded Alert Track
```

---

## Setup & Usage

### 1. Open in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/18XmPhchOToIMH76vB5NUqvpvJeQ9fxBo?usp=sharing)

### 2. Install Dependencies

```bash
pip install ultralytics opencv-python
apt-get install -y ffmpeg libsndfile1
```

### 3. Add Your Video

Upload your road video (e.g., `night.mp4`) to the Colab session, then update:

```python
VIDEO_INPUT = "your_video.mp4"
```

### 4. Run the Notebook

Execute all cells. The system will:
- Load the YOLOe-26s-seg model
- Process each frame with tracking
- Write an annotated output video
- Mux audio alerts into the final MP4
- Auto-download `animal_detection_with_audio.mp4`

---

## Results Summary

Five real-world video samples were tested across varied conditions:

| Sample | Scene | Animals Present | Detected | Alert |
|--------|-------|----------------|----------|-------|
| 1 | Herd crossing road (day) | Cows | 13 Cow + 1 Horse (misclassified) | ✅ |
| 2 | Single animal at frame edge (day) | Elephant | Elephant + Dog (partial-body error) | ✅ |
| 3 | Mixed group, fast motion (day) | Zebras + Deer | 20 Zebras, 0 Deer (missed) | ✅ |
| 4 | Sudden appearance at night | Dog | Dog | ✅ |
| 5 | Animal walking at night | Elephant | Elephant + Cow (false positive) | ✅ |

### Key Findings

| Issue | Root Cause | Severity |
|-------|-----------|----------|
| Cow/Horse confusion | Embedding space proximity | Low |
| Elephant → Dog (partial body) | Incomplete visual feature vector | High |
| Deer missed entirely | Small size + motion blur | Critical |
| Zebra: 100% accuracy | High-contrast stripe pattern | ✅ Positive |
| Night-time dog detection | Embedding robust under low light | ✅ Positive |
| Alert state machine | Correct across all 5 samples | ✅ No issues |

---

## Key Design Decisions

**Majority-Vote Label Stabilizer** — Each track ID accumulates a class vote histogram. The most-voted class wins, preventing flickering labels across frames.

**75-Frame Grace Period** — The "road clear" signal is only fired after 75 consecutive frames with no detections (~3 seconds at 25 fps). This avoids false all-clears during brief detection gaps with fast-moving animals.

**Confidence Threshold: 0.40** — Balances sensitivity and false positive rate. A lower threshold increases recall for small/occluded animals but risks phantom detections.

---

## Known Limitations

- **Partial visibility** at frame edges causes misclassification (entry-zone failure)
- **Small animals in fast motion** fall below confidence threshold (missed detections)
- **Inter-class confusion** between morphologically similar species (cow ↔ horse, elephant ↔ cow)
- **Night-time conditions** can cause secondary false-positive bounding boxes on large animals

---

## Future Work

- [ ] Edge-zone confidence relaxation for animals entering from frame boundaries
- [ ] Adaptive confidence threshold based on bounding box size
- [ ] Multi-camera overlap to eliminate blind entry zones
- [ ] Post-processing filter to suppress overlapping secondary bounding boxes
- [ ] Fine-tuning on Indian road-specific animal datasets
