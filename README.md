# Animal Detection on Roads
### Using YOLOe-26s Open-Vocabulary Object Detection

## Overview

Animal intrusions on highways and rural roads are a persistent road safety hazard. Sudden encounters with livestock, wildlife, or stray animals lead to accidents that result in vehicle damage, human injury, and animal fatalities. Manual surveillance of roads for animal intrusions is impractical at scale.

This project implements a fully automated, real-time detection and alert system using YOLOe-26s-seg, an open-vocabulary segmentation variant of the YOLO (You Only Look Once) object detection family. The system processes video feeds, tracks animals across frames, and triggers audio alerts to notify operators of road hazards.

## Objectives

* Detect the presence of one or more animals on a road from a video feed in real time.
* Track individual animals across frames using persistent track IDs.
* Trigger distinct audio alerts: a single beep when an animal is first detected, and two beeps once the road is confirmed clear.
* Produce a fully annotated output video with bounding boxes, class labels, confidence scores, track IDs, and a real-time statistics panel.
* Evaluate model performance across varied real-world scenarios involving different species, motion speeds, and partial visibility conditions.

## Key Features

### 1. Open-Vocabulary Detection

Unlike standard YOLO models trained on fixed class vocabularies such as COCO-80, YOLOe supports open-vocabulary detection at runtime. The user defines target classes as text prompts, and the model encodes them into detection embeddings via a language-vision alignment mechanism. This makes it ideal for domain-specific deployment without requiring retraining.

Animal classes used as prompts: cow, sheep, duck, goat, dog, cat, horse, deer, elephant, fox, rabbit, bird, camel, buffalo, zebra, and tiger.

### 2. Persistent Animal Tracking

Each detected animal is assigned a persistent track ID across frames using the built-in tracker. A majority-vote label stabilizer ensures that per-track class labels do not flicker across frames, producing stable and consistent identifications throughout the video.

### 3. Alert State Machine

The system manages alert logic through a two-state machine. Transitioning from CLEAR to ANIMAL DETECTED triggers a single beep. Transitioning back to CLEAR, after a 75-frame grace period with no detections, triggers two beeps. This design prevents false all-clear signals during brief inter-frame detection gaps common with fast-moving animals.

### 4. Annotated Output Video

Every processed frame is annotated with bounding boxes, class labels, confidence scores, and track IDs. A status banner at the top indicates whether animals are present or the road is clear. A statistics panel at the bottom tracks per-class unique individual counts throughout the clip.

### 5. Audio-Video Muxing

Alert sounds are synthesized as WAV files and timestamped against the video timeline. FFmpeg is used to mux the combined audio alert track into the final MP4 output.

## System Architecture

```
Video Input (MP4)
      |
      v
Frame Extraction (OpenCV)
      |
      v
YOLOe-26s-seg  <--- Text Prompts (16 animal classes)
  model.track()
      |
      v
Bounding Box + Track ID Assignment
      |
      v
Majority-Vote Label Stabilizer
      |
      v
Alert State Machine
  |-- CLEAR to ANIMAL DETECTED  -->  1 Beep
  +-- ANIMAL DETECTED to CLEAR  -->  2 Beeps (after 75-frame grace period)
      |
      v
Annotated Frame Writer (OpenCV VideoWriter)
      |
      v
Audio + Video Mux (FFmpeg)
      |
      v
Output MP4 with Embedded Alert Track
```

## Tech Stack

**Model**

* YOLOe-26s-seg (Ultralytics)

**Computer Vision**

* OpenCV

**Audio Processing**

* NumPy, Python wave module

**Video Processing**

* FFmpeg

**Environment**

* Google Colab, Python 3

## Installation and Usage

### Prerequisites

* Google Colab account or local Python environment
* FFmpeg installed
* libsndfile1 installed

### Steps to Run

1. Open the Colab notebook:

```
https://colab.research.google.com/drive/18XmPhchOToIMH76vB5NUqvpvJeQ9fxBo?usp=sharing
```

2. Install dependencies:

```
pip install ultralytics opencv-python
apt-get install -y ffmpeg libsndfile1
```

3. Upload your road video to the Colab session and update the input filename:

```python
VIDEO_INPUT = "your_video.mp4"
```

4. Run all cells. The system will process the video, annotate frames, synthesize alert audio, and mux the final output.

5. The completed file `animal_detection_with_audio.mp4` will be automatically downloaded.

## Results

Five video samples were used to evaluate the system across varied conditions.

| Sample | Scene | Animals Present | Detected | Alert Fired |
|--------|-------|----------------|----------|-------------|
| 1 | Herd crossing road (day) | Cows | 13 Cow + 1 Horse (misclassified) | Yes |
| 2 | Single animal at frame edge (day) | Elephant | Elephant + Dog (partial-body error) | Yes |
| 3 | Mixed group, fast motion (day) | Zebras + Deer | 20 Zebras, 0 Deer (missed) | Yes |
| 4 | Sudden appearance at night | Dog | Dog | Yes |
| 5 | Animal walking at night | Elephant | Elephant + Cow (false positive) | Yes |

### Performance Summary

| Aspect | Observed Behaviour | Root Cause | Severity |
|--------|--------------------|-----------|----------|
| Cow/Horse confusion | 7.2% misclassification rate | Embedding space proximity | Low |
| Elephant labelled as Dog | Partial-body misclassification at frame edge | Incomplete visual feature vector | High |
| Deer missed entirely | 0% recall | Small size + motion blur below threshold | Critical |
| Zebra detection | 100% class accuracy | High-contrast stripe pattern resists blur | Positive |
| Alert state machine | Correct in all 5 samples | Grace period design robust | No issue |
| Night-time dog detection | Correctly detected and alerted | Embedding robust under low light | Positive |
| Night-time elephant false positive | Phantom cow bounding box alongside correct elephant | Low-light feature compression | Low |

## Key Design Decisions

**Confidence Threshold: 0.40** — Balances sensitivity and false positive rate across diverse species and lighting conditions.

**Majority-Vote Label Stabilizer** — Each track ID accumulates a class vote histogram over frames. The most-voted class wins, preventing noisy label flickering.

**75-Frame Grace Period** — The road-clear signal is only fired after 75 consecutive frames with no detections, approximately 3 seconds at 25 fps. This absorbs brief detection gaps without introducing excessive latency.

## Known Limitations

* Partial visibility at frame edges causes misclassification, which is the highest-risk failure zone since animals entering the road are most dangerous at the moment of entry.
* Small animals in fast motion fall below the confidence threshold, resulting in missed detections.
* Inter-class confusion is highest between morphologically similar species that share body structure and size.
* Low-light conditions can cause secondary false-positive bounding boxes on large animals.

## Future Enhancements

* Edge-zone confidence relaxation for animals entering from frame boundaries.
* Adaptive confidence threshold based on detected bounding box size.
* Multi-camera overlap to eliminate blind entry zones.
* Post-processing filter to suppress overlapping secondary bounding boxes.
* Fine-tuning on Indian road-specific animal datasets.
* Integration with physical alert systems for on-road deployment.
