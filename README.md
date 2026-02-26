# Fine-Grained VRU Classification Pipeline

This repository contains the code accompanying the paper:

> **A Track-Level Pipeline for Fine-Grained Vulnerable Road User
> Classification from Urban Surveillance Video**
> *Anonymous submission — double-blind review*

---

## Overview

A three-stage pipeline for classifying vulnerable road users (VRUs) from 
fixed infrastructure camera video into four classes: **pedestrian**, 
**cyclist**, **motorcyclist**, and **PMD rider**.

The pipeline combines:
- **YOLO11m** for person detection
- **BoT-SORT** for multi-object tracking
- **Swin Transformer Tiny** for per-crop classification
- **Confidence-weighted voting** for stable per-track label assignment

---

## Repository Structure
```
├── run_pipeline.py      # Main pipeline: detection → tracking → classification
├── annotate_gt.py       # Tkinter GUI for ground-truth annotation
├── evaluate_multi.py    # Evaluation script: accuracy, F1, confusion matrix
├── requirements.txt     # Python dependencies
```

---

## Pretrained Weights

The Swin Transformer Tiny classifier was trained from scratch on a 
purpose-built four-class VRU dataset of 82,200 bounding box crops 
collected from urban surveillance footage.

Pretrained weights (~340 MB) are available on HuggingFace:  
https://huggingface.co/vru-research/vru-classifier  (Anonymous link)

Download `swin.pth` and place it in the same directory as the scripts.

YOLO11m weights are downloaded automatically by Ultralytics on first run.

---

## Installation
```bash
pip install -r requirements.txt
```

Python 3.10+ recommended. A CUDA-capable GPU is strongly advised for 
the pipeline; annotation and evaluation run on CPU.

---

## Usage

### 1. Run the pipeline

Edit the configuration block at the top of `run_pipeline.py` to set 
your video path, then:
```bash
python run_pipeline.py
```

Output: `tracks_pipeline.json`

This must be run **once only** per video. Track IDs are stable within 
a single run and are referenced by the annotation and evaluation tools.

### 2. Annotate ground truth
```bash
python annotate_gt.py
```

A Tkinter GUI displays keyframe crops for each track. Assign one of 
the four VRU classes using buttons or keyboard shortcuts (1–4), or 
press Space to skip ambiguous tracks. Sessions are resumable — 
already-annotated tracks are skipped automatically on restart.

Output: `tracks_gt.json`

### 3. Evaluate

Edit `VIDEO_PAIRS` in `evaluate.py` to list your 
`(tracks_pipeline.json, tracks_gt.json)` pairs, then:
```bash
python evaluate.py
```

Outputs per-video and global accuracy, per-class precision/recall/F1, 
and a normalized confusion matrix saved as `confusion_matrix.eps`.

---

## Evaluation Protocol

The pipeline is run **once** on each evaluation video. Tracks are 
annotated by a human using the provided annotation tool. Accuracy is 
measured by direct track identifier lookup — no spatial IoU matching 
is required. This design isolates classifier performance from detection 
and tracking performance.

---

## License

MIT License
