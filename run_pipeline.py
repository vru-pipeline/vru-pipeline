"""
run_pipeline.py
===============
Runs the VRU classification pipeline on a single video file.

YOLO11 → BoTSORT → Swin Tiny → confidence-weighted voting

Writes one output file:
    tracks_pipeline.json  — one entry per track that passes quality filters

Run once. Never re-run on the same video (track IDs would change).
"""

import json
import time
import datetime
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import timm
from torchvision import transforms
from ultralytics import YOLO


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

VIDEO_PATH   = r"video.dav"
OUTPUT_PATH  = r"tracks_pipeline.json"
YOLO_PATH    = "yolo11m.pt"
SWIN_PATH    = "swin.pth"
DEVICE       = "cuda"   # "cpu" if no GPU


# ─────────────────────────────────────────────────────────────────────────────
# QUALITY FILTERS
# ─────────────────────────────────────────────────────────────────────────────

MIN_BBOX_HEIGHT  = 60    # pixels — median bbox height across track
MIN_TRACK_FRAMES = 20    # frames — total frames the track was seen


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE CONSTANTS  (keep in sync with annotate_gt.py / evaluate.py)
# ─────────────────────────────────────────────────────────────────────────────

CLASS_NAMES               = ["pedestrian", "cyclist", "motorcyclist", "pmd"]
YOLO_CONF                 = 0.4
YOLO_IOU                  = 0.5
CLASSIFIER_CONF_THRESHOLD = 0.4
MIN_FRAMES_FOR_LABEL      = 5
IMG_SIZE                  = 160

EXPAND_WIDTH  = 0.20
EXPAND_TOP    = 0.30
EXPAND_BOTTOM = 0.10

# Keyframe sampling stored in JSON (for annotation GUI)
KEYFRAMES_STORED = 12    # evenly spaced frames stored per track


# ─────────────────────────────────────────────────────────────────────────────
# CROP HELPER
# ─────────────────────────────────────────────────────────────────────────────

def expand_and_square_crop(frame: np.ndarray, x1: int, y1: int,
                           x2: int, y2: int) -> np.ndarray | None:
    h_f, w_f = frame.shape[:2]
    bw, bh   = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return None

    ex1 = x1 - int(bw * EXPAND_WIDTH)
    ex2 = x2 + int(bw * EXPAND_WIDTH)
    ey1 = y1 - int(bh * EXPAND_TOP)
    ey2 = y2 + int(bh * EXPAND_BOTTOM)

    sz  = max(ex2 - ex1, ey2 - ey1)
    cx  = (ex1 + ex2) // 2
    cy  = (ey1 + ey2) // 2
    sx1 = cx - sz // 2
    sy1 = cy - sz // 2
    sx2 = sx1 + sz
    sy2 = sy1 + sz

    pl = max(0, -sx1);  pt = max(0, -sy1)
    pr = max(0, sx2 - w_f); pb = max(0, sy2 - h_f)
    crop = frame[max(0, sy1):min(h_f, sy2), max(0, sx1):min(w_f, sx2)]
    if crop.size == 0:
        return None
    if pl or pt or pr or pb:
        crop = cv2.copyMakeBorder(crop, pt, pb, pl, pr,
                                  cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return cv2.resize(crop, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)


# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────

class SwinTinyClassifier:
    _transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"[Classifier] device: {self.device}")

        self.model = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=False,
            num_classes=len(CLASS_NAMES),
        )
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(ckpt, dict):
            sd = (ckpt.get("model_state_dict") or ckpt.get("state_dict")
                  or ckpt.get("model") or ckpt)
        else:
            sd = ckpt
        self.model.load_state_dict(sd, strict=True)
        self.model.to(self.device).eval()
        print(f"[Classifier] loaded: {checkpoint_path}")

    @torch.no_grad()
    def classify_batch(self, bgr_crops: list) -> list:
        tensors, valid = [], []
        for i, crop in enumerate(bgr_crops):
            if crop is not None:
                tensors.append(self._transform(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
                valid.append(i)
        results = [("unknown", 0.0)] * len(bgr_crops)
        if not tensors:
            return results
        probs = F.softmax(self.model(torch.stack(tensors).to(self.device)), dim=1)
        for j, i in enumerate(valid):
            conf, idx = probs[j].max(0)
            results[i] = (CLASS_NAMES[idx.item()], conf.item())
        return results


# ─────────────────────────────────────────────────────────────────────────────
# TRACK STORE
# ─────────────────────────────────────────────────────────────────────────────

class TrackStore:
    def __init__(self):
        self._votes:  dict[int, list] = defaultdict(list)   # (cls, conf, frame)
        self._frames: dict[int, list] = defaultdict(list)   # (frame, x1,y1,x2,y2)

    def add(self, tid: int, cls: str, conf: float,
            frame_idx: int, bbox: tuple):
        self._frames[tid].append((frame_idx, *bbox))
        if conf >= CLASSIFIER_CONF_THRESHOLD:
            self._votes[tid].append((cls, conf, frame_idx))

    def _weighted_label(self, tid: int) -> tuple:
        votes = self._votes.get(tid, [])
        if len(votes) < MIN_FRAMES_FOR_LABEL:
            return "unknown", 0.0, {}
        weighted: dict[str, float] = defaultdict(float)
        for cls, conf, _ in votes:
            weighted[cls] += conf
        total    = sum(weighted.values())
        best_cls = max(weighted, key=weighted.get)
        breakdown = {c: round(w / total, 4) for c, w in weighted.items()}
        return best_cls, round(weighted[best_cls] / total, 4), breakdown

    def _median_bbox_height(self, tid: int) -> float:
        heights = [r[4] - r[2] for r in self._frames[tid]]   # y2 - y1
        return float(np.median(heights)) if heights else 0.0

    def _sample_keyframes(self, frames: list, n: int) -> list:
        if len(frames) <= n:
            return list(range(len(frames)))
        return list(np.linspace(0, len(frames) - 1, n, dtype=int))

    def finalize(self) -> dict:
        """
        Returns only tracks that pass quality filters.
        Dict keyed by str(track_id) for JSON serialisation.
        """
        out = {}
        for tid in self._frames:
            frame_list = self._frames[tid]
            n_frames   = len(frame_list)

            if n_frames < MIN_TRACK_FRAMES:
                continue
            if self._median_bbox_height(tid) < MIN_BBOX_HEIGHT:
                continue

            cls, conf, breakdown = self._weighted_label(tid)

            # Evenly-spaced keyframe indices stored for the annotation GUI
            kf_indices = self._sample_keyframes(frame_list, KEYFRAMES_STORED)
            keyframes  = [
                {
                    "frame": frame_list[i][0],
                    "bbox":  list(frame_list[i][1:]),   # [x1,y1,x2,y2]
                }
                for i in kf_indices
            ]

            # Full bbox list (for optional spatial queries)
            all_frames = [
                {"frame": f[0], "bbox": list(f[1:])}
                for f in frame_list
            ]

            out[str(tid)] = {
                "pipeline_class":  cls,
                "confidence":      conf,
                "vote_breakdown":  breakdown,
                "n_frames":        n_frames,
                "n_vote_frames":   len(self._votes.get(tid, [])),
                "frame_start":     frame_list[0][0],
                "frame_end":       frame_list[-1][0],
                "median_h_px":     round(self._median_bbox_height(tid), 1),
                "keyframes":       keyframes,   # used by annotate_gt.py
                "all_frames":      all_frames,  # full spatial record
            }
        return out


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(video_path: str, output_path: str,
                 yolo_path: str, swin_path: str, device: str):

    out_file = Path(output_path)
    if out_file.exists():
        print(f"[Pipeline] Output already exists: {out_file}")
        print("[Pipeline] Delete it manually if you want to re-run.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[Pipeline] Video  : {video_path}")
    print(f"[Pipeline] FPS    : {fps:.2f}  |  Frames (est): {total_frames}")

    detector   = YOLO(yolo_path)
    classifier = SwinTinyClassifier(swin_path, device=device)
    store      = TrackStore()

    frame_idx = 0
    t_start   = time.time()
    log_every = max(1, int(fps * 30))   # print progress every ~30 s of video

    print("[Pipeline] Running …")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = detector.track(
            frame,
            persist=True,
            tracker="botsort.yaml",
            classes=[0],          # person class only
            conf=YOLO_CONF,
            iou=YOLO_IOU,
            verbose=False,
        )

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            tids  = results[0].boxes.id.cpu().numpy().astype(int)

            crops = [expand_and_square_crop(frame, *box) for box in boxes]
            preds = classifier.classify_batch(crops)

            for box, tid, (cls, conf) in zip(boxes, tids, preds):
                store.add(int(tid), cls, conf, frame_idx,
                          (int(box[0]), int(box[1]), int(box[2]), int(box[3])))

        frame_idx += 1
        if frame_idx % log_every == 0:
            elapsed = time.time() - t_start
            pct = frame_idx / total_frames * 100 if total_frames > 0 else 0
            print(f"  frame {frame_idx:>7}  ({pct:.1f}%)  {elapsed:.0f}s elapsed")

    cap.release()
    elapsed_total = time.time() - t_start

    tracks = store.finalize()
    print(f"\n[Pipeline] Done in {elapsed_total:.1f}s")
    print(f"[Pipeline] Tracks passing filters : {len(tracks)}")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "video":         video_path,
            "fps":           fps,
            "total_frames":  frame_idx,
            "processed_at":  datetime.datetime.now().isoformat(timespec="seconds"),
            "yolo_model":    yolo_path,
            "swin_model":    swin_path,
            "min_bbox_h_px": MIN_BBOX_HEIGHT,
            "min_frames":    MIN_TRACK_FRAMES,
        },
        "tracks": tracks,
    }
    with open(out_file, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"[Pipeline] Saved → {out_file}")


if __name__ == "__main__":
    run_pipeline(
        video_path  = VIDEO_PATH,
        output_path = OUTPUT_PATH,
        yolo_path   = YOLO_PATH,
        swin_path   = SWIN_PATH,
        device      = DEVICE,
    )
