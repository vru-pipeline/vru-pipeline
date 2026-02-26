"""
annotate_gt.py
==============
Tkinter GUI for assigning ground-truth labels to pipeline tracks.

Reads  : tracks_pipeline.json   (written by run_pipeline.py — never modified)
Writes : tracks_gt.json         (your annotations — updated after every track)

Resume: already-annotated tracks (labelled or skipped) are skipped automatically.
        Just re-run the script to continue where you left off.

Performance: on startup, all keyframe crops for pending tracks are extracted in a
             single forward pass through the video (no random seeking).  The GUI
             then reads only from the in-memory cache — display is instant regardless
             of where in the video a track appears.
"""

import json
import tkinter as tk
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image, ImageTk


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  — edit these paths before running
# ─────────────────────────────────────────────────────────────────────────────

PIPELINE_JSON = r"tracks_pipeline.json"
GT_JSON       = r"tracks_gt.json"
VIDEO_PATH    = r"video.dav"


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

CLASSES = ["pedestrian", "cyclist", "motorcyclist", "pmd"]
COLORS  = {"pedestrian": "#4CAF50", "cyclist": "#2196F3",
           "motorcyclist": "#FF9800", "pmd": "#9C27B0"}

GRID_COLS    = 4    # keyframes per row
THUMB_W      = 160
THUMB_H      = 180
PAD          = 4    # pixels between thumbnails

# Padding applied when extracting keyframe crops for display
CROP_PAD = 0.25


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_pipeline(path: str) -> dict:
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def load_gt(path: str) -> dict:
    p = Path(path)
    if p.exists():
        with open(p, 'r') as f:
            return json.load(f)
    return {}


def save_gt(path: str, gt: dict):
    with open(path, 'w') as f:
        json.dump(gt, f, indent=2)


def _crop_frame(frame: np.ndarray, x1: int, y1: int,
                x2: int, y2: int) -> np.ndarray | None:
    """Extract a padded crop from a single decoded frame and return as RGB."""
    h, w = frame.shape[:2]
    bw   = x2 - x1
    bh   = y2 - y1
    px   = int(bw * CROP_PAD)
    py   = int(bh * CROP_PAD)
    cx1  = max(0, int(x1) - px)
    cy1  = max(0, int(y1) - py)
    cx2  = min(w, int(x2) + px)
    cy2  = min(h, int(y2) + py)
    crop = frame[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return None
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def preextract_crops(video_path: str, all_tracks: dict,
                     pending: list) -> dict[str, list]:
    """
    Single forward pass through the video.

    Builds a mapping  { frame_number → [(tid, kf_index, bbox), ...] }
    then reads the video sequentially, extracting crops whenever the current
    frame is needed.  No random seeking — the video is read exactly once.

    Returns
    -------
    cache : dict[str, list]
        { tid → [crop_0, crop_1, …] }  (RGB numpy arrays, in keyframe order)
        Only pending tracks are included.
    """

    # ── 1. Collect every (frame_no, bbox) needed, keyed by frame number ──────
    # needed[frame_no] = list of (tid, kf_index, (x1,y1,x2,y2))
    needed: dict[int, list] = defaultdict(list)
    for tid in pending:
        kfs = all_tracks[tid].get("keyframes", [])
        for kf_idx, kf in enumerate(kfs):
            needed[kf["frame"]].append((tid, kf_idx, tuple(kf["bbox"])))

    if not needed:
        return {}

    first_needed = min(needed)
    last_needed  = max(needed)

    # ── 2. Prepare output cache ───────────────────────────────────────────────
    # Pre-allocate None slots so we can insert by index order later
    kf_counts = {tid: len(all_tracks[tid].get("keyframes", [])) for tid in pending}
    cache: dict[str, list] = {tid: [None] * kf_counts[tid] for tid in pending}

    # ── 3. Forward pass ───────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"[Annotate] Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    print(f"[Annotate] Pre-extracting crops for {len(pending)} pending tracks …")
    print(f"[Annotate] Scanning frames {first_needed} – {last_needed}"
          f"  (total video frames: {total_frames})")

    # Seek to the first needed frame to avoid decoding the entire leading portion
    if first_needed > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, first_needed)

    frame_idx   = first_needed
    crops_found = 0
    log_every   = max(1, (last_needed - first_needed + 1) // 10)  # ~10 progress prints

    while frame_idx <= last_needed:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in needed:
            for tid, kf_idx, (x1, y1, x2, y2) in needed[frame_idx]:
                crop = _crop_frame(frame, x1, y1, x2, y2)
                if crop is not None:
                    cache[tid][kf_idx] = crop
                    crops_found += 1

        if (frame_idx - first_needed) % log_every == 0:
            pct = (frame_idx - first_needed) / max(1, last_needed - first_needed) * 100
            print(f"  … {pct:5.1f}%  (frame {frame_idx})")

        frame_idx += 1

    cap.release()

    # Drop None slots (failed seeks / empty crops) and convert to plain lists
    for tid in cache:
        cache[tid] = [c for c in cache[tid] if c is not None]

    print(f"[Annotate] Pre-extraction done — {crops_found} crops cached "
          f"across {len(pending)} tracks.")
    return cache


def make_grid_image(crops: list, cols: int,
                    thumb_w: int, thumb_h: int, pad: int) -> Image.Image:
    """Arrange crops in a grid and return a PIL image."""
    n    = len(crops)
    rows = max(1, (n + cols - 1) // cols)
    W    = cols * thumb_w + (cols + 1) * pad
    H    = rows * thumb_h + (rows + 1) * pad
    grid = Image.new("RGB", (W, H), color=(40, 40, 40))

    for i, crop in enumerate(crops):
        img = Image.fromarray(crop).resize((thumb_w, thumb_h), Image.LANCZOS)
        col = i % cols
        row = i // cols
        x   = pad + col * (thumb_w + pad)
        y   = pad + row * (thumb_h + pad)
        grid.paste(img, (x, y))

    return grid


# ─────────────────────────────────────────────────────────────────────────────
# GUI
# ─────────────────────────────────────────────────────────────────────────────

class AnnotationGUI:
    def __init__(self):
        # ── Load data ────────────────────────────────────────────────────────
        print("[Annotate] Loading pipeline data …")
        pipeline_data = load_pipeline(PIPELINE_JSON)
        self.all_tracks  = pipeline_data["tracks"]   # {str(tid): {...}}
        self.meta        = pipeline_data.get("meta", {})
        self.video_path  = VIDEO_PATH

        self.gt = load_gt(GT_JSON)

        # Tracks still needing annotation
        self.pending = [
            tid for tid in self.all_tracks
            if tid not in self.gt
        ]
        self.pending.sort(key=lambda t: self.all_tracks[t]["frame_start"])

        total = len(self.all_tracks)
        done  = total - len(self.pending)
        print(f"[Annotate] Tracks total   : {total}")
        print(f"[Annotate] Already done   : {done}")
        print(f"[Annotate] Remaining      : {len(self.pending)}")

        # ── Pre-extract all crops in one forward pass ─────────────────────────
        if self.pending:
            self.crop_cache = preextract_crops(
                self.video_path, self.all_tracks, self.pending
            )
        else:
            self.crop_cache = {}

        self.idx    = 0      # index into self.pending
        self._photo = None   # keep PhotoImage reference alive

        # ── Window ───────────────────────────────────────────────────────────
        self.root = tk.Tk()
        self.root.title("VRU Ground-Truth Annotation")
        self.root.configure(bg="#1e1e1e")
        self.root.resizable(True, True)

        self._build_ui()
        self.root.update()

        if self.pending:
            self._show_track()
        else:
            self._show_done()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        # Info bar
        info_frame = tk.Frame(self.root, bg="#1e1e1e")
        info_frame.pack(fill="x", padx=10, pady=(10, 4))
        self.info_var = tk.StringVar()
        tk.Label(info_frame, textvariable=self.info_var,
                 font=("Consolas", 11), fg="#e0e0e0", bg="#1e1e1e").pack()

        # Pipeline prediction label
        self.pred_var = tk.StringVar()
        tk.Label(info_frame, textvariable=self.pred_var,
                 font=("Consolas", 10), fg="#aaaaaa", bg="#1e1e1e").pack()

        # Progress bar (text)
        self.prog_var = tk.StringVar()
        tk.Label(info_frame, textvariable=self.prog_var,
                 font=("Consolas", 10), fg="#888888", bg="#1e1e1e").pack()

        # Canvas for keyframe grid
        cols     = GRID_COLS
        canvas_w = cols * THUMB_W + (cols + 1) * PAD
        canvas_h = 3 * THUMB_H + 4 * PAD          # up to 3 rows
        self.canvas = tk.Canvas(self.root, width=canvas_w, height=canvas_h,
                                bg="#2a2a2a", highlightthickness=0)
        self.canvas.pack(padx=10, pady=8)

        # Class buttons
        btn_frame = tk.Frame(self.root, bg="#1e1e1e")
        btn_frame.pack(fill="x", padx=10, pady=6)

        tk.Label(btn_frame, text="Assign class:",
                 font=("Arial", 11, "bold"), fg="#e0e0e0",
                 bg="#1e1e1e").pack(side="left", padx=(0, 10))

        for cls in CLASSES:
            color = COLORS[cls]
            tk.Button(
                btn_frame, text=cls.upper(),
                font=("Arial", 11, "bold"),
                bg=color, fg="white",
                activebackground=color, activeforeground="white",
                width=14, height=2, relief="flat",
                command=lambda c=cls: self._assign(c),
            ).pack(side="left", padx=4)

        tk.Button(
            btn_frame, text="SKIP",
            font=("Arial", 11, "bold"),
            bg="#555555", fg="white",
            activebackground="#777777",
            width=10, height=2, relief="flat",
            command=self._skip,
        ).pack(side="left", padx=12)

        # Keyboard shortcuts
        for i, cls in enumerate(CLASSES, 1):
            self.root.bind(str(i), lambda e, c=cls: self._assign(c))
        self.root.bind("<space>", lambda e: self._skip())
        self.root.bind("<Escape>", lambda e: self.root.quit())

        # Shortcut hint
        hint = "  Shortcuts:  1=pedestrian  2=cyclist  3=motorcyclist  4=pmd  Space=skip  Esc=quit"
        tk.Label(self.root, text=hint,
                 font=("Consolas", 9), fg="#666666", bg="#1e1e1e").pack(pady=(0, 6))

    # ── Track display ─────────────────────────────────────────────────────────

    def _show_track(self):
        if self.idx >= len(self.pending):
            self._show_done()
            return

        tid   = self.pending[self.idx]
        track = self.all_tracks[tid]
        total = len(self.all_tracks)
        done  = total - len(self.pending) + self.idx

        # Info labels
        self.info_var.set(
            f"Track {tid}  |  frames {track['frame_start']}–{track['frame_end']}"
            f"  ({track['n_frames']} frames)  |  median h: {track['median_h_px']} px"
        )
        pred_cls  = track.get("pipeline_class", "?")
        pred_conf = track.get("confidence", 0.0)
        self.pred_var.set(f"Pipeline prediction: {pred_cls.upper()}  ({pred_conf:.2f})")
        self.prog_var.set(
            f"Progress: {done}/{total} annotated  |  {len(self.pending) - self.idx} remaining"
        )

        # Fetch crops from cache — instant, no video I/O
        crops = self.crop_cache.get(tid, [])

        if not crops:
            # No crops cached (very rare) — skip automatically
            print(f"[Annotate] Warning: no crops cached for track {tid}, skipping.")
            self._skip()
            return

        grid_img    = make_grid_image(crops, GRID_COLS, THUMB_W, THUMB_H, PAD)
        self._photo = ImageTk.PhotoImage(grid_img)

        self.canvas.config(width=grid_img.width, height=grid_img.height)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo)

    def _show_done(self):
        self.canvas.delete("all")
        self.canvas.create_text(
            self.canvas.winfo_width() // 2,
            self.canvas.winfo_height() // 2,
            text="All tracks annotated!",
            font=("Arial", 18, "bold"), fill="#4CAF50",
        )
        self.info_var.set("Annotation complete.")
        self.pred_var.set("")
        total    = len(self.all_tracks)
        labelled = sum(1 for v in self.gt.values() if v is not None)
        skipped  = sum(1 for v in self.gt.values() if v is None)
        self.prog_var.set(
            f"Total: {total}  |  Labelled: {labelled}  |  Skipped: {skipped}"
        )

    # ── Actions ───────────────────────────────────────────────────────────────

    def _assign(self, cls: str):
        if self.idx >= len(self.pending):
            return
        tid = self.pending[self.idx]
        self.gt[tid] = cls
        save_gt(GT_JSON, self.gt)
        print(f"[Annotate] Track {tid:>5}  →  {cls}")
        self.idx += 1
        self._show_track()

    def _skip(self):
        if self.idx >= len(self.pending):
            return
        tid = self.pending[self.idx]
        self.gt[tid] = None   # null = skipped, excluded from evaluation
        save_gt(GT_JSON, self.gt)
        print(f"[Annotate] Track {tid:>5}  →  SKIP")
        self.idx += 1
        self._show_track()

    # ── Run ───────────────────────────────────────────────────────────────────

    def run(self):
        self.root.mainloop()


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = AnnotationGUI()
    app.run()