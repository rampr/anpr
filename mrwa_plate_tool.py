import argparse
import os
import re
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np
import pytesseract
from tqdm import tqdm

# Optional YOLOv8 plate detector (recommended). This is lazy-loaded so the script still works
# without ultralytics installed when you use the contour-based detector.
try:
    from ultralytics import YOLO  # type: ignore
except Exception:  # pragma: no cover
    YOLO = None  # type: ignore

_YOLO_PLATE_MODEL = None
_YOLO_MODEL_PATH = ""

def set_yolo_model_path(path: str) -> None:
    global _YOLO_MODEL_PATH, _YOLO_PLATE_MODEL
    _YOLO_MODEL_PATH = path
    _YOLO_PLATE_MODEL = None  # reset so it reloads

def _get_yolo_model():
    global _YOLO_PLATE_MODEL
    if not _YOLO_MODEL_PATH:
        return None
    if YOLO is None:
        raise SystemExit(
            "YOLO model requested but 'ultralytics' is not installed. "
            "Install it with: pip install ultralytics"
        )
    if _YOLO_PLATE_MODEL is None:
        # This must be a local .pt path or a URL that does NOT require authentication.
        _YOLO_PLATE_MODEL = YOLO(_YOLO_MODEL_PATH)
    return _YOLO_PLATE_MODEL


# --- Indian plate regex (covers common formats + BH-series) ---
PLATE_PATTERNS = [
    # Common: KA01AB1234, DL3CAB1234, TN10A1234, MH12DE1433 etc.
    re.compile(r"^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{3,4}$"),
    # BH series: 21BH2345AA or 21 BH 2345 AA (we normalize spaces out)
    re.compile(r"^\d{2}BH\d{4}[A-Z]{1,2}$"),
]

def normalize_text(s: str) -> str:
    s = s.upper()
    s = re.sub(r"[^A-Z0-9]", "", s)  # keep only alnum
    # common OCR confusions (optional): uncomment if helpful
    # s = s.replace("O", "0")  # can hurt sometimes; use carefully
    # s = s.replace("I", "1")
    return s

def looks_like_plate(s: str) -> bool:
    if len(s) < 8 or len(s) > 12:
        return False
    return any(p.match(s) for p in PLATE_PATTERNS)

@dataclass
class Candidate:
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    score: float

def detect_plate_candidates(frame_bgr: np.ndarray) -> List[Candidate]:
    """Detect likely plate regions.

    If `--yolo-model` is provided, uses YOLOv8 detections.
    Otherwise falls back to a lightweight contour-based heuristic.
    """

    # --- YOLO path (preferred) ---
    model = _get_yolo_model()
    if model is not None:
        # Ultralytics expects BGR numpy array just fine.
        results = model(frame_bgr, conf=0.35, iou=0.5, verbose=False)
        cands: List[Candidate] = []
        for r in results:
            if getattr(r, "boxes", None) is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                ww = x2 - x1
                hh = y2 - y1
                if ww <= 0 or hh <= 0:
                    continue
                # small-object guards for 640x360-ish CCTV
                if ww < 35 or hh < 14:
                    continue
                conf = float(box.conf[0]) if getattr(box, "conf", None) is not None else 1.0
                cands.append(Candidate((x1, y1, ww, hh), conf))
        # Keep top few detections
        cands.sort(key=lambda c: c.score, reverse=True)
        return cands[:5]

    # --- Fallback: contour heuristic ---
    h, w = frame_bgr.shape[:2]

    # Resize to speed up while preserving enough detail
    target_w = 960
    scale = target_w / w if w > target_w else 1.0
    if scale != 1.0:
        frame_bgr = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    H, W = frame_bgr.shape[:2]

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    # Edge emphasis
    grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    grad_x = cv2.convertScaleAbs(grad_x)

    # Threshold and morphology to connect characters into a plate-like blob
    _, bw = cv2.threshold(grad_x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
    morph = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cands2: List[Candidate] = []
    for cnt in contours:
        x, y, ww, hh = cv2.boundingRect(cnt)
        area = ww * hh
        if area < (W * H) * 0.00015:
            continue
        if area > (W * H) * 0.2:
            continue

        aspect = ww / float(hh + 1e-6)
        # Looser than before to handle perspective / small plates
        if aspect < 1.4 or aspect > 9.0:
            continue

        # Score: prefer medium-large, high-aspect rectangles
        score = (area / (W * H)) * min(aspect, 9.0)
        cands2.append(Candidate((x, y, ww, hh), score))

    # Keep top candidates
    cands2.sort(key=lambda c: c.score, reverse=True)
    cands2 = cands2[:5]

    # Scale bbox back to original frame coordinates
    if scale != 1.0:
        inv = 1.0 / scale
        scaled = []
        for c in cands2:
            x, y, ww, hh = c.bbox
            scaled.append(Candidate((int(x * inv), int(y * inv), int(ww * inv), int(hh * inv)), c.score))
        cands2 = scaled

    return cands2

def ocr_plate(roi_bgr: np.ndarray) -> str:
    """
    OCR tuned for a single line of plate text.
    """
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Adaptive threshold helps in varying illumination
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 15)

    config = "--oem 1 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    text = pytesseract.image_to_string(th, config=config)
    return normalize_text(text)

def blur_region(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> None:
    x, y, w, h = bbox
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
    roi = frame[y0:y1, x0:x1]
    if roi.size == 0:
        return
    roi_blur = cv2.GaussianBlur(roi, (0, 0), sigmaX=12, sigmaY=12)
    frame[y0:y1, x0:x1] = roi_blur

def main():
    ap = argparse.ArgumentParser(description="MRWA car parking video plate helper (consent-based).")
    ap.add_argument("--debug-dir", default="", help="Save candidate crops + OCR text for inspection")
    ap.add_argument(
        "--yolo-model",
        default="",
        help=(
            "Optional: path to a YOLOv8 license-plate model (.pt). "
            "If provided, the script will use YOLO detections instead of contour heuristics. "
            "Example: models/yolov8-license-plate.pt"
        ),
    )
    ap.add_argument("video", help="Path to input video file")
    ap.add_argument("--sample-fps", type=float, default=2.0, help="How many frames per second to sample (default: 2)")
    ap.add_argument("--max-minutes", type=float, default=0.0, help="Stop after N minutes (0 = full video)")
    ap.add_argument("--blur-output", default="", help="If set, writes a video with detected plate regions blurred")
    ap.add_argument("--export-plates", action="store_true",
                    help="If set, outputs recognized plates (use only with permission/authorized use)")
    ap.add_argument("--csv", default="plates.csv", help="CSV output path (only used with --export-plates)")
    ap.add_argument("--tesseract-cmd", default="", help="Optional path to tesseract executable (Windows)")
    ap.add_argument(
        "--no-regex",
        action="store_true",
        help="Debug: export OCR strings even if they do not match plate regex",
    )
    ap.add_argument(
        "--min-ocr-len",
        type=int,
        default=6,
        help="Minimum OCR length to accept when --no-regex is used (default: 6)",
    )
    ap.add_argument(
        "--save-annotated",
        default="",
        help="If set, saves sampled frames annotated with detection boxes to this folder",
    )
    args = ap.parse_args()

    # If a YOLO model is provided, use it for plate detection (recommended)
    if args.yolo_model:
        set_yolo_model_path(args.yolo_model)

    if args.save_annotated:
        os.makedirs(args.save_annotated, exist_ok=True)

    if args.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = args.tesseract_cmd

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    sample_every = max(1, int(round(fps / max(args.sample_fps, 0.1))))

    # Optional output writer (blurred)
    writer = None
    if args.blur_output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.blur_output, fourcc, fps, (w, h))

    # Aggregation
    plate_hits = Counter()
    plate_times = defaultdict(list)  # plate -> list of seconds

    max_frames = total_frames
    if args.max_minutes and args.max_minutes > 0:
        max_frames = min(max_frames, int(args.max_minutes * 60 * fps)) if total_frames else int(args.max_minutes * 60 * fps)

    pbar_total = max_frames if max_frames else None
    pbar = tqdm(total=pbar_total, desc="Processing frames")

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_frames and frame_idx >= max_frames:
            break

        if frame_idx % sample_every == 0:
            t_sec = frame_idx / fps

            cands = detect_plate_candidates(frame)
            if args.debug_dir:
                print(f"[{t_sec:.1f}s] detections: {len(cands)}", flush=True)

            annotated = frame.copy() if args.save_annotated else None

            found_this_frame = []

            for cand_i, cand in enumerate(cands):
                x, y, w, h = cand.bbox
                if annotated is not None:
                    cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

                pad_x = int(0.08 * w)
                pad_y = int(0.25 * h)

                x0 = max(0, x - pad_x)
                y0 = max(0, y - pad_y)
                x1 = min(frame.shape[1], x + w + pad_x)
                y1 = min(frame.shape[0], y + h + pad_y)

                roi = frame[y0:y1, x0:x1]
                if roi.size == 0:
                    continue

                if args.debug_dir:
                    os.makedirs(args.debug_dir, exist_ok=True)
                    cv2.imwrite(
                        os.path.join(args.debug_dir, f"t{int(t_sec*10):06d}_cand{cand_i}.jpg"),
                        roi,
                    )

                text = ocr_plate(roi)

                if args.debug_dir and text:
                    with open(os.path.join(args.debug_dir, "ocr_log.txt"), "a", encoding="utf-8") as f:
                        f.write(f"{t_sec:.2f}s\t{text}\n")

                if (looks_like_plate(text)) or (args.no_regex and len(text) >= args.min_ocr_len):
                    plate_hits[text] += 1
                    plate_times[text].append(t_sec)
                    found_this_frame.append((text, cand.bbox))

            if annotated is not None:
                cv2.imwrite(
                    os.path.join(args.save_annotated, f"frame_{frame_idx:06d}.jpg"),
                    annotated,
                )

            # Blur output (privacy-safe)
            if writer is not None:
                out = frame.copy()
                for _, bb in found_this_frame:
                    blur_region(out, bb)
                writer.write(out)

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    if writer is not None:
        writer.release()

    print("\nTop plate candidates (by frequency):")
    for plate, cnt in plate_hits.most_common(20):
        times = plate_times[plate]
        # show a couple timestamps for quick validation
        preview = ", ".join(f"{t:.1f}s" for t in times[:5])
        print(f"  {plate}: {cnt} hits (e.g. {preview})")

    if args.export_plates:
        # Write minimal CSV with aggregated timestamps
        import csv
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["plate", "hits", "timestamps_seconds"])
            for plate, cnt in plate_hits.most_common():
                w.writerow([plate, cnt, " ".join(f"{t:.2f}" for t in plate_times[plate])])
        print(f"\nSaved CSV: {args.csv}")
    else:
        print("\n(Not exporting plate text. Use --export-plates only for authorized/consented use.)")


if __name__ == "__main__":
    main()
