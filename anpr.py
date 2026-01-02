import argparse
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

# Optional EasyOCR (recommended). Install with: pip install easyocr
try:
    import easyocr
except Exception:  # pragma: no cover
    easyocr = None

# Optional: download YOLO weights from Hugging Face Hub when a `hf://` model spec is used.
# This stays optional so local .pt paths continue to work without extra deps.
try:
    from huggingface_hub import hf_hub_download  # type: ignore
except Exception:  # pragma: no cover
    hf_hub_download = None  # type: ignore

# Optional YOLOv8/YOLO11 plate detector. Lazy-loaded so the script still works
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


def _resolve_model_path(spec: str) -> str:
    """Resolve a model spec into a local file path.

    Supported:
      - Local filesystem path to .pt
      - Hugging Face Hub spec: hf://<owner>/<repo>/<filename>
        e.g. hf://morsetechlab/yolov11-license-plate-detection/license-plate-finetune-v1s.pt

    Notes:
      - Some HF repos may require accepting terms or authentication.
        If so, set env var HF_TOKEN with a token and/or accept the model terms in browser.
    """
    if spec.startswith("hf://"):
        if hf_hub_download is None:
            raise SystemExit(
                "Hugging Face model spec used (hf://...) but 'huggingface_hub' is not installed. "
                "Install it with: pip install huggingface_hub"
            )
        rest = spec[len("hf://") :]
        parts = rest.split("/", 2)
        if len(parts) < 3:
            raise SystemExit("Invalid hf:// spec. Use: hf://<owner>/<repo>/<filename>.pt")
        repo_id = parts[0] + "/" + parts[1]
        filename = parts[2]
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        return hf_hub_download(repo_id=repo_id, filename=filename, token=token)
    return spec


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
        resolved = _resolve_model_path(_YOLO_MODEL_PATH)
        _YOLO_PLATE_MODEL = YOLO(resolved)
    return _YOLO_PLATE_MODEL


# -----------------
# EasyOCR helpers
# -----------------
_EASYOCR_READER = None


def _get_easyocr_reader(gpu: bool = False):
    global _EASYOCR_READER
    if easyocr is None:
        raise SystemExit("EasyOCR requested but not installed. Install with: pip install easyocr")
    if _EASYOCR_READER is None:
        _EASYOCR_READER = easyocr.Reader(["en"], gpu=gpu)
    return _EASYOCR_READER


def _order_quad(pts: List[Tuple[float, float]]) -> np.ndarray:
    arr = np.array(pts, dtype=np.float32)
    s = arr.sum(axis=1)
    diff = np.diff(arr, axis=1).reshape(-1)
    tl = arr[np.argmin(s)]
    br = arr[np.argmax(s)]
    tr = arr[np.argmin(diff)]
    bl = arr[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _warp_from_quad(base_bgr: np.ndarray, quad_pts: List[Tuple[float, float]], out_w: int = 360, out_h: int = 96) -> np.ndarray:
    quad = _order_quad(quad_pts)
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(base_bgr, M, (out_w, out_h), flags=cv2.INTER_CUBIC)


def _preprocess_variants(roi_bgr: np.ndarray) -> List[np.ndarray]:
    """Return a few variants; different OCR models like different preprocessing."""
    variants: List[np.ndarray] = []

    # Variant 1: original
    variants.append(roi_bgr)

    # Variant 2: CLAHE + upscale + mild sharpen
    g = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(g)
    g = cv2.resize(g, None, fx=4.0, fy=4.0, interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(g, (0, 0), 1.2)
    g = cv2.addWeighted(g, 1.8, blur, -0.8, 0)
    variants.append(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR))

    # Variant 3: adaptive threshold (good for some high-contrast plates)
    g2 = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    g2 = cv2.resize(g2, None, fx=4.0, fy=4.0, interpolation=cv2.INTER_CUBIC)
    g2 = cv2.GaussianBlur(g2, (3, 3), 0)
    th = cv2.adaptiveThreshold(g2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)
    variants.append(cv2.cvtColor(th, cv2.COLOR_GRAY2BGR))

    return variants


# --- Indian plate regex (covers common formats + BH-series) ---
PLATE_PATTERNS = [
    re.compile(r"^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{3,4}$"),
    re.compile(r"^\d{2}BH\d{4}[A-Z]{1,2}$"),
]


def normalize_text(s: str) -> str:
    s = s.upper()
    s = re.sub(r"[^A-Z0-9]", "", s)
    return s


def looks_like_plate(s: str) -> bool:
    if len(s) < 8 or len(s) > 12:
        return False
    return any(p.match(s) for p in PLATE_PATTERNS)


# --- Plate canonicalization and fuzzy merge helpers ---
_CANON_MAP = str.maketrans({
    "O": "0",
    "I": "1",
    "Z": "2",
    "S": "5",
    "B": "8",
    "G": "6",
})


def canonicalize_plate(s: str) -> str:
    s = normalize_text(s)
    if looks_like_plate(s):
        return s
    cand = s.translate(_CANON_MAP)
    if looks_like_plate(cand):
        return cand
    return s


def _edit_distance_leq1(a: str, b: str) -> bool:
    """Return True if Levenshtein distance(a,b) <= 1 (fast special-case)."""
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    # substitution
    if la == lb:
        diff = 0
        for i in range(la):
            if a[i] != b[i]:
                diff += 1
                if diff > 1:
                    return False
        return True
    # insertion/deletion
    if la > lb:
        a, b = b, a
        la, lb = lb, la
    i = j = 0
    edits = 0
    while i < la and j < lb:
        if a[i] == b[j]:
            i += 1
            j += 1
        else:
            edits += 1
            if edits > 1:
                return False
            j += 1
    return True


def merge_key(existing_keys: List[str], new_key: str, hits: Counter) -> str:
    """Merge near-duplicates (1-edit away) into the strongest existing key."""
    for k in existing_keys:
        if len(k) == len(new_key) and _edit_distance_leq1(k, new_key):
            # merge into the one with more hits
            return k if hits[k] >= hits[new_key] else new_key
    return new_key


@dataclass
class Candidate:
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    score: float


def detect_plate_candidates(frame_bgr: np.ndarray) -> List[Candidate]:
    """Detect likely plate regions.

    If `--yolo-model` is provided, uses Ultralytics YOLO (v8/v11) detections.
    Otherwise falls back to a lightweight contour-based heuristic.
    """

    # --- YOLO path (preferred) ---
    model = _get_yolo_model()
    if model is not None:
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
                if ww < 35 or hh < 14:
                    continue
                conf = float(box.conf[0]) if getattr(box, "conf", None) is not None else 1.0
                cands.append(Candidate((x1, y1, ww, hh), conf))
        cands.sort(key=lambda c: c.score, reverse=True)
        return cands[:5]

    # --- Fallback: contour heuristic ---
    h, w = frame_bgr.shape[:2]
    target_w = 960
    scale = target_w / w if w > target_w else 1.0
    if scale != 1.0:
        frame_bgr = cv2.resize(
            frame_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
        )
    H, W = frame_bgr.shape[:2]

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    grad_x = cv2.convertScaleAbs(grad_x)

    _, bw = cv2.threshold(grad_x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
    morph = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=2)

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
        if aspect < 1.4 or aspect > 9.0:
            continue
        score = (area / (W * H)) * min(aspect, 9.0)
        cands2.append(Candidate((x, y, ww, hh), score))

    cands2.sort(key=lambda c: c.score, reverse=True)
    cands2 = cands2[:5]

    if scale != 1.0:
        inv = 1.0 / scale
        scaled = []
        for c in cands2:
            x, y, ww, hh = c.bbox
            scaled.append(
                Candidate((int(x * inv), int(y * inv), int(ww * inv), int(hh * inv)), c.score)
            )
        cands2 = scaled

    return cands2



def ocr_plate_easyocr(roi_bgr: np.ndarray, reader, min_conf: float = 0.50) -> Tuple[str, float]:
    """Run EasyOCR on a crop and return (best_text, best_conf).

    Strategy:
      - Try multiple preprocess variants
      - Prefer regex-valid plates
      - If EasyOCR returns a tilted quadrilateral bbox, warp/rectify and re-run OCR
    """
    allow = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    best_text = ""
    best_conf = 0.0

    def _score(text: str, conf: float) -> float:
        t = normalize_text(text)
        s = conf
        if looks_like_plate(t):
            s += 1.0
        if 8 <= len(t) <= 12:
            s += 0.2
        return s

    for var in _preprocess_variants(roi_bgr):
        res = reader.readtext(var, allowlist=allow, detail=1)
        for (bbox, text, conf) in res:
            if conf < 0.01:
                continue
            t = normalize_text(text)
            sc = _score(t, float(conf))
            if sc > _score(best_text, best_conf):
                best_text, best_conf = t, float(conf)

            # If bbox is a quadrilateral (tilted), try perspective-rectify and OCR again
            try:
                if bbox and len(bbox) == 4:
                    warped = _warp_from_quad(var, bbox, out_w=360, out_h=96)
                    res2 = reader.readtext(warped, allowlist=allow, detail=1)
                    for (_b2, text2, conf2) in res2:
                        t2 = normalize_text(text2)
                        sc2 = _score(t2, float(conf2))
                        if sc2 > _score(best_text, best_conf):
                            best_text, best_conf = t2, float(conf2)
            except Exception:
                pass

    if best_conf < min_conf:
        return "", best_conf
    return best_text, best_conf


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
    ap.add_argument("video", help="Path to input video file")
    ap.add_argument("--sample-fps", type=float, default=2.0, help="How many frames per second to sample (default: 2)")
    ap.add_argument("--max-minutes", type=float, default=0.0, help="Stop after N minutes (0 = full video)")
    ap.add_argument("--blur-output", default="", help="If set, writes a video with detected plate regions blurred")

    ap.add_argument(
        "--yolo-model",
        default="",
        help=(
            "Optional: local .pt path or Hugging Face spec (hf://...). If provided, uses YOLO detections. "
            "Example: models/license_plate.pt or hf://morsetechlab/yolov11-license-plate-detection/license-plate-finetune-v1s.pt"
        ),
    )

    ap.add_argument("--debug-dir", default="", help="Save candidate crops + OCR text for inspection")
    ap.add_argument(
        "--save-annotated",
        default="",
        help="If set, saves sampled frames annotated with detection boxes to this folder",
    )

    ap.add_argument(
        "--export-plates",
        action="store_true",
        help="If set, outputs recognized plates (use only with permission/authorized use)",
    )
    ap.add_argument("--csv", default="plates.csv", help="CSV output path (only used with --export-plates)")

    ap.add_argument(
        "--ocr",
        choices=["easyocr"],
        default="easyocr",
        help="OCR engine (currently: easyocr)",
    )
    ap.add_argument(
        "--easyocr-gpu",
        action="store_true",
        help="Use GPU for EasyOCR if available",
    )
    ap.add_argument(
        "--min-ocr-conf",
        type=float,
        default=0.50,
        help="Minimum EasyOCR confidence to accept a read (0-1). Default 0.50",
    )
    ap.add_argument(
        "--strengthen",
        action="store_true",
        help="Strengthen across frames by canonicalizing and merging near-duplicate reads",
    )

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

    args = ap.parse_args()

    if args.yolo_model:
        set_yolo_model_path(args.yolo_model)

    if args.save_annotated:
        os.makedirs(args.save_annotated, exist_ok=True)

    reader = _get_easyocr_reader(gpu=args.easyocr_gpu)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    sample_every = max(1, int(round(fps / max(args.sample_fps, 0.1))))

    writer = None
    if args.blur_output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.blur_output, fourcc, fps, (w, h))

    plate_hits = Counter()
    plate_times = defaultdict(list)

    max_frames = total_frames
    if args.max_minutes and args.max_minutes > 0:
        max_frames = (
            min(max_frames, int(args.max_minutes * 60 * fps))
            if total_frames
            else int(args.max_minutes * 60 * fps)
        )

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

                text, ocr_conf = ocr_plate_easyocr(roi, reader, min_conf=args.min_ocr_conf)

                if args.debug_dir:
                    with open(os.path.join(args.debug_dir, "ocr_log.txt"), "a", encoding="utf-8") as f:
                        f.write(f"{t_sec:.2f}s\t{ocr_conf:.3f}\t{text}\n")

                if not text:
                    continue

                key = canonicalize_plate(text) if args.strengthen else text

                # Merge near-duplicates (e.g., O/0, S/5, one-off errors) into a stable key
                if args.strengthen and plate_hits:
                    key = merge_key(list(plate_hits.keys()), key, plate_hits)

                if looks_like_plate(key) or (args.no_regex and len(key) >= args.min_ocr_len):
                    plate_hits[key] += 1
                    plate_times[key].append(t_sec)
                    found_this_frame.append((key, cand.bbox))

            if annotated is not None:
                cv2.imwrite(os.path.join(args.save_annotated, f"frame_{frame_idx:06d}.jpg"), annotated)

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
        preview = ", ".join(f"{t:.1f}s" for t in times[:5])
        print(f"  {plate}: {cnt} hits (e.g. {preview})")

    if args.export_plates:
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
