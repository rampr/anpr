import sys
import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import math

# Optional EasyOCR (install with: pip install easyocr)
try:
    import easyocr
except Exception:
    easyocr = None

# Optional PaddleOCR (install with: pip install paddlepaddle paddleocr)
try:
    from paddleocr import PaddleOCR, TextRecognition
except Exception:
    PaddleOCR = None
    TextRecognition = None


img_path = sys.argv[1]
img = cv2.imread(img_path)

if img is None:
    raise SystemExit(f"Could not read image: {img_path}")

# Preprocess (shared)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
th = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
)

# -----------------
# Tesseract debug
# -----------------
config = "--oem 1 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
d = pytesseract.image_to_data(th, config=config, output_type=Output.DICT)

out_tess = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
words = []
confs = []

for i in range(len(d["text"])):
    txt = d["text"][i].strip()
    conf = float(d["conf"][i]) if d["conf"][i] != "-1" else -1
    if txt and conf >= 0:
        x, y, w, h = d["left"][i], d["top"][i], d["width"][i], d["height"][i]
        cv2.rectangle(out_tess, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            out_tess,
            f"{txt} {conf:.0f}",
            (x, max(0, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        words.append(txt)
        confs.append(conf)

raw_tess = pytesseract.image_to_string(th, config=config)
mean_conf = (sum(confs) / len(confs)) if confs else 0.0
print("TESSERACT_RAW:", raw_tess.strip())
print("TESSERACT_JOINED:", "".join(words))
print(f"TESSERACT_MEAN_CONF: {mean_conf:.1f}")

cv2.imwrite("ocr_debug_tesseract.png", out_tess)
print("Wrote ocr_debug_tesseract.png")

# -----------------
# EasyOCR debug
# -----------------
if easyocr is None:
    print("EASYOCR: not installed. Install with: pip install easyocr")
else:
    # Reader init is heavy; for single-file debug it's fine.
    reader = easyocr.Reader(["en"], gpu=False)
    # Run on the original image and on the thresholded image
    res_img = reader.readtext(img)
    res_th = reader.readtext(th)

    def _order_quad(pts):
        # pts: list of 4 (x,y)
        pts = np.array(pts, dtype=np.float32)
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1).reshape(-1)
        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmin(diff)]
        bl = pts[np.argmax(diff)]
        return np.array([tl, tr, br, bl], dtype=np.float32)

    def _warp_plate(base_bgr, bbox_pts, out_w=320, out_h=80):
        # bbox_pts: 4 points from EasyOCR
        quad = _order_quad(bbox_pts)
        dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(quad, dst)
        warped = cv2.warpPerspective(base_bgr, M, (out_w, out_h), flags=cv2.INTER_CUBIC)
        return warped

    def _render_easy(results, base, out_name):
        out = base.copy()
        print(f"\nEASYOCR_RESULTS ({out_name}):")
        if not results:
            print("  (no text detected)")
        for idx, (bbox, text, conf) in enumerate(results):
            pts = [(int(p[0]), int(p[1])) for p in bbox]
            cv2.polylines(out, [np.array(pts)], isClosed=True, color=(0, 255, 255), thickness=2)
            tl = pts[0]
            cv2.putText(
                out,
                f"{text} {conf:.2f}",
                (tl[0], max(0, tl[1] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )
            print(f"  text={text!r} conf={conf:.3f}")

            # Estimate tilt angle from the top edge of the quadrilateral
            try:
                (x1, y1), (x2, y2) = bbox[0], bbox[1]
                angle = math.degrees(math.atan2((y2 - y1), (x2 - x1)))
                print(f"  TILT_DEG: {angle:.2f}")
            except Exception:
                angle = 0.0

            # Deskew/rectify only if the plate is meaningfully tilted.
            if abs(angle) >= 3.0:
                try:
                    warped = _warp_plate(base, bbox, out_w=360, out_h=96)
                    cv2.imwrite(f"easyocr_warp_{idx}.png", warped)

                    # Re-run EasyOCR on the rectified crop with an allowlist.
                    res_warp = reader.readtext(
                        warped,
                        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                        detail=1,
                    )
                    print(f"  EASYOCR_WARP_{idx}:")
                    if not res_warp:
                        print("    (no text detected on warped)")
                    for (_wb, wt, wc) in res_warp:
                        print(f"    text={wt!r} conf={wc:.3f}")
                except Exception as e:
                    print(f"  (warp failed idx={idx}: {e})")
            else:
                print("  (warp skipped: tilt < 3 deg)")

        cv2.imwrite(out_name, out)
        print(f"Wrote {out_name}")

    _render_easy(res_img, img, "ocr_debug_easyocr_img.png")
    # For thresholded image, convert to 3-channel for drawing
    th_bgr = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
    _render_easy(res_th, th_bgr, "ocr_debug_easyocr_th.png")

# -----------------
# PaddleOCR debug (PaddleOCR 3.x)
# -----------------
if PaddleOCR is None:
    print("PADDLEOCR: not installed. Install with: pip install paddlepaddle paddleocr")
else:
    # For English plates
    ocr = PaddleOCR(lang="en", use_textline_orientation=False)
    rec_only = None
    if TextRecognition is not None:
        # Recognition-only model (best for tightly-cropped plate images)
        rec_only = TextRecognition(model_name="PP-OCRv5_server_rec")

    def _run_paddle(img_in, out_name):
        pred = ocr.predict(img_in)  # <- no cls/det kwargs in 3.x
        out = img_in.copy()
        print(f"\nPADDLEOCR_RESULTS ({out_name}):")

        if not pred:
            print("  (no result objects returned from pipeline; trying recognition-only)")
            if rec_only is not None:
                out_rec = rec_only.predict(input=img_in, batch_size=1)
                if out_rec:
                    jsr = getattr(out_rec[0], "json", {})
                    text = jsr.get("res", {}).get("rec_text", "")
                    conf = jsr.get("res", {}).get("rec_score", 0.0)
                    print(f"  REC_ONLY text={text!r} conf={conf:.3f}")
                else:
                    print("  REC_ONLY (no result)")
            cv2.imwrite(out_name, out)
            print(f"Wrote {out_name}")
            return

        js = getattr(pred[0], "json", None)
        if not js:
            print("  (no .json on result; unexpected PaddleOCR return)")
            cv2.imwrite(out_name, out)
            print(f"Wrote {out_name}")
            return

        texts = js.get("rec_texts", [])
        scores = js.get("rec_scores", [])
        polys = js.get("rec_polys", [])  # list of polygons

        if not texts:
            print("  (no text detected by pipeline; trying recognition-only)")
            if rec_only is not None:
                out_rec = rec_only.predict(input=img_in, batch_size=1)
                if out_rec:
                    jsr = getattr(out_rec[0], "json", {})
                    text = jsr.get("res", {}).get("rec_text", "")
                    conf = jsr.get("res", {}).get("rec_score", 0.0)
                    print(f"  REC_ONLY text={text!r} conf={conf:.3f}")
                else:
                    print("  REC_ONLY (no result)")
            cv2.imwrite(out_name, out)
            print(f"Wrote {out_name}")
            return

        for text, conf, poly in zip(texts, scores, polys):
            # poly is an array/list of points
            pts = [(int(p[0]), int(p[1])) for p in poly]
            cv2.polylines(out, [np.array(pts)], isClosed=True, color=(255, 0, 255), thickness=2)
            tl = pts[0]
            cv2.putText(
                out,
                f"{text} {conf:.2f}",
                (tl[0], max(0, tl[1] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 255),
                2,
            )
            print(f"  text={text!r} conf={conf:.3f}")

        cv2.imwrite(out_name, out)
        print(f"Wrote {out_name}")

    _run_paddle(img, "ocr_debug_paddleocr_img.png")
    th_bgr2 = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
    _run_paddle(th_bgr2, "ocr_debug_paddleocr_th.png")
