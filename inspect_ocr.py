import sys
import cv2
import numpy as np
import math

# Optional EasyOCR (install with: pip install easyocr)
try:
    import easyocr
except Exception:
    easyocr = None


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

# CLAHE variant (often helps low-light / low-contrast plates)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_gray = clahe.apply(gray)
clahe_bgr = cv2.cvtColor(clahe_gray, cv2.COLOR_GRAY2BGR)
cv2.imwrite("ocr_debug_clahe.png", clahe_bgr)
print("Wrote ocr_debug_clahe.png")

# LAB-CLAHE variant (often preserves character stroke structure better than grayscale CLAHE)
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
L, A, B = cv2.split(lab)
clahe_L = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(L)
lab2 = cv2.merge([clahe_L, A, B])
lab_clahe_bgr = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
cv2.imwrite("ocr_debug_lab_clahe.png", lab_clahe_bgr)
print("Wrote ocr_debug_lab_clahe.png")

# Sharpened + upscaled color variant (helps preserve diagonals like 'M' under mild blur)
up = cv2.resize(img, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
blur = cv2.GaussianBlur(up, (0, 0), 1.2)
sharp_up = cv2.addWeighted(up, 1.8, blur, -0.8, 0)
cv2.imwrite("ocr_debug_sharp_up.png", sharp_up)
print("Wrote ocr_debug_sharp_up.png")

def _pad_border(bgr, pad=24):
    return cv2.copyMakeBorder(bgr, pad, pad, pad, pad, cv2.BORDER_REPLICATE)

# Padded + 6x upscale + gentle sharpen (often helps preserve diagonals like 'M' and 'W')
pad6 = _pad_border(img, pad=24)
up6 = cv2.resize(pad6, None, fx=6.0, fy=6.0, interpolation=cv2.INTER_CUBIC)
blur6 = cv2.GaussianBlur(up6, (0, 0), 1.0)
sharp6 = cv2.addWeighted(up6, 1.4, blur6, -0.4, 0)
cv2.imwrite("ocr_debug_pad6_sharp6.png", sharp6)
print("Wrote ocr_debug_pad6_sharp6.png")

# -----------------
# EasyOCR debug
# -----------------
if easyocr is None:
    print("EASYOCR: not installed. Install with: pip install easyocr")
else:
    # Reader init is heavy; for single-file debug it's fine.
    reader = easyocr.Reader(["en"], gpu=False)
    # Run on the original image and on the thresholded image
    allow = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    res_img = reader.readtext(img, allowlist=allow)
    res_th = reader.readtext(th, allowlist=allow)
    res_clahe = reader.readtext(clahe_bgr, allowlist=allow)
    res_lab_clahe = reader.readtext(lab_clahe_bgr, allowlist=allow)
    res_sharp_up = reader.readtext(sharp_up, allowlist=allow)

    res_pad6 = reader.readtext(sharp6, allowlist=allow)

    # Beamsearch + magnification + contrast tuning (helps with ambiguous glyphs)
    res_pad6_beam = reader.readtext(
        sharp6,
        allowlist=allow,
        detail=1,
        decoder="beamsearch",
        beamWidth=5,
        mag_ratio=2.0,
        contrast_ths=0.1,
        adjust_contrast=0.7,
        text_threshold=0.6,
        low_text=0.3,
    )

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
    _render_easy(res_clahe, clahe_bgr, "ocr_debug_easyocr_clahe.png")
    _render_easy(res_lab_clahe, lab_clahe_bgr, "ocr_debug_easyocr_lab_clahe.png")
    _render_easy(res_sharp_up, sharp_up, "ocr_debug_easyocr_sharp_up.png")
    _render_easy(res_pad6, sharp6, "ocr_debug_easyocr_pad6_sharp6.png")
    _render_easy(res_pad6_beam, sharp6, "ocr_debug_easyocr_pad6_sharp6_beam.png")
