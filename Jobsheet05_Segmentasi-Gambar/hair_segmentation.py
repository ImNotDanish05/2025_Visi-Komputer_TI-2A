# MediaPipe Image Segmenter demos (Python)
# -----------------------------------------------------
# This script uses MediaPipe Tasks (vision.ImageSegmenter)
# with the 'selfie_multiclass_256x256' model which supports:
# 0=background, 1=hair, 2=body-skin, 3=face-skin, 4=clothes, 5=others.
#
# Model (auto-download on first run):
#   https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite
#
# Install dependencies:
#   pip install mediapipe opencv-python numpy requests
#
# Notes:
# - This version automatically opens VideoCapture(2) → OBS virtual camera.
# - Press 'q' to quit the window.
#
# © For educational use.

# Task 2: Hair segmentation (or any class segmentation)
# -----------------------------------------------------
# Uses the multiclass selfie model and selects CLASS_ID (default = 1).

import os, cv2, numpy as np, requests
import mediapipe as mp
from mediapipe.tasks.python import vision

MODEL_PATH = os.environ.get("MP_SEG_MODEL", "models/selfie_multiclass_256x256.tflite")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"

# === Change this ID if you want other segmentation parts ===
# 1 = Hair, 2 = Body-skin, 3 = Face-skin, 4 = Clothes, 5 = Others
CLASS_ID = 3

def ensure_model(path=MODEL_PATH, url=MODEL_URL):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        print(f"[INFO] Downloading model to {path} ...")
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
        print("[OK] Model downloaded.")
    return path

def build_segmenter(running_mode: vision.RunningMode):
    BaseOptions = mp.tasks.BaseOptions
    ImageSegmenter = mp.tasks.vision.ImageSegmenter
    ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
    options = ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=ensure_model()),
        running_mode=running_mode,
        output_category_mask=True,
        output_confidence_masks=False
    )
    return ImageSegmenter.create_from_options(options)

def to_mp_image_bgr(img_bgr):
    return mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    )

def extract_class(img_bgr, mask_u8, class_id=CLASS_ID):
    """Highlight a specific class (e.g. hair) using tint overlay."""
    class_mask = (mask_u8 == class_id).astype(np.uint8) * 255

    # Avoid error if no pixel detected
    if np.count_nonzero(class_mask) == 0:
        return img_bgr.copy(), class_mask

    tint = cv2.applyColorMap(class_mask, cv2.COLORMAP_OCEAN)
    result = img_bgr.copy()

    # Only apply blend where mask is detected
    mask_indices = class_mask > 0
    blended = cv2.addWeighted(result, 1.0, tint, 0.6, 0)
    result[mask_indices] = blended[mask_indices]
    return result, class_mask

def main():
    cap = cv2.VideoCapture(1)  # OBS virtual camera
    if not cap.isOpened():
        print("[ERROR] Cannot open camera index 2 (OBS).")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f"[INFO] Running segmentation from OBS camera (class_id={CLASS_ID})...")

    with build_segmenter(vision.RunningMode.VIDEO) as seg:
        ts = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            mp_img = to_mp_image_bgr(frame)
            res = seg.segment_for_video(mp_img, int(ts))
            mask = res.category_mask.numpy_view()

            vis, _ = extract_class(frame, mask, CLASS_ID)
            cv2.imshow(f"Segmentation (Class ID {CLASS_ID})", vis)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            ts += int(1000 / fps)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
