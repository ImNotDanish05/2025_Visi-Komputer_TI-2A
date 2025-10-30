# MediaPipe Image Segmenter demos (Python)
# -----------------------------------------------------
# This script uses MediaPipe Tasks (vision.ImageSegmenter)
# with the 'selfie_multiclass_256x256' model which supports:
# 0=background, 1=hair, 2=body-skin, 3=face-skin, 4=clothes, 5=others.
#
# Model (auto-download on first run):
#   https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite
#
# Install dependencies (Python 3.9+ recommended):
#   pip install mediapipe opencv-python numpy requests
#
# Docs:
# - Image Segmenter overview & Python guide:
#   https://ai.google.dev/edge/mediapipe/solutions/vision/image_segmenter/python
#
# Tips:
# - This version automatically opens VideoCapture(2) → OBS virtual camera.
# - Press 'q' to quit a live window.
#
# © For educational use.

# Task 1: Selfie segmentation (foreground mask visualization)
# -----------------------------------------------------------
# Output: shows original and mask overlay (person vs background)

import os, cv2, numpy as np, requests, mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = os.environ.get("MP_SEG_MODEL", "models/selfie_multiclass_256x256.tflite")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"

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
    return mp.Image(image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

def draw_mask_overlay(img_bgr, mask_u8):
    foreground = (mask_u8 > 0).astype(np.uint8) * 255
    colored = cv2.applyColorMap(foreground, cv2.COLORMAP_OCEAN)
    blended = cv2.addWeighted(img_bgr, 0.4, colored, 0.6, 0)
    return blended

def main():
    cap = cv2.VideoCapture(2)  # OBS virtual camera
    if not cap.isOpened():
        print("[ERROR] Cannot open camera index 2 (OBS).")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    with build_segmenter(vision.RunningMode.VIDEO) as seg:
        ts = 0
        print("[INFO] Running selfie segmentation from OBS virtual camera...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            mp_img = to_mp_image_bgr(frame)
            result = seg.segment_for_video(mp_img, int(ts))
            mask = result.category_mask.numpy_view()
            overlay = draw_mask_overlay(frame, mask)
            cv2.imshow("Selfie Segmentation (OBS)", overlay)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            ts += int(1000 / fps)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
