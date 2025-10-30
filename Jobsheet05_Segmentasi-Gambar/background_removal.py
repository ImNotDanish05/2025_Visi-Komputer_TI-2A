# MediaPipe Image Segmenter demos (Python)
# -----------------------------------------------------
# Background removal (alpha matte)
# Automatically uses OBS virtual camera (cv2.VideoCapture(2))
# Press 'q' to quit
# -----------------------------------------------------

import os, cv2, numpy as np, requests
import mediapipe as mp
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

def composite_foreground(img_bgr, mask_u8):
    """Return binary mask where non-background = 255."""
    return (mask_u8 > 0).astype(np.uint8) * 255

def main():
    cap = cv2.VideoCapture(2)   # <-- OBS virtual camera
    if not cap.isOpened():
        print("[ERROR] Cannot open camera index 2 (OBS).")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    print("[INFO] Running Background Removal from OBS camera...")

    black = None
    with build_segmenter(vision.RunningMode.VIDEO) as seg:
        ts = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if black is None:
                black = np.zeros_like(frame)

            mp_img = to_mp_image_bgr(frame)
            res = seg.segment_for_video(mp_img, int(ts))
            mask = res.category_mask.numpy_view()

            fg_mask = composite_foreground(frame, mask)
            fg_mask_3 = cv2.merge([fg_mask, fg_mask, fg_mask])
            out = np.where(fg_mask_3 > 0, frame, black)

            cv2.imshow("Background Removal (OBS)", out)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            ts += int(1000 / fps)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
