# MediaPipe Image Segmenter demos (Python)
# -----------------------------------------------------
# Task 4: Background Replace (Zoom-like)
# Automatically:
# - Uses OBS virtual camera (cv2.VideoCapture(2))
# - Looks for background image in the same folder
# - Falls back to random image or generated text if not found
# -----------------------------------------------------

import os, cv2, random, numpy as np, requests
import mediapipe as mp
from mediapipe.tasks.python import vision

MODEL_PATH = os.environ.get("MP_SEG_MODEL", "models/selfie_multiclass_256x256.tflite")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"
VALID_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

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

def load_background(folder_path="."):
    """Find 'background' image or random one, else generate fallback."""
    candidates = [f for f in os.listdir(folder_path)
                  if f.lower().endswith(VALID_EXTS)]

    # 1️⃣ Cari file bernama background.*
    for name in candidates:
        if os.path.splitext(name)[0].lower() == "background":
            bg = cv2.imread(os.path.join(folder_path, name))
            if bg is not None:
                print(f"[INFO] Found background file: {name}")
                return bg

    # 2️⃣ Kalau gak ada, ambil gambar random di folder
    if candidates:
        random_name = random.choice(candidates)
        bg = cv2.imread(os.path.join(folder_path, random_name))
        if bg is not None:
            print(f"[INFO] Using random background: {random_name}")
            return bg

    # 3️⃣ Kalau tetap gak ada gambar, buat teks default
    print("[WARN] No background image found. Creating fallback background.")
    img = np.full((480, 640, 3), (50, 50, 50), dtype=np.uint8)
    cv2.putText(img, "ImNotDanish05 Cool", (40, 220),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(img, "Please add a file named 'background'", (40, 270),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return img

def fit_background(bg_bgr, target_shape):
    th, tw = target_shape[:2]
    return cv2.resize(bg_bgr, (tw, th), interpolation=cv2.INTER_CUBIC)

def replace_bg_frame(frame_bgr, mask_u8, bg_bgr):
    fg_mask = (mask_u8 > 0).astype(np.uint8) * 255
    fg_mask_3 = cv2.merge([fg_mask, fg_mask, fg_mask])
    return np.where(fg_mask_3 > 0, frame_bgr, bg_bgr)

def main():
    cap = cv2.VideoCapture(2)  # OBS virtual camera
    if not cap.isOpened():
        print("[ERROR] Cannot open camera index 2 (OBS).")
        return

    folder_path = os.path.dirname(os.path.abspath(__file__))
    bg_bgr = load_background(folder_path)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    print("[INFO] Running Background Replace from OBS camera...")

    with build_segmenter(vision.RunningMode.VIDEO) as seg:
        ts = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            fitted_bg = fit_background(bg_bgr, frame.shape)
            res = seg.segment_for_video(to_mp_image_bgr(frame), int(ts))
            mask = res.category_mask.numpy_view()
            comp = replace_bg_frame(frame, mask, fitted_bg)

            cv2.imshow("Background Replace (OBS)", comp)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            ts += int(1000 / fps)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
