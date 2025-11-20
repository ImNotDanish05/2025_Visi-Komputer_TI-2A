# ================================================================
# SAM 2 Hair Auto-Segmentation (Python, CPU-Optimized)
# ----------------------------------------------------------------
# - Uses SAM 2 (Segment Anything Model 2)
# - Auto-detects face → chooses a prompt point above forehead
# - Applies segmentation only around head area
# - Works with OBS Virtual Camera (Device 2)
# - Press 'q' to quit
#
# Install:
#   pip install opencv-python numpy sam2 torch facenet-pytorch tqdm
#
# NOTE:
# - SAM 2 is CPU-heavy. This version uses small model + optimized resize.
# - Resolution lower = better speed.
#
# © Danish & Aria
# ================================================================

import cv2
import numpy as np
import torch
from sam2 import Sam2Predictor
from facenet_pytorch import MTCNN

# ==========================
# CONFIG
# ==========================
DEVICE = "cpu"
CAM_INDEX = 2        # OBS Virtual Camera
MODEL_NAME = "sam2_hiera_tiny"   # fastest
RESIZE_W = 512       # reduce to 512px for faster CPU inference

# ==========================
# INIT MODELS
# ==========================
print("[INFO] Loading face detector (MTCNN)...")
mtcnn = MTCNN(keep_all=False, device=DEVICE)

print("[INFO] Loading SAM 2 model...")
predictor = Sam2Predictor.from_pretrained(MODEL_NAME)
predictor.to(DEVICE)

# ==========================
# UTILS
# ==========================
def auto_prompt_point_bgr(frame):
    """Detect face → pick point above forehead for SAM prompt."""
    boxes, _ = mtcnn.detect(frame)

    if boxes is None:
        return None  # no face detected

    x1, y1, x2, y2 = boxes[0]
    cx = int((x1 + x2) / 2)
    cy = int(y1) - 15  # slightly above top of face → hair area

    # clamp to frame
    cy = max(0, cy)

    return (cx, cy)

def segment_hair(frame_bgr):
    """Use SAM2 with automatic forehead prompt point."""
    point = auto_prompt_point_bgr(frame_bgr)
    if point is None:
        return frame_bgr

    img_resized = cv2.resize(frame_bgr, (RESIZE_W, RESIZE_W))
    H0, W0 = frame_bgr.shape[:2]

    # Coordinates scale
    sx = point[0] / W0
    sy = point[1] / H0
    prompt_resized = (int(sx * RESIZE_W), int(sy * RESIZE_W))

    # SAM 2 inference
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    predictor.set_image(img_rgb)

    masks, _, _ = predictor.predict(
        point_coords=np.array([[prompt_resized]]),
        point_labels=np.array([[1]]),     # foreground prompt
        multimask_output=False,
    )

    mask = masks[0].astype(np.uint8)

    # Resize mask back to original camera size
    mask_full = cv2.resize(mask, (W0, H0))

    # Apply color tint
    tint = cv2.applyColorMap(mask_full * 255, cv2.COLORMAP_OCEAN)
    blended = cv2.addWeighted(frame_bgr, 1.0, tint, 0.6, 0)

    result = frame_bgr.copy()
    result[mask_full == 1] = blended[mask_full == 1]
    return result

# ==========================
# MAIN LOOP
# ==========================
def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {CAM_INDEX}.")
        return

    print("[INFO] Running SAM 2 Hair Segmentation...")
    print("[INFO] Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # SAM 2 segmentation
        vis = segment_hair(frame)

        cv2.imshow("SAM2 Hair Segmentation", vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ==========================
if __name__ == "__main__":
    main()
