import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import cv2
import numpy as np
import requests
import torch
from importlib import resources

# SAM 2 imports have slightly different entry points across versions, so we try both.
try:  # type: ignore
    from sam2.build_sam import build_sam2 as _build_sam_fn  # type: ignore
except Exception:  # pylint: disable=broad-except
    _build_sam_fn = None

try:  # type: ignore
    from sam2.build_sam import build_sam2_hiera as _build_sam_hiera_fn  # type: ignore
except Exception:  # pylint: disable=broad-except
    _build_sam_hiera_fn = None

try:  # type: ignore
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator  # type: ignore
except Exception as exc:  # pylint: disable=broad-except
    raise ImportError("sam2 package is required for segmentation") from exc


_MODEL_NAME = os.environ.get("SAM2_MODEL_NAME", "sam2_hiera_tiny")
_DEVICE = os.environ.get("SAM2_DEVICE", "cpu")
_CACHE_DIR = Path(os.environ.get("SAM2_CACHE_DIR", ".checkpoints"))

# Default checkpoints from Meta's published weights.
_CHECKPOINT_URLS: Dict[str, str] = {
    "sam2_hiera_tiny": "https://dl.fbaipublicfiles.com/segment_anything_2/checkpoints/sam2_hiera_tiny.pt",
    "sam2_hiera_small": "https://dl.fbaipublicfiles.com/segment_anything_2/checkpoints/sam2_hiera_small.pt",
    "sam2_hiera_base": "https://dl.fbaipublicfiles.com/segment_anything_2/checkpoints/sam2_hiera_base_plus.pt",
}

_MODEL_CONFIG_CANDIDATES: Dict[str, list[str]] = {
    "sam2_hiera_tiny": [
        "configs/sam2.1/sam2.1_hiera_t.yaml",
        "configs/sam2/sam2_hiera_t.yaml",
        "configs/sam2_hiera_t.yaml",
    ],
    "sam2_hiera_small": [
        "configs/sam2.1/sam2.1_hiera_s.yaml",
        "configs/sam2/sam2_hiera_s.yaml",
        "configs/sam2_hiera_s.yaml",
    ],
    "sam2_hiera_base": [
        "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "configs/sam2.1/sam2.1_hiera_b.yaml",
        "configs/sam2/sam2_hiera_b+.yaml",
        "configs/sam2/sam2_hiera_b.yaml",
        "configs/sam2_hiera_b.yaml",
        "configs/sam2_hiera_base.yaml",
        "configs/sam2_hiera_base_plus.yaml",
    ],
}

_CONFIG_URLS: Dict[str, str] = {
    "sam2_hiera_tiny": "https://raw.githubusercontent.com/facebookresearch/sam2/main/configs/sam2.1/sam2.1_hiera_t.yaml",
    "sam2_hiera_small": "https://raw.githubusercontent.com/facebookresearch/sam2/main/configs/sam2.1/sam2.1_hiera_s.yaml",
    "sam2_hiera_base": "https://raw.githubusercontent.com/facebookresearch/sam2/main/configs/sam2.1/sam2.1_hiera_b+.yaml",
}

MASK_GENERATOR: Optional[SAM2AutomaticMaskGenerator] = None


def _get_checkpoint(model_name: str) -> Path:
    """Download checkpoint if missing and return its path."""
    url = _CHECKPOINT_URLS.get(model_name)
    if url is None:
        raise ValueError(f"Unknown model name: {model_name}")

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = _CACHE_DIR / f"{model_name}.pt"
    if checkpoint_path.exists():
        return checkpoint_path

    with requests.get(url, stream=True, timeout=300) as response:
        response.raise_for_status()
        with checkpoint_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 15):
                if chunk:
                    handle.write(chunk)

    return checkpoint_path


def _maybe_find_embedded_config(model_name: str) -> Optional[Path]:
    """Search for a packaged config file inside the installed sam2 package."""
    candidates = _MODEL_CONFIG_CANDIDATES.get(model_name, [])
    if not candidates:
        return None

    try:
        base = resources.files("sam2")
    except Exception:
        return None

    for rel_path in candidates:
        candidate = base / rel_path
        try:
            if candidate.is_file():
                return Path(candidate)
        except FileNotFoundError:
            continue
    return None


def _get_config(model_name: str) -> Path:
    """Resolve a config file path via env override, packaged file, or download."""
    env_path = os.environ.get("SAM2_CONFIG_PATH")
    if env_path:
        cfg_path = Path(env_path)
        if cfg_path.exists():
            return cfg_path
        raise FileNotFoundError(f"SAM2_CONFIG_PATH points to missing file: {cfg_path}")

    embedded = _maybe_find_embedded_config(model_name)
    if embedded:
        return embedded

    url = _CONFIG_URLS.get(model_name)
    if not url:
        raise ValueError(f"Unknown model name for config: {model_name}")

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dest = _CACHE_DIR / f"{model_name}.yaml"
    if dest.exists():
        return dest

    with requests.get(url, stream=True, timeout=300) as response:
        response.raise_for_status()
        with dest.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 15):
                if chunk:
                    handle.write(chunk)
    return dest


def _build_model(checkpoint_path: Path) -> SAM2AutomaticMaskGenerator:
    """Construct the SAM2 mask generator with broad compatibility."""
    if _build_sam_hiera_fn:
        builder: Callable[..., Any] = _build_sam_hiera_fn
    elif _build_sam_fn:
        builder = _build_sam_fn
    else:
        raise ImportError("sam2.build_sam import failed; please verify the sam2 installation.")

    config_path = _get_config(_MODEL_NAME)

    try:
        model = builder(
            config_file=str(config_path),
            model_type=_MODEL_NAME,
            checkpoint=str(checkpoint_path),
            device=_DEVICE,
        )
    except TypeError:
        model = builder(
            checkpoint=str(checkpoint_path),
            model_type=_MODEL_NAME,
            device=_DEVICE,
            config_file=str(config_path),
        )
    except Exception as exc:  # pylint: disable=broad-except
        raise RuntimeError(f"Failed to construct SAM2 model: {exc}") from exc

    if torch.cuda.is_available() and _DEVICE == "cpu":
        # Explicitly keep the model on CPU; SAM2 can be large for mobile demos.
        model.to("cpu")

    return SAM2AutomaticMaskGenerator(model)


def _load_generator() -> SAM2AutomaticMaskGenerator:
    checkpoint_path = _get_checkpoint(_MODEL_NAME)
    return _build_model(checkpoint_path)


# Load the generator once on module import.
MASK_GENERATOR = _load_generator()


def _select_mask(masks: Any) -> Optional[np.ndarray]:
    """Pick the largest/highest scoring binary mask from SAM2 outputs."""
    if not masks:
        return None

    best = max(
        masks,
        key=lambda m: (
            float(m.get("area", 0)),
            float(m.get("predicted_iou", 0)),
        ),
    )
    seg = best.get("segmentation")
    return seg.astype(np.uint8) if seg is not None else None


def segment_and_color(image_bgr: np.ndarray) -> np.ndarray:
    """Run SAM2, pick the primary mask, and apply a colored overlay."""
    if MASK_GENERATOR is None:
        raise RuntimeError("SAM2 mask generator was not initialized.")

    original_h, original_w = image_bgr.shape[:2]
    target_size = 512
    scale_w = target_size
    scale_h = target_size
    resized = cv2.resize(image_bgr, (scale_w, scale_h), interpolation=cv2.INTER_LINEAR)
    image_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    masks = MASK_GENERATOR.generate(image_rgb)
    mask_small = _select_mask(masks)
    if mask_small is None:
        return image_bgr.copy()

    mask = cv2.resize(mask_small, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
    tint = np.zeros_like(image_bgr, dtype=np.uint8)
    tint[:, :] = (255, 140, 40)  # Warm orange overlay in BGR

    output = image_bgr.copy()
    mask_bool = mask.astype(bool)
    output[mask_bool] = cv2.addWeighted(image_bgr[mask_bool], 0.20, tint[mask_bool], 0.80, 0)

    # Emphasize edges of the mask.
    edges = cv2.Canny(mask.astype(np.uint8) * 255, 50, 150)
    output[edges > 0] = (0, 200, 255)

    return output
