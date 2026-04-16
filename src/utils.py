from __future__ import annotations

from pathlib import Path
import shutil
from typing import Sequence

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def reset_dir(path: str | Path) -> Path:
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def clip_image(image: np.ndarray) -> np.ndarray:
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def read_image(path: str | Path) -> np.ndarray:
    path = Path(path)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image.astype(np.float32) / 255.0


def to_uint8(image: np.ndarray) -> np.ndarray:
    return np.round(clip_image(image) * 255.0).astype(np.uint8)


def write_image(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    bgr = cv2.cvtColor(to_uint8(image), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def rgb_to_gray(image: np.ndarray) -> np.ndarray:
    image_uint8 = to_uint8(image)
    gray = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2GRAY)
    return gray.astype(np.float32) / 255.0


def apply_to_channels(image: np.ndarray, func) -> np.ndarray:
    channels = [func(channel) for channel in cv2.split(image)]
    return clip_image(cv2.merge(channels))


def unsharp_mask(image: np.ndarray, sigma: float = 1.0, strength: float = 0.7) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
    sharpened = image + strength * (image - blurred)
    return clip_image(sharpened)


def save_comparison_figure(
    images: Sequence[np.ndarray],
    titles: Sequence[str],
    save_path: str | Path,
    max_cols: int = 4,
) -> None:
    if len(images) != len(titles):
        raise ValueError("images and titles must have the same length")

    save_path = Path(save_path)
    ensure_dir(save_path.parent)

    total = len(images)
    cols = min(max_cols, total)
    rows = int(np.ceil(total / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.2 * rows))
    axes = np.array(axes, ndmin=1).reshape(rows, cols)

    for index, (image, title) in enumerate(zip(images, titles)):
        row = index // cols
        col = index % cols
        axes[row, col].imshow(clip_image(image))
        axes[row, col].set_title(title, fontsize=10)
        axes[row, col].axis("off")

    for index in range(total, rows * cols):
        row = index // cols
        col = index % cols
        axes[row, col].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
