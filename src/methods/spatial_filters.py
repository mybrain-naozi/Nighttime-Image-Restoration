from __future__ import annotations

import cv2
import numpy as np

from src.utils import clip_image, to_uint8


def mean_filter(image: np.ndarray, ksize: int = 5) -> np.ndarray:
    return clip_image(cv2.blur(image, (ksize, ksize)))


def gaussian_filter(image: np.ndarray, ksize: int = 5, sigma: float = 1.2) -> np.ndarray:
    return clip_image(cv2.GaussianBlur(image, (ksize, ksize), sigmaX=sigma, sigmaY=sigma))


def median_filter(image: np.ndarray, ksize: int = 5) -> np.ndarray:
    image_uint8 = to_uint8(image)
    filtered = cv2.medianBlur(image_uint8, ksize)
    return filtered.astype(np.float32) / 255.0
