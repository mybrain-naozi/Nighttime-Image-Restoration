from __future__ import annotations

import cv2
import numpy as np

from src.utils import clip_image, rgb_to_gray, to_uint8


def gamma_correction(image: np.ndarray, gamma: float = 0.65) -> np.ndarray:
    corrected = np.power(np.maximum(image, 1e-6), gamma)
    return clip_image(corrected)


def adaptive_gamma(image: np.ndarray, target_mean: float = 0.42) -> np.ndarray:
    gray = rgb_to_gray(image)
    brightness = float(np.mean(gray))
    if brightness >= target_mean:
        return image.copy()
    gamma = np.clip(np.log(target_mean) / np.log(max(brightness, 1e-3)), 0.45, 0.95)
    return gamma_correction(image, gamma=float(gamma))


def clahe_enhance(
    image: np.ndarray,
    clip_limit: float = 2.5,
    tile_grid_size: int = 8,
) -> np.ndarray:
    image_uint8 = to_uint8(image)
    lab = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_grid_size, tile_grid_size),
    )
    lab[..., 0] = clahe.apply(lab[..., 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return enhanced.astype(np.float32) / 255.0


def single_scale_retinex(image: np.ndarray, sigma: float = 45.0) -> np.ndarray:
    image_uint8 = to_uint8(image)
    hsv = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
    value = hsv[..., 2] / 255.0
    blurred = cv2.GaussianBlur(value, (0, 0), sigmaX=sigma, sigmaY=sigma)
    retinex = np.log(value + 1e-3) - np.log(blurred + 1e-3)
    retinex = cv2.normalize(retinex, None, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)
    hsv[..., 2] = np.clip(0.35 * value + 0.65 * retinex, 0.0, 1.0) * 255.0
    restored = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return restored.astype(np.float32) / 255.0


def bilateral_denoise(
    image: np.ndarray,
    diameter: int = 7,
    sigma_color: float = 35.0,
    sigma_space: float = 35.0,
) -> np.ndarray:
    image_uint8 = to_uint8(image)
    filtered = cv2.bilateralFilter(
        image_uint8,
        d=diameter,
        sigmaColor=sigma_color,
        sigmaSpace=sigma_space,
    )
    return filtered.astype(np.float32) / 255.0


def retinex_bilateral(image: np.ndarray) -> np.ndarray:
    retinex = single_scale_retinex(image, sigma=45.0)
    return bilateral_denoise(retinex, diameter=7, sigma_color=30.0, sigma_space=30.0)
