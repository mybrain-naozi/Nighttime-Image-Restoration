from __future__ import annotations

import cv2
import numpy as np


def rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    diff = prediction.astype(np.float64) - target.astype(np.float64)
    return float(np.sqrt(np.mean(diff * diff)))


def psnr(prediction: np.ndarray, target: np.ndarray, max_value: float = 1.0) -> float:
    mse = np.mean((prediction.astype(np.float64) - target.astype(np.float64)) ** 2)
    if mse <= 1e-12:
        return 99.0
    return float(20.0 * np.log10(max_value / np.sqrt(mse)))


def _ssim_single_channel(channel_a: np.ndarray, channel_b: np.ndarray) -> float:
    channel_a = channel_a.astype(np.float64)
    channel_b = channel_b.astype(np.float64)

    c1 = (0.01 * 1.0) ** 2
    c2 = (0.03 * 1.0) ** 2

    mu_a = cv2.GaussianBlur(channel_a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(channel_b, (11, 11), 1.5)

    mu_a_sq = mu_a * mu_a
    mu_b_sq = mu_b * mu_b
    mu_ab = mu_a * mu_b

    sigma_a_sq = cv2.GaussianBlur(channel_a * channel_a, (11, 11), 1.5) - mu_a_sq
    sigma_b_sq = cv2.GaussianBlur(channel_b * channel_b, (11, 11), 1.5) - mu_b_sq
    sigma_ab = cv2.GaussianBlur(channel_a * channel_b, (11, 11), 1.5) - mu_ab

    numerator = (2 * mu_ab + c1) * (2 * sigma_ab + c2)
    denominator = (mu_a_sq + mu_b_sq + c1) * (sigma_a_sq + sigma_b_sq + c2)
    ssim_map = numerator / (denominator + 1e-12)
    return float(np.mean(ssim_map))


def ssim(prediction: np.ndarray, target: np.ndarray) -> float:
    if prediction.ndim == 2:
        return _ssim_single_channel(prediction, target)
    channel_scores = [
        _ssim_single_channel(prediction[..., channel], target[..., channel])
        for channel in range(prediction.shape[2])
    ]
    return float(np.mean(channel_scores))


def evaluate_image(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    return {
        "psnr": psnr(prediction, target),
        "ssim": ssim(prediction, target),
        "rmse": rmse(prediction, target),
    }
