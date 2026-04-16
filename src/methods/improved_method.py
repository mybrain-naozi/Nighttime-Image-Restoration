from __future__ import annotations

import numpy as np

from src.methods.deconv_filters import wiener_filter
from src.methods.enhancement import adaptive_gamma, bilateral_denoise, retinex_bilateral
from src.utils import clip_image, rgb_to_gray, unsharp_mask


def adaptive_night_restoration(image: np.ndarray) -> np.ndarray:
    gamma_boosted = adaptive_gamma(image, target_mean=0.36)
    base_restored = retinex_bilateral(gamma_boosted)
    dark_mask = np.clip((0.42 - rgb_to_gray(image)) / 0.42, 0.0, 1.0)[..., None]
    fused = clip_image(base_restored * (1.0 - 0.22 * dark_mask) + gamma_boosted * (0.22 * dark_mask))
    return clip_image(unsharp_mask(fused, sigma=1.0, strength=0.12))


def guided_night_restoration(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    stage1 = adaptive_gamma(image, target_mean=0.38)
    stage2 = bilateral_denoise(stage1, diameter=7, sigma_color=24.0, sigma_space=24.0)
    stage3 = wiener_filter(stage2, kernel, k=0.018)
    blended = clip_image(0.75 * stage2 + 0.25 * stage3)
    return clip_image(unsharp_mask(blended, sigma=1.2, strength=0.18))
