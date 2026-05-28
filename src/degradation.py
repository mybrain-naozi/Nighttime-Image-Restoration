from __future__ import annotations

import cv2
import numpy as np

from src.utils import apply_to_channels, clip_image


def normalize_kernel(kernel: np.ndarray) -> np.ndarray:
    kernel = kernel.astype(np.float32)
    total = float(np.sum(kernel))
    if total == 0:
        raise ValueError("Kernel sum cannot be zero")
    return kernel / total


def motion_kernel(length: int = 19, angle: float = 15.0) -> np.ndarray:
    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1.0
    center = (length / 2.0 - 0.5, length / 2.0 - 0.5)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    kernel = cv2.warpAffine(kernel, matrix, (length, length))
    return normalize_kernel(kernel)


def defocus_kernel(radius: int = 5) -> np.ndarray:
    diameter = radius * 2 + 1
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    mask = (xx * xx + yy * yy) <= radius * radius
    kernel = np.zeros((diameter, diameter), dtype=np.float32)
    kernel[mask] = 1.0
    return normalize_kernel(kernel)


def apply_blur(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return apply_to_channels(
        image,
        lambda channel: cv2.filter2D(channel, -1, kernel, borderType=cv2.BORDER_REFLECT),
    )


def add_gaussian_noise(image: np.ndarray, sigma: float = 0.015) -> np.ndarray:
    noise = np.random.normal(loc=0.0, scale=sigma, size=image.shape).astype(np.float32)
    return clip_image(image + noise)


def add_salt_pepper_noise(
    image: np.ndarray,
    amount: float = 0.002,
    salt_vs_pepper: float = 0.5,
) -> np.ndarray:
    noisy = image.copy()
    total = image.shape[0] * image.shape[1]
    salt_count = int(total * amount * salt_vs_pepper)
    pepper_count = int(total * amount * (1.0 - salt_vs_pepper))

    if salt_count > 0:
        coords = (
            np.random.randint(0, image.shape[0], salt_count),
            np.random.randint(0, image.shape[1], salt_count),
        )
        noisy[coords[0], coords[1], :] = 1.0
    if pepper_count > 0:
        coords = (
            np.random.randint(0, image.shape[0], pepper_count),
            np.random.randint(0, image.shape[1], pepper_count),
        )
        noisy[coords[0], coords[1], :] = 0.0
    return noisy


def darken_image(image: np.ndarray, gamma: float = 1.7, gain: float = 0.85) -> np.ndarray:
    darkened = gain * np.power(np.maximum(image, 1e-6), gamma)
    return clip_image(darkened)


def apply_vignette(
    image: np.ndarray,
    strength: float = 0.22,
    sigma: float = 0.85,
) -> np.ndarray:
    height, width = image.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    xx = (xx / max(width - 1, 1) - 0.5) * 2.0
    yy = (yy / max(height - 1, 1) - 0.5) * 2.0
    radius = np.sqrt(xx * xx + yy * yy)
    vignette = np.exp(-(radius * radius) / max(2.0 * sigma * sigma, 1e-6))
    vignette = 1.0 - strength * (1.0 - vignette)
    return clip_image(image * vignette[..., None])


def add_color_cast(
    image: np.ndarray,
    channel_gains: tuple[float, float, float] = (1.05, 1.00, 0.90),
) -> np.ndarray:
    gains = np.array(channel_gains, dtype=np.float32).reshape(1, 1, 3)
    return clip_image(image * gains)


def reduce_contrast(image: np.ndarray, factor: float = 1.0) -> np.ndarray:
    if factor >= 0.999:
        return clip_image(image)
    channel_mean = np.mean(image, axis=(0, 1), keepdims=True)
    return clip_image(channel_mean + factor * (image - channel_mean))


def add_jpeg_artifacts(image: np.ndarray, quality: int = 38) -> np.ndarray:
    image_uint8 = np.round(clip_image(image) * 255.0).astype(np.uint8)
    bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise ValueError("Unable to encode image to JPEG")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("Unable to decode JPEG image")
    decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    return decoded.astype(np.float32) / 255.0


def synthetic_degradation(
    image: np.ndarray,
    kernel: np.ndarray,
    noise_sigma: float = 0.015,
    salt_pepper_amount: float = 0.0,
    gamma: float = 1.7,
    gain: float = 0.85,
) -> np.ndarray:
    degraded = apply_blur(image, kernel)
    degraded = darken_image(degraded, gamma=gamma, gain=gain)
    degraded = add_gaussian_noise(degraded, sigma=noise_sigma)
    if salt_pepper_amount > 0:
        degraded = add_salt_pepper_noise(degraded, amount=salt_pepper_amount)
    return clip_image(degraded)


def surveillance_degradation(
    image: np.ndarray,
    kernel: np.ndarray,
    noise_sigma: float = 0.014,
    salt_pepper_amount: float = 0.0,
    gamma: float = 1.75,
    gain: float = 0.82,
    vignette_strength: float = 0.18,
    contrast_factor: float = 1.0,
    jpeg_quality: int = 38,
    channel_gains: tuple[float, float, float] = (1.04, 1.00, 0.92),
) -> np.ndarray:
    degraded = apply_blur(image, kernel)
    degraded = reduce_contrast(degraded, factor=contrast_factor)
    degraded = darken_image(degraded, gamma=gamma, gain=gain)
    degraded = apply_vignette(degraded, strength=vignette_strength, sigma=0.90)
    degraded = add_color_cast(degraded, channel_gains=channel_gains)
    degraded = add_gaussian_noise(degraded, sigma=noise_sigma)
    if salt_pepper_amount > 0:
        degraded = add_salt_pepper_noise(degraded, amount=salt_pepper_amount)
    degraded = add_jpeg_artifacts(degraded, quality=jpeg_quality)
    return clip_image(degraded)
