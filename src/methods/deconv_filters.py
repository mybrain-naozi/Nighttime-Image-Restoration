from __future__ import annotations

import cv2
import numpy as np

from src.utils import clip_image


LAPLACIAN_KERNEL = np.array(
    [
        [0.0, -1.0, 0.0],
        [-1.0, 4.0, -1.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float32,
)


def psf_to_otf(psf: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    psf = psf.astype(np.float32)
    padded = np.zeros(shape, dtype=np.float32)
    kh, kw = psf.shape
    padded[:kh, :kw] = psf
    padded = np.roll(padded, -kh // 2, axis=0)
    padded = np.roll(padded, -kw // 2, axis=1)
    return np.fft.fft2(padded)


def _restore_channel(
    channel: np.ndarray,
    kernel: np.ndarray,
    mode: str,
    eps: float = 1e-3,
    k: float = 0.01,
    gamma: float = 0.001,
) -> np.ndarray:
    height, width = channel.shape
    spectrum = np.fft.fft2(channel)
    transfer = psf_to_otf(kernel, (height, width))
    transfer_conj = np.conj(transfer)

    if mode == "inverse":
        restored_freq = spectrum * transfer_conj / (np.abs(transfer) ** 2 + eps)
    elif mode == "wiener":
        restored_freq = spectrum * transfer_conj / (np.abs(transfer) ** 2 + k)
    elif mode == "cls":
        regularizer = psf_to_otf(LAPLACIAN_KERNEL, (height, width))
        restored_freq = spectrum * transfer_conj / (
            np.abs(transfer) ** 2 + gamma * (np.abs(regularizer) ** 2)
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    restored = np.real(np.fft.ifft2(restored_freq))
    return clip_image(restored)


def _restore_image(image: np.ndarray, kernel: np.ndarray, **kwargs) -> np.ndarray:
    channels = cv2.split(image.astype(np.float32))
    restored_channels = [_restore_channel(channel, kernel, **kwargs) for channel in channels]
    return clip_image(cv2.merge(restored_channels))


def inverse_filter(image: np.ndarray, kernel: np.ndarray, eps: float = 3e-3) -> np.ndarray:
    return _restore_image(image, kernel, mode="inverse", eps=eps)


def wiener_filter(image: np.ndarray, kernel: np.ndarray, k: float = 0.01) -> np.ndarray:
    return _restore_image(image, kernel, mode="wiener", k=k)


def constrained_least_squares_filter(
    image: np.ndarray,
    kernel: np.ndarray,
    gamma: float = 0.002,
) -> np.ndarray:
    return _restore_image(image, kernel, mode="cls", gamma=gamma)
