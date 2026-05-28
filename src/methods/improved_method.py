from __future__ import annotations

import cv2
import numpy as np

from src.methods.deconv_filters import wiener_filter
from src.methods.enhancement import bilateral_denoise
from src.utils import clip_image, rgb_to_gray, to_uint8, unsharp_mask


def _inverse_gamma(image: np.ndarray, gamma: float, gain: float) -> np.ndarray:
    safe = np.maximum(image / max(gain, 1e-6), 1e-6)
    inv_gamma = 1.0 / max(gamma, 1e-6)
    return clip_image(np.power(safe, inv_gamma))


def _gray_world_balance(image: np.ndarray) -> np.ndarray:
    channel_means = np.mean(image, axis=(0, 1), keepdims=True)
    reference = float(np.mean(channel_means))
    gains = reference / np.clip(channel_means, 1e-4, None)
    gains = np.clip(gains, 0.85, 1.18)
    return clip_image(image * gains)


def _remove_vignette(image: np.ndarray, strength: float = 0.15, sigma: float = 0.90) -> np.ndarray:
    height, width = image.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    xx = (xx / max(width - 1, 1) - 0.5) * 2.0
    yy = (yy / max(height - 1, 1) - 0.5) * 2.0
    radius = np.sqrt(xx * xx + yy * yy)
    vignette = np.exp(-(radius * radius) / max(2.0 * sigma * sigma, 1e-6))
    correction = 1.0 - strength * (1.0 - vignette)
    correction = np.clip(correction, 0.3, 1.0)
    return clip_image(image / correction[..., None])


def _restore_contrast(image: np.ndarray, factor: float = 1.0, gain: float = 1.0) -> np.ndarray:
    if factor >= 0.999:
        return clip_image(image)
    channel_mean = np.mean(image, axis=(0, 1), keepdims=True)
    contrast_gain = gain / max(factor, 1e-3)
    return clip_image(channel_mean + contrast_gain * (image - channel_mean))


def _remove_color_cast(image: np.ndarray, channel_gains: tuple[float, float, float]) -> np.ndarray:
    gains = np.array(channel_gains, dtype=np.float32).reshape(1, 1, 3)
    return clip_image(image / np.maximum(gains, 1e-3))


def _richardson_lucy_deconv(
    image: np.ndarray,
    kernel: np.ndarray,
    iterations: int = 6,
) -> np.ndarray:
    psf = kernel.astype(np.float32)
    psf_flipped = np.flip(psf)
    restored_channels = []
    for channel in cv2.split(np.clip(image, 1e-4, 1.0).astype(np.float32)):
        estimate = channel.copy()
        for _ in range(iterations):
            blurred = cv2.filter2D(estimate, -1, psf, borderType=cv2.BORDER_REFLECT)
            relative_blur = channel / np.maximum(blurred, 1e-4)
            correction = cv2.filter2D(relative_blur, -1, psf_flipped, borderType=cv2.BORDER_REFLECT)
            estimate = np.clip(estimate * correction, 0.0, 1.0)
        restored_channels.append(estimate)
    return clip_image(cv2.merge(restored_channels))


def _edge_weighted_blend(
    base: np.ndarray,
    deblurred: np.ndarray,
    min_weight: float = 0.0,
    max_weight: float = 0.25,
) -> np.ndarray:
    gray = rgb_to_gray(base)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge = cv2.magnitude(grad_x, grad_y)
    high = float(np.percentile(edge, 96))
    if high <= 1e-5:
        weight = float(np.clip(max_weight, 0.0, 1.0))
        return clip_image((1.0 - weight) * base + weight * deblurred)

    edge_mask = np.clip(edge / high, 0.0, 1.0)
    edge_mask = cv2.GaussianBlur(edge_mask, (0, 0), sigmaX=1.2, sigmaY=1.2)
    min_weight = float(np.clip(min_weight, 0.0, 1.0))
    max_weight = float(np.clip(max_weight, min_weight, 1.0))
    weight = np.clip(min_weight + (max_weight - min_weight) * edge_mask, min_weight, max_weight)[..., None]
    return clip_image((1.0 - weight) * base + weight * deblurred)


def _detail_boost(
    image: np.ndarray,
    strength: float = 0.0,
    edge_strength: float = 0.0,
) -> np.ndarray:
    if strength <= 1e-6 and edge_strength <= 1e-6:
        return clip_image(image)

    lab = cv2.cvtColor(to_uint8(image), cv2.COLOR_RGB2LAB).astype(np.float32)
    luminance = lab[..., 0] / 255.0
    smooth = cv2.GaussianBlur(luminance, (0, 0), sigmaX=0.75, sigmaY=0.75)
    detail = luminance - smooth

    grad_x = cv2.Sobel(luminance, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(luminance, cv2.CV_32F, 0, 1, ksize=3)
    edge = cv2.magnitude(grad_x, grad_y)
    high = float(np.percentile(edge, 96))
    if high <= 1e-5:
        edge_mask = 0.0
    else:
        edge_mask = np.clip(edge / high, 0.0, 1.0)

    boosted = np.clip(luminance + strength * detail + edge_strength * edge_mask * detail, 0.0, 1.0)
    lab[..., 0] = boosted * 255.0
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0


def _clahe_on_luminance(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: int = 8,
) -> np.ndarray:
    image_uint8 = to_uint8(image)
    lab = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    lab[..., 0] = clahe.apply(lab[..., 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return enhanced.astype(np.float32) / 255.0


def adaptive_night_restoration(image: np.ndarray) -> np.ndarray:
    gray = rgb_to_gray(image)
    mean_brightness = float(np.mean(gray))
    target_mean = 0.40
    if mean_brightness < target_mean:
        gamma = float(
            np.clip(
                np.log(target_mean) / np.log(max(mean_brightness, 1e-3)),
                0.40,
                0.90,
            )
        )
        brightened = clip_image(np.power(np.maximum(image, 1e-6), gamma))
    else:
        brightened = image.copy()

    denoised = bilateral_denoise(brightened, diameter=7, sigma_color=28.0, sigma_space=28.0)
    enhanced = _clahe_on_luminance(denoised, clip_limit=2.2, tile_grid_size=8)
    fused = clip_image(0.80 * enhanced + 0.20 * denoised)
    return clip_image(unsharp_mask(fused, sigma=1.0, strength=0.10))


def guided_night_restoration(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    gray = rgb_to_gray(image)
    mean_brightness = float(np.mean(gray))
    target_mean = 0.38
    if mean_brightness < target_mean:
        gamma = float(
            np.clip(
                np.log(target_mean) / np.log(max(mean_brightness, 1e-3)),
                0.45,
                0.95,
            )
        )
        brightened = clip_image(np.power(np.maximum(image, 1e-6), gamma))
    else:
        brightened = image.copy()

    denoised = bilateral_denoise(brightened, diameter=7, sigma_color=24.0, sigma_space=24.0)
    deblurred = wiener_filter(denoised, kernel, k=0.018)
    blended = clip_image(0.65 * deblurred + 0.35 * denoised)
    return clip_image(unsharp_mask(blended, sigma=1.2, strength=0.15))


def _patch_adaptive_blend(
    original: np.ndarray,
    restored: np.ndarray,
    patch_size: int = 32,
) -> np.ndarray:
    h, w = original.shape[:2]
    result = restored.copy()
    diff_sq = (restored.astype(np.float64) - original.astype(np.float64)) ** 2

    for y0 in range(0, h, patch_size):
        y1 = min(y0 + patch_size, h)
        for x0 in range(0, w, patch_size):
            x1 = min(x0 + patch_size, w)
            patch_orig = original[y0:y1, x0:x1].astype(np.float64)
            patch_rest = restored[y0:y1, x0:x1].astype(np.float64)
            local_mse = float(np.mean(diff_sq[y0:y1, x0:x1]))
            if local_mse > 0.012:
                alpha = float(np.clip((local_mse - 0.012) / 0.02, 0.0, 0.60))
                result[y0:y1, x0:x1] = clip_image(
                    (1.0 - alpha) * patch_rest + alpha * patch_orig
                )
    return result


def surveillance_guided_restoration_steps(
    image: np.ndarray,
    kernel: np.ndarray,
    *,
    degrade_gamma: float = 1.65,
    degrade_gain: float = 0.80,
    degrade_vignette_strength: float = 0.15,
    degrade_contrast_factor: float = 1.0,
    degrade_channel_gains: tuple[float, float, float] = (1.0, 1.0, 1.0),
    deblur_iterations: int = 6,
    deblur_min_weight: float = 0.0,
    deblur_weight: float = 0.15,
    contrast_gain: float = 0.94,
    denoise_sigma: float = 2.5,
    sharpen_sigma: float = 0.6,
    sharpen_strength: float = 0.15,
    detail_strength: float = 0.0,
    edge_detail_strength: float = 0.0,
) -> dict[str, np.ndarray]:
    no_extra_lowlight = (
        abs(degrade_gamma - 1.0) <= 1e-3
        and abs(degrade_gain - 1.0) <= 1e-3
        and degrade_vignette_strength <= 1e-3
    )
    if no_extra_lowlight:
        deblurred = _richardson_lucy_deconv(image, kernel, iterations=deblur_iterations)
        blended = _edge_weighted_blend(
            image,
            deblurred,
            min_weight=deblur_min_weight,
            max_weight=deblur_weight,
        )
        denoised = bilateral_denoise(blended, diameter=3, sigma_color=denoise_sigma, sigma_space=denoise_sigma)
        sharpened = unsharp_mask(denoised, sigma=sharpen_sigma, strength=sharpen_strength)
        final = _detail_boost(sharpened, strength=detail_strength, edge_strength=edge_detail_strength)
        final = clip_image(final)
        return {
            "退化图": image,
            "过程1：反卷积去模糊": deblurred,
            "过程2：边缘加权融合": blended,
            "过程3：降噪锐化": sharpened,
            "最终改进复原": final,
        }

    color_corrected = _remove_color_cast(image, degrade_channel_gains)
    brightness_restored = _inverse_gamma(color_corrected, gamma=degrade_gamma, gain=degrade_gain)
    vignette_restored = brightness_restored
    if degrade_vignette_strength > 1e-3:
        vignette_restored = _remove_vignette(
            brightness_restored,
            strength=degrade_vignette_strength,
            sigma=0.90,
        )

    deblurred = _richardson_lucy_deconv(vignette_restored, kernel, iterations=deblur_iterations)
    edge_blended = _edge_weighted_blend(
        vignette_restored,
        deblurred,
        min_weight=deblur_min_weight,
        max_weight=deblur_weight,
    )
    contrast_restored = _restore_contrast(edge_blended, factor=degrade_contrast_factor, gain=contrast_gain)
    denoised = bilateral_denoise(contrast_restored, diameter=3, sigma_color=denoise_sigma, sigma_space=denoise_sigma)
    sharpened = unsharp_mask(denoised, sigma=sharpen_sigma, strength=sharpen_strength)
    final = _detail_boost(sharpened, strength=detail_strength, edge_strength=edge_detail_strength)
    final = clip_image(final)
    return {
        "退化图": image,
        "过程1：去色偏": color_corrected,
        "过程2：亮度恢复": brightness_restored,
        "过程3：去暗角": vignette_restored,
        "过程4：反卷积去模糊": deblurred,
        "过程5：边缘加权融合": edge_blended,
        "过程6：降噪锐化": sharpened,
        "最终改进复原": final,
    }


def surveillance_guided_restoration(
    image: np.ndarray,
    kernel: np.ndarray,
    *,
    degrade_gamma: float = 1.65,
    degrade_gain: float = 0.80,
    degrade_vignette_strength: float = 0.15,
    degrade_contrast_factor: float = 1.0,
    degrade_channel_gains: tuple[float, float, float] = (1.0, 1.0, 1.0),
    deblur_iterations: int = 6,
    deblur_min_weight: float = 0.0,
    deblur_weight: float = 0.15,
    contrast_gain: float = 0.94,
    denoise_sigma: float = 2.5,
    sharpen_sigma: float = 0.6,
    sharpen_strength: float = 0.15,
    detail_strength: float = 0.0,
    edge_detail_strength: float = 0.0,
) -> np.ndarray:
    steps = surveillance_guided_restoration_steps(
        image,
        kernel,
        degrade_gamma=degrade_gamma,
        degrade_gain=degrade_gain,
        degrade_vignette_strength=degrade_vignette_strength,
        degrade_contrast_factor=degrade_contrast_factor,
        degrade_channel_gains=degrade_channel_gains,
        deblur_iterations=deblur_iterations,
        deblur_min_weight=deblur_min_weight,
        deblur_weight=deblur_weight,
        contrast_gain=contrast_gain,
        denoise_sigma=denoise_sigma,
        sharpen_sigma=sharpen_sigma,
        sharpen_strength=sharpen_strength,
        detail_strength=detail_strength,
        edge_detail_strength=edge_detail_strength,
    )
    return steps["最终改进复原"]


def surveillance_scene_restoration(image: np.ndarray) -> np.ndarray:
    image_uint8 = to_uint8(image)
    lab = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2LAB).astype(np.float32)
    luminance = lab[..., 0] / 255.0
    mean_luminance = float(np.mean(luminance))

    target_luminance = min(0.38, mean_luminance + 0.16)
    if mean_luminance < target_luminance:
        gamma = float(
            np.clip(
                np.log(target_luminance) / np.log(max(mean_luminance, 1e-3)),
                0.50,
                0.85,
            )
        )
        enhanced_luminance = np.power(np.clip(luminance, 1e-4, 1.0), gamma)
    else:
        enhanced_luminance = luminance.copy()

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_luminance = clahe.apply(np.round(luminance * 255.0).astype(np.uint8)).astype(np.float32) / 255.0

    shadow_mask = np.clip((0.50 - luminance) / 0.50, 0.0, 1.0) ** 1.2
    blended_luminance = (
        enhanced_luminance * (0.60 * shadow_mask + 0.40)
        + clahe_luminance * (0.40 * shadow_mask)
    )
    blended_luminance = np.clip(blended_luminance, 0.0, 1.0)

    lab[..., 0] = np.clip(blended_luminance * 255.0, 0.0, 255.0)
    restored = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
    restored = bilateral_denoise(restored, diameter=5, sigma_color=16.0, sigma_space=16.0)
    return clip_image(unsharp_mask(restored, sigma=0.9, strength=0.06))
