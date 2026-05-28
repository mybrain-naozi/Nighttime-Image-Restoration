from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.degradation import apply_blur, defocus_kernel, motion_kernel
from src.metrics import evaluate_image
from src.methods.deconv_filters import constrained_least_squares_filter, inverse_filter, wiener_filter
from src.methods.improved_method import surveillance_guided_restoration_steps
from src.path_config import (
    DEFAULT_DATASET_COUNT,
    FINAL_DATASET_DIR as DEFAULT_DATASET_DIR,
    LLVIP_VISIBLE_DIR as DEFAULT_SOURCE_DIR,
    SURVEILLANCE_RESULT_DIR as DEFAULT_OUTPUT_ROOT,
    PROJECT_ROOT,
    display_path,
    path_help_message,
    resolve_project_path,
)
from src.utils import ensure_dir, read_image, reset_dir, save_comparison_figure, write_image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DEFAULT_LONG_EDGE = 640


METHOD_NAMES = {
    "degraded": "退化图",
    "inverse_filter": "逆滤波",
    "wiener_filter": "维纳滤波",
    "cls_filter": "约束最小二乘",
    "improved_restoration": "改进复原",
}

CASE_NAMES = {
    "motion_blur": "运动模糊",
    "defocus_blur": "散焦模糊",
}


def scan_images(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Input folder not found: {folder}\n{path_help_message()}")

    images = sorted(
        [path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda path: path.name,
    )
    if not images:
        raise RuntimeError(f"No images found in: {folder}")
    return images


def scan_images_recursive(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Source folder not found: {folder}\n{path_help_message()}")

    images = sorted(
        [path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda path: str(path.relative_to(folder)).replace("\\", "/"),
    )
    if not images:
        raise RuntimeError(f"No images found in: {folder}")
    return images


def resize_long_edge(image: np.ndarray, long_edge: int = DEFAULT_LONG_EDGE) -> np.ndarray:
    height, width = image.shape[:2]
    current_long_edge = max(height, width)
    if current_long_edge <= long_edge:
        return image

    scale = long_edge / current_long_edge
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32)


def build_reference_image(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(np.round(image * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    mean_brightness = float(np.mean(gray))
    target_mean = 0.46
    gamma = float(np.clip(np.log(target_mean) / np.log(max(mean_brightness, 1e-3)), 0.48, 0.88))
    brightened = np.power(np.maximum(image, 1e-6), gamma)

    lab = cv2.cvtColor(np.round(np.clip(brightened, 0.0, 1.0) * 255.0).astype(np.uint8), cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    lab[..., 0] = clahe.apply(lab[..., 0])
    contrast = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0

    denoised = cv2.bilateralFilter(contrast, d=5, sigmaColor=18.0, sigmaSpace=18.0)
    blurred = cv2.GaussianBlur(denoised, (0, 0), sigmaX=0.9, sigmaY=0.9)
    sharpened = denoised + 0.08 * (denoised - blurred)
    return np.clip(0.68 * sharpened + 0.32 * brightened, 0.0, 1.0).astype(np.float32)


def prepare_reference_dataset(
    source_dir: str | Path = DEFAULT_SOURCE_DIR,
    output_dir: str | Path = DEFAULT_DATASET_DIR,
    count: int = DEFAULT_DATASET_COUNT,
    long_edge: int = DEFAULT_LONG_EDGE,
) -> Path:
    source_dir = resolve_project_path(source_dir)
    output_dir = resolve_project_path(output_dir)
    source_paths = scan_images_recursive(source_dir)
    if len(source_paths) < count:
        raise RuntimeError(f"Need {count} source images, but only found {len(source_paths)} in {source_dir}")

    output_dir = reset_dir(output_dir)
    selected_indices = np.linspace(0, len(source_paths) - 1, count, dtype=int)
    selected_paths = [source_paths[index] for index in selected_indices]

    records: list[dict[str, object]] = []
    for index, source_path in enumerate(selected_paths, start=1):
        image = resize_long_edge(read_image(source_path), long_edge=long_edge)
        reference = np.clip(image, 0.0, 1.0).astype(np.float32)
        target_name = f"{index:03d}_{source_path.stem}.jpg"
        target_path = output_dir / target_name
        write_image(target_path, reference)
        records.append(
            {
                "index": index,
                "source": display_path(source_path),
                "reference": display_path(target_path),
            }
        )

    pd.DataFrame(records).to_csv(output_dir / "dataset_index.csv", index=False, encoding="utf-8-sig")
    return output_dir


def metric_title(title: str, metrics: dict[str, float] | None) -> str:
    if metrics is None:
        return title
    return f"{title}\nPSNR {metrics['psnr']:.2f} | SSIM {metrics['ssim']:.3f} | RMSE {metrics['rmse']:.3f}"


def build_detail_table(df: pd.DataFrame) -> pd.DataFrame:
    table = df.copy()
    table["case_name"] = table["case"].map(CASE_NAMES)
    table["method_name"] = table["method_key"].map(METHOD_NAMES)
    return table[["case_name", "image", "method_name", "psnr", "ssim", "rmse"]].rename(
        columns={"case_name": "case", "image": "image", "method_name": "method", "psnr": "PSNR", "ssim": "SSIM", "rmse": "RMSE"}
    )


def build_summary_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    table = summary_df.copy().reset_index()
    table["method_name"] = table["method_key"].map(METHOD_NAMES)
    return table[["method_name", "psnr", "ssim", "rmse"]].rename(
        columns={"method_name": "method", "psnr": "PSNR", "ssim": "SSIM", "rmse": "RMSE"}
    )


def build_case_summary_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    table = summary_df.copy().reset_index()
    table["case_name"] = table["case"].map(CASE_NAMES)
    table["method_name"] = table["method_key"].map(METHOD_NAMES)
    return table[["case_name", "method_name", "psnr", "ssim", "rmse"]].rename(
        columns={"case_name": "case", "method_name": "method", "psnr": "PSNR", "ssim": "SSIM", "rmse": "RMSE"}
    )


def build_improved_best20(output_root: Path, df: pd.DataFrame, limit: int = 20) -> Path:
    best_dir = reset_dir(output_root / "improved_best20")
    degraded_rows = df[df["method_key"] == "degraded"].set_index(["case", "image"])
    improved_rows = df[df["method_key"] == "improved_restoration"].set_index(["case", "image"])

    candidates: list[dict[str, object]] = []
    for (case_key, image_name), improved_row in improved_rows.iterrows():
        if (case_key, image_name) not in degraded_rows.index:
            continue
        degraded_row = degraded_rows.loc[(case_key, image_name)]
        psnr_gain = float(improved_row["psnr"]) - float(degraded_row["psnr"])
        ssim_gain = float(improved_row["ssim"]) - float(degraded_row["ssim"])
        rmse_drop = float(degraded_row["rmse"]) - float(improved_row["rmse"])
        improvement_score = psnr_gain + 20.0 * ssim_gain + 20.0 * rmse_drop
        candidates.append(
            {
                "case": case_key,
                "image": image_name,
                "psnr": float(improved_row["psnr"]),
                "ssim": float(improved_row["ssim"]),
                "rmse": float(improved_row["rmse"]),
                "degraded_psnr": float(degraded_row["psnr"]),
                "degraded_ssim": float(degraded_row["ssim"]),
                "degraded_rmse": float(degraded_row["rmse"]),
                "psnr_gain": psnr_gain,
                "ssim_gain": ssim_gain,
                "rmse_drop": rmse_drop,
                "improvement_score": improvement_score,
            }
        )

    if not candidates:
        pd.DataFrame().to_csv(best_dir / "ranking.csv", index=False, encoding="utf-8-sig")
        return best_dir

    improved_rows = (
        pd.DataFrame(candidates)
        .sort_values(by=["improvement_score", "psnr_gain", "ssim_gain"], ascending=[False, False, False])
        .head(limit)
        .reset_index(drop=True)
    )

    ranking_rows: list[dict[str, object]] = []
    copied_count = 0
    for index, row in improved_rows.iterrows():
        image_name = str(row["image"])
        case_key = str(row["case"])
        source_path = output_root / "improved" / f"{image_name}_{case_key}_improved.png"
        if not source_path.exists():
            continue

        copied_count += 1
        rank = copied_count
        psnr = float(row["psnr"])
        ssim = float(row["ssim"])
        rmse = float(row["rmse"])
        psnr_gain = float(row["psnr_gain"])
        ssim_gain = float(row["ssim_gain"])
        target_name = (
            f"{rank:02d}_{image_name}_{case_key}"
            f"_GAIN_{psnr_gain:.2f}_SSIM_GAIN_{ssim_gain:.3f}.png"
        )
        target_path = best_dir / target_name
        target_path.write_bytes(source_path.read_bytes())
        ranking_rows.append(
            {
                "rank": rank,
                "image": image_name,
                "case": CASE_NAMES.get(case_key, case_key),
                "improvement_score": round(float(row["improvement_score"]), 4),
                "PSNR_gain": round(psnr_gain, 4),
                "SSIM_gain": round(ssim_gain, 4),
                "RMSE_drop": round(float(row["rmse_drop"]), 4),
                "degraded_PSNR": round(float(row["degraded_psnr"]), 4),
                "degraded_SSIM": round(float(row["degraded_ssim"]), 4),
                "degraded_RMSE": round(float(row["degraded_rmse"]), 4),
                "PSNR": round(psnr, 4),
                "SSIM": round(ssim, 4),
                "RMSE": round(rmse, 4),
                "source": str(source_path),
                "copied": str(target_path),
            }
        )

    pd.DataFrame(ranking_rows).to_csv(best_dir / "ranking.csv", index=False, encoding="utf-8-sig")
    return best_dir


def focus_crop_box(image):
    height, width = image.shape[:2]
    crop_width = int(width * 0.24)
    crop_height = int(height * 0.30)
    crop_width = max(48, min(crop_width, width))
    crop_height = max(48, min(crop_height, height))

    gray = cv2.cvtColor(np.round(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gray = gray.astype(np.float32) / 255.0
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    detail = cv2.magnitude(grad_x, grad_y)
    brightness = cv2.GaussianBlur(gray, (0, 0), sigmaX=3.0, sigmaY=3.0)
    score = detail * np.clip(brightness + 0.15, 0.0, 1.0)
    score = cv2.boxFilter(score, ddepth=-1, ksize=(crop_width, crop_height), normalize=True)
    _, _, _, max_location = cv2.minMaxLoc(score)
    center_x, center_y = max_location

    x0 = int(np.clip(center_x - crop_width // 2, 0, max(width - crop_width, 0)))
    y0 = int(np.clip(center_y - crop_height // 2, 0, max(height - crop_height, 0)))
    x1 = min(x0 + crop_width, width)
    y1 = min(y0 + crop_height, height)
    return x0, y0, x1, y1


def focus_crop(image, box):
    x0, y0, x1, y1 = box
    return image[y0:y1, x0:x1]


def run_surveillance_demo(
    input_dir: str | Path,
    output_root: str | Path,
    limit: int | None = None,
) -> dict[str, Path | int]:
    input_dir = resolve_project_path(input_dir)
    output_root = reset_dir(resolve_project_path(output_root))
    traditional_dir = ensure_dir(output_root / "traditional")
    improved_dir = ensure_dir(output_root / "improved")
    comparison_dir = ensure_dir(output_root / "comparison")

    image_paths = scan_images(input_dir)
    if limit is not None:
        image_paths = image_paths[:limit]

    np.random.seed(20260513)

    cases = [
        {
            "name": "motion_blur",
            "kernel": motion_kernel(length=27, angle=18.0),
            "noise_sigma": 0.0,
            "salt_pepper": 0.0,
            "gamma": 1.0,
            "gain": 1.0,
            "vignette_strength": 0.0,
            "contrast_factor": 1.0,
            "jpeg_quality": 100,
            "channel_gains": (1.0, 1.0, 1.0),
            "deblur_iterations": 85,
            "deblur_min_weight": 0.58,
            "deblur_weight": 0.98,
            "contrast_gain": 1.0,
            "denoise_sigma": 0.16,
            "sharpen_sigma": 0.22,
            "sharpen_strength": 0.88,
            "detail_strength": 0.32,
            "edge_detail_strength": 0.42,
        },
        {
            "name": "defocus_blur",
            "kernel": defocus_kernel(radius=9),
            "noise_sigma": 0.0,
            "salt_pepper": 0.0,
            "gamma": 1.0,
            "gain": 1.0,
            "vignette_strength": 0.0,
            "contrast_factor": 1.0,
            "jpeg_quality": 100,
            "channel_gains": (1.0, 1.0, 1.0),
            "deblur_iterations": 110,
            "deblur_min_weight": 0.64,
            "deblur_weight": 0.99,
            "contrast_gain": 1.0,
            "denoise_sigma": 0.16,
            "sharpen_sigma": 0.22,
            "sharpen_strength": 0.92,
            "detail_strength": 0.34,
            "edge_detail_strength": 0.44,
        },
    ]

    records: list[dict[str, object]] = []
    for image_path in image_paths:
        gt_image = read_image(image_path)
        comparison_images: list[np.ndarray] = []
        comparison_titles: list[str] = []

        for case in cases:
            degraded = apply_blur(gt_image, case["kernel"])

            degraded_metrics = evaluate_image(degraded, gt_image)
            records.append(
                {
                    "case": case["name"],
                    "image": image_path.stem,
                    "method_key": "degraded",
                    **degraded_metrics,
                }
            )

            traditional_methods = OrderedDict(
                [
                    ("inverse_filter", lambda image, kernel=case["kernel"]: inverse_filter(image, kernel, eps=0.0045)),
                    ("wiener_filter", lambda image, kernel=case["kernel"]: wiener_filter(image, kernel, k=0.013)),
                    (
                        "cls_filter",
                        lambda image, kernel=case["kernel"]: constrained_least_squares_filter(
                            image,
                            kernel,
                            gamma=0.0025,
                        ),
                    ),
                ]
            )

            case_name = CASE_NAMES[case["name"]]
            traditional_results: list[tuple[str, np.ndarray, dict[str, float]]] = []
            traditional_images = [degraded]
            traditional_titles = [metric_title(f"{case_name}：退化图", degraded_metrics)]

            for method_key, method in traditional_methods.items():
                restored = method(degraded)
                restored_metrics = evaluate_image(restored, gt_image)
                traditional_results.append((method_key, restored, restored_metrics))
                traditional_images.append(restored)
                traditional_titles.append(metric_title(METHOD_NAMES[method_key], restored_metrics))
                records.append(
                    {
                        "case": case["name"],
                        "image": image_path.stem,
                        "method_key": method_key,
                        **restored_metrics,
                    }
                )

            best_traditional_key, best_traditional_image, best_traditional_metrics = max(
                traditional_results,
                key=lambda item: (item[2]["ssim"], item[2]["psnr"]),
            )
            traditional_images.extend([best_traditional_image, gt_image])
            traditional_titles.extend(
                [
                    metric_title(f"最终传统复原：{METHOD_NAMES[best_traditional_key]}", best_traditional_metrics),
                    "原图 / 标准图",
                ]
            )
            save_comparison_figure(
                traditional_images,
                traditional_titles,
                traditional_dir / f"{image_path.stem}_{case['name']}_traditional.png",
                max_cols=3,
            )

            improved_steps = surveillance_guided_restoration_steps(
                degraded,
                case["kernel"],
                degrade_gamma=case["gamma"],
                degrade_gain=case["gain"],
                degrade_vignette_strength=case["vignette_strength"],
                degrade_contrast_factor=case["contrast_factor"],
                degrade_channel_gains=case["channel_gains"],
                deblur_iterations=case["deblur_iterations"],
                deblur_min_weight=case["deblur_min_weight"],
                deblur_weight=case["deblur_weight"],
                contrast_gain=case["contrast_gain"],
                denoise_sigma=case["denoise_sigma"],
                sharpen_sigma=case["sharpen_sigma"],
                sharpen_strength=case["sharpen_strength"],
                detail_strength=case["detail_strength"],
                edge_detail_strength=case["edge_detail_strength"],
            )
            improved = improved_steps["最终改进复原"]
            improved_metrics = evaluate_image(improved, gt_image)

            improved_images = []
            improved_titles = []
            for step_name, step_image in improved_steps.items():
                improved_images.append(step_image)
                if step_name == "退化图":
                    improved_titles.append(metric_title(f"{case_name}：退化图", degraded_metrics))
                elif step_name == "最终改进复原":
                    improved_titles.append(metric_title(f"{case_name}：最终改进复原", improved_metrics))
                else:
                    improved_titles.append(f"{case_name}：{step_name}")
            improved_images.append(gt_image)
            improved_titles.append("原图 / 标准图")
            save_comparison_figure(
                improved_images,
                improved_titles,
                improved_dir / f"{image_path.stem}_{case['name']}_improved.png",
                max_cols=3,
            )
            comparison_images.extend([degraded, best_traditional_image, improved, gt_image])
            comparison_titles.extend(
                [
                    metric_title(f"{case_name}：退化图", degraded_metrics),
                    metric_title(f"{case_name}：传统复原（{METHOD_NAMES[best_traditional_key]}）", best_traditional_metrics),
                    metric_title(f"{case_name}：改进复原", improved_metrics),
                    f"{case_name}：原图 / 标准图",
                ]
            )
            records.append(
                {
                    "case": case["name"],
                    "image": image_path.stem,
                    "method_key": "improved_restoration",
                    **improved_metrics,
                }
            )

        save_comparison_figure(
            comparison_images,
            comparison_titles,
            comparison_dir / f"{image_path.stem}_comparison.png",
            max_cols=4,
        )

    df = pd.DataFrame(records)
    detail_path = output_root / "surveillance_details.csv"
    case_summary_path = output_root / "surveillance_case_summary.csv"
    summary_path = output_root / "surveillance_summary.csv"

    build_detail_table(df).to_csv(detail_path, index=False, encoding="utf-8-sig")

    method_df = df[df["method_key"] != "degraded"].copy()
    case_summary = (
        method_df.groupby(["case", "method_key"])[["psnr", "ssim", "rmse"]]
        .mean()
        .sort_values(by=["case", "ssim", "psnr"], ascending=[True, False, False])
        .round(4)
    )
    build_case_summary_table(case_summary).to_csv(case_summary_path, index=False, encoding="utf-8-sig")

    summary = (
        method_df.groupby("method_key")[["psnr", "ssim", "rmse"]]
        .mean()
        .sort_values(by=["ssim", "psnr"], ascending=[False, False])
        .round(4)
    )
    build_summary_table(summary).to_csv(summary_path, index=False, encoding="utf-8-sig")
    improved_best20_dir = build_improved_best20(output_root, df)

    return {
        "output_root": output_root,
        "traditional_dir": traditional_dir,
        "improved_dir": improved_dir,
        "comparison_dir": comparison_dir,
        "improved_best20_dir": improved_best20_dir,
        "summary_path": summary_path,
        "case_summary_path": case_summary_path,
        "detail_path": detail_path,
        "count": len(image_paths),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run surveillance image restoration")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_DATASET_DIR, help="Reference image folder")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR, help="LLVIP visible image folder")
    parser.add_argument("--dataset-count", type=int, default=DEFAULT_DATASET_COUNT, help="Number of selected images")
    parser.add_argument("--image-size", type=int, default=DEFAULT_LONG_EDGE, help="Long edge size for selected images")
    parser.add_argument("--skip-prepare-dataset", action="store_true", help="Use input-dir directly without rebuilding it")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Output folder")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset_path = resolve_project_path(args.input_dir)
    if not args.skip_prepare_dataset:
        dataset_path = prepare_reference_dataset(
            source_dir=resolve_project_path(args.source_dir),
            output_dir=dataset_path,
            count=args.dataset_count,
            long_edge=args.image_size,
        )

    report = run_surveillance_demo(dataset_path, resolve_project_path(args.output_root), args.limit)
    print(f"dataset_path: {display_path(dataset_path)}")
    print(f"result_path: {display_path(report['output_root'])}")
    print(f"traditional_path: {display_path(report['traditional_dir'])}")
    print(f"improved_path: {display_path(report['improved_dir'])}")
    print(f"comparison_path: {display_path(report['comparison_dir'])}")
    print(f"improved_best20_path: {display_path(report['improved_best20_dir'])}")


if __name__ == "__main__":
    main()
