from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path

import pandas as pd

from src.dataset import ImagePair, limit_pairs, scan_pairs
from src.degradation import defocus_kernel, motion_kernel, synthetic_degradation
from src.metrics import evaluate_image
from src.methods.deconv_filters import (
    constrained_least_squares_filter,
    inverse_filter,
    wiener_filter,
)
from src.methods.enhancement import (
    clahe_enhance,
    gamma_correction,
    retinex_bilateral,
    single_scale_retinex,
)
from src.methods.improved_method import adaptive_night_restoration, guided_night_restoration
from src.methods.spatial_filters import gaussian_filter, mean_filter, median_filter
from src.utils import ensure_dir, read_image, reset_dir, save_comparison_figure


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def display_path(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


METHOD_NAMES = {
    "input_low": "Original",
    "gamma": "Gamma",
    "clahe": "CLAHE",
    "retinex": "Retinex",
    "retinex_bilateral": "Retinex+Bilateral",
    "improved": "Improved",
    "degraded_input": "Degraded",
    "mean_filter": "MeanFilter",
    "gaussian_filter": "GaussianFilter",
    "median_filter": "MedianFilter",
    "inverse_filter": "InverseFilter",
    "wiener_filter": "WienerFilter",
    "cls_filter": "CLSFilter",
    "improved_guided": "Improved",
}

CASE_NAMES = {
    "motion": "MotionBlur",
    "defocus": "DefocusBlur",
}

EXPERIMENT_NAMES = {
    "real": "RealLowLight",
    "synthetic": "SyntheticDegradation",
}


def metric_title(title: str, metrics: dict[str, float] | None) -> str:
    if metrics is None:
        return title
    return f"{title}\nPSNR {metrics['psnr']:.2f} | SSIM {metrics['ssim']:.3f} | RMSE {metrics['rmse']:.3f}"


def build_detail_table(df: pd.DataFrame) -> pd.DataFrame:
    table = df.copy()
    table["experiment_name"] = table["experiment"].map(EXPERIMENT_NAMES)
    table["method_name"] = table["method_key"].map(METHOD_NAMES)
    table["case_name"] = table["case"].fillna("").map(lambda value: CASE_NAMES.get(value, "") if value else "")
    return table[
        ["experiment_name", "split", "case_name", "image", "method_name", "psnr", "ssim", "rmse"]
    ].rename(
        columns={
            "experiment_name": "experiment",
            "split": "split",
            "case_name": "case",
            "image": "image",
            "method_name": "method",
            "psnr": "PSNR",
            "ssim": "SSIM",
            "rmse": "RMSE",
        }
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


def save_real_visualization(
    pair: ImagePair,
    low_image,
    gt_image,
    predictions: OrderedDict[str, object],
    output_dir: Path,
) -> None:
    images = [low_image]
    titles = [metric_title("Original", evaluate_image(low_image, gt_image))]
    for method_key, restored in predictions.items():
        images.append(restored)
        titles.append(metric_title(METHOD_NAMES[method_key], evaluate_image(restored, gt_image)))
    images.append(gt_image)
    titles.append("GroundTruth")
    save_comparison_figure(images, titles, output_dir / f"{pair.name}.png")


def save_synthetic_visualization(
    image_name: str,
    case_name: str,
    degraded_image,
    gt_image,
    predictions: OrderedDict[str, object],
    output_dir: Path,
) -> None:
    images = [degraded_image]
    titles = [metric_title("Degraded", evaluate_image(degraded_image, gt_image))]
    for method_key, restored in predictions.items():
        images.append(restored)
        titles.append(metric_title(METHOD_NAMES[method_key], evaluate_image(restored, gt_image)))
    images.append(gt_image)
    titles.append("GroundTruth")
    save_comparison_figure(images, titles, output_dir / f"{image_name}_{case_name}.png")


def run_real_experiment(dataset_root: Path, split: str, output_root: Path, limit: int | None) -> dict[str, Path]:
    pairs = limit_pairs(scan_pairs(dataset_root / split), limit)
    figures_dir = ensure_dir(output_root / "real_figures")

    methods = OrderedDict(
        [
            ("gamma", lambda image: gamma_correction(image, gamma=0.65)),
            ("clahe", clahe_enhance),
            ("retinex", single_scale_retinex),
            ("retinex_bilateral", retinex_bilateral),
            ("improved", adaptive_night_restoration),
        ]
    )

    records: list[dict[str, object]] = []
    for pair in pairs:
        low_image = read_image(pair.low_path)
        gt_image = read_image(pair.high_path)

        predictions: OrderedDict[str, object] = OrderedDict()
        records.append(
            {
                "experiment": "real",
                "split": split,
                "case": "",
                "image": pair.name,
                "method_key": "input_low",
                **evaluate_image(low_image, gt_image),
            }
        )

        for method_key, method in methods.items():
            restored = method(low_image)
            predictions[method_key] = restored
            records.append(
                {
                    "experiment": "real",
                    "split": split,
                    "case": "",
                    "image": pair.name,
                    "method_key": method_key,
                    **evaluate_image(restored, gt_image),
                }
            )

        save_real_visualization(pair, low_image, gt_image, predictions, figures_dir)

    df = pd.DataFrame(records)
    detail_path = output_root / "real_details.csv"
    summary_path = output_root / "real_summary.csv"
    build_detail_table(df).to_csv(detail_path, index=False, encoding="utf-8-sig")

    summary = (
        df.groupby("method_key")[["psnr", "ssim", "rmse"]]
        .mean()
        .sort_values(by=["ssim", "psnr"], ascending=[False, False])
        .round(4)
    )
    build_summary_table(summary).to_csv(summary_path, index=False, encoding="utf-8-sig")
    return {"summary_path": summary_path, "figures_dir": figures_dir}


def run_synthetic_experiment(dataset_root: Path, split: str, output_root: Path, limit: int | None) -> dict[str, Path]:
    pairs = limit_pairs(scan_pairs(dataset_root / split), limit)
    figures_dir = ensure_dir(output_root / "synthetic_figures")
    cases = [
        {
            "name": "motion",
            "kernel": motion_kernel(length=21, angle=20.0),
            "noise_sigma": 0.012,
            "salt_pepper": 0.0,
            "gamma": 1.7,
            "gain": 0.82,
        },
        {
            "name": "defocus",
            "kernel": defocus_kernel(radius=5),
            "noise_sigma": 0.010,
            "salt_pepper": 0.001,
            "gamma": 1.6,
            "gain": 0.84,
        },
    ]

    records: list[dict[str, object]] = []
    for pair in pairs:
        gt_image = read_image(pair.high_path)
        for case in cases:
            degraded_image = synthetic_degradation(
                gt_image,
                kernel=case["kernel"],
                noise_sigma=case["noise_sigma"],
                salt_pepper_amount=case["salt_pepper"],
                gamma=case["gamma"],
                gain=case["gain"],
            )

            records.append(
                {
                    "experiment": "synthetic",
                    "split": split,
                    "case": case["name"],
                    "image": pair.name,
                    "method_key": "degraded_input",
                    **evaluate_image(degraded_image, gt_image),
                }
            )

            methods = OrderedDict(
                [
                    ("mean_filter", lambda image: mean_filter(image, ksize=5)),
                    ("gaussian_filter", lambda image: gaussian_filter(image, ksize=5, sigma=1.2)),
                    ("median_filter", lambda image: median_filter(image, ksize=5)),
                    ("inverse_filter", lambda image, kernel=case["kernel"]: inverse_filter(image, kernel, eps=0.004)),
                    ("wiener_filter", lambda image, kernel=case["kernel"]: wiener_filter(image, kernel, k=0.012)),
                    (
                        "cls_filter",
                        lambda image, kernel=case["kernel"]: constrained_least_squares_filter(
                            image,
                            kernel,
                            gamma=0.0025,
                        ),
                    ),
                    ("improved_guided", lambda image, kernel=case["kernel"]: guided_night_restoration(image, kernel)),
                ]
            )

            predictions: OrderedDict[str, object] = OrderedDict()
            for method_key, method in methods.items():
                restored = method(degraded_image)
                predictions[method_key] = restored
                records.append(
                    {
                        "experiment": "synthetic",
                        "split": split,
                        "case": case["name"],
                        "image": pair.name,
                        "method_key": method_key,
                        **evaluate_image(restored, gt_image),
                    }
                )

            save_synthetic_visualization(pair.name, case["name"], degraded_image, gt_image, predictions, figures_dir)

    df = pd.DataFrame(records)
    detail_path = output_root / "synthetic_details.csv"
    case_summary_path = output_root / "synthetic_case_summary.csv"
    summary_path = output_root / "synthetic_summary.csv"
    build_detail_table(df).to_csv(detail_path, index=False, encoding="utf-8-sig")

    case_summary = (
        df.groupby(["case", "method_key"])[["psnr", "ssim", "rmse"]]
        .mean()
        .sort_values(by=["case", "ssim", "psnr"], ascending=[True, False, False])
        .round(4)
    )
    build_case_summary_table(case_summary).to_csv(case_summary_path, index=False, encoding="utf-8-sig")

    summary = (
        df.groupby("method_key")[["psnr", "ssim", "rmse"]]
        .mean()
        .sort_values(by=["ssim", "psnr"], ascending=[False, False])
        .round(4)
    )
    build_summary_table(summary).to_csv(summary_path, index=False, encoding="utf-8-sig")
    return {
        "summary_path": summary_path,
        "case_summary_path": case_summary_path,
        "figures_dir": figures_dir,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run low-light experiments")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT, help="Data root")
    parser.add_argument("--split", type=str, default="eval15", choices=["our485", "eval15"], help="Dataset split")
    parser.add_argument("--mode", type=str, default="all", choices=["all", "real", "synthetic"], help="Run mode")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "results", help="Output folder")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data_root = resolve_project_path(args.data_root)
    output_root = reset_dir(resolve_project_path(args.output_root))
    real_report = None
    synthetic_report = None

    if args.mode in {"all", "real"}:
        real_report = run_real_experiment(data_root, args.split, output_root, args.limit)
    if args.mode in {"all", "synthetic"}:
        synthetic_report = run_synthetic_experiment(data_root, args.split, output_root, args.limit)

    print(f"result_path: {display_path(output_root)}")
    if real_report is not None:
        print(f"real_summary: {display_path(real_report['summary_path'])}")
        print(f"real_figures: {display_path(real_report['figures_dir'])}")
    if synthetic_report is not None:
        print(f"synthetic_summary: {display_path(synthetic_report['summary_path'])}")
        print(f"synthetic_figures: {display_path(synthetic_report['figures_dir'])}")


if __name__ == "__main__":
    main()
