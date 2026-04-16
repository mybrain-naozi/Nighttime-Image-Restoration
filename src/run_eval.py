from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.dataset import ImagePair, limit_pairs, scan_pairs
from src.degradation import defocus_kernel, motion_kernel, synthetic_degradation
from src.metrics import evaluate_image
from src.methods.deconv_filters import (
    constrained_least_squares_filter,
    inverse_filter,
    wiener_filter,
)
from src.methods.enhancement import clahe_enhance, gamma_correction, retinex_bilateral, single_scale_retinex
from src.methods.improved_method import adaptive_night_restoration, guided_night_restoration
from src.methods.spatial_filters import gaussian_filter, mean_filter, median_filter
from src.utils import ensure_dir, read_image, reset_dir, save_comparison_figure


def _metric_title(name: str, metrics: dict[str, float] | None) -> str:
    if metrics is None:
        return name
    return f"{name}\nPSNR {metrics['psnr']:.2f} | SSIM {metrics['ssim']:.3f} | RMSE {metrics['rmse']:.3f}"


def _save_real_visualization(
    pair: ImagePair,
    low_image,
    gt_image,
    predictions: OrderedDict[str, object],
    output_dir: Path,
) -> None:
    images = [low_image]
    titles = [_metric_title("Input Low", evaluate_image(low_image, gt_image))]
    for method_name, restored in predictions.items():
        images.append(restored)
        titles.append(_metric_title(method_name, evaluate_image(restored, gt_image)))
    images.append(gt_image)
    titles.append("Ground Truth")
    save_comparison_figure(images, titles, output_dir / f"{pair.name}.png")


def _save_synthetic_visualization(
    image_name: str,
    case_name: str,
    degraded_image,
    gt_image,
    predictions: OrderedDict[str, object],
    output_dir: Path,
) -> None:
    images = [degraded_image]
    titles = [_metric_title("Degraded Input", evaluate_image(degraded_image, gt_image))]
    for method_name, restored in predictions.items():
        images.append(restored)
        titles.append(_metric_title(method_name, evaluate_image(restored, gt_image)))
    images.append(gt_image)
    titles.append("Ground Truth")
    save_comparison_figure(images, titles, output_dir / f"{image_name}_{case_name}.png")


def run_real_experiment(
    dataset_root: Path,
    split: str,
    output_root: Path,
    limit: int | None,
) -> dict[str, object]:
    pairs = limit_pairs(scan_pairs(dataset_root / split), limit)
    figures_root = ensure_dir(output_root / "06_真实低照度对比图")

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

    for pair in tqdm(pairs, desc=f"Real low-light ({split})"):
        low_image = read_image(pair.low_path)
        gt_image = read_image(pair.high_path)

        predictions: OrderedDict[str, object] = OrderedDict()
        low_metrics = evaluate_image(low_image, gt_image)
        records.append(
            {
                "experiment": "real",
                "split": split,
                "image": pair.name,
                "method": "input_low",
                **low_metrics,
            }
        )

        for method_name, method in methods.items():
            restored = method(low_image)
            predictions[method_name] = restored
            metrics = evaluate_image(restored, gt_image)
            records.append(
                {
                    "experiment": "real",
                    "split": split,
                    "image": pair.name,
                    "method": method_name,
                    **metrics,
                }
            )

        _save_real_visualization(pair, low_image, gt_image, predictions, figures_root)

    df = pd.DataFrame(records)
    details_path = output_root / "04_真实低照度逐图结果.csv"
    summary_path = output_root / "01_真实低照度汇总.csv"
    df.to_csv(details_path, index=False, encoding="utf-8-sig")
    summary = (
        df.groupby("method")[["psnr", "ssim", "rmse"]]
        .mean()
        .sort_values(by=["ssim", "psnr"], ascending=[False, False])
        .round(4)
    )
    summary.to_csv(summary_path, encoding="utf-8-sig")
    return {
        "summary_path": summary_path,
        "details_path": details_path,
        "figures_dir": figures_root,
        "best_method": str(summary.index[0]),
        "best_psnr": float(summary.iloc[0]["psnr"]),
        "best_ssim": float(summary.iloc[0]["ssim"]),
        "best_rmse": float(summary.iloc[0]["rmse"]),
        "figure_count": len(pairs),
    }


def run_synthetic_experiment(
    dataset_root: Path,
    split: str,
    output_root: Path,
    limit: int | None,
) -> dict[str, object]:
    pairs = limit_pairs(scan_pairs(dataset_root / split), limit)
    figures_root = ensure_dir(output_root / "07_人工退化对比图")

    synthetic_cases = [
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

    for pair in tqdm(pairs, desc=f"Synthetic restoration ({split})"):
        gt_image = read_image(pair.high_path)

        for case in synthetic_cases:
            degraded_image = synthetic_degradation(
                gt_image,
                kernel=case["kernel"],
                noise_sigma=case["noise_sigma"],
                salt_pepper_amount=case["salt_pepper"],
                gamma=case["gamma"],
                gain=case["gain"],
            )
            case_name = case["name"]

            degraded_metrics = evaluate_image(degraded_image, gt_image)
            records.append(
                {
                    "experiment": "synthetic",
                    "split": split,
                    "case": case_name,
                    "image": pair.name,
                    "method": "degraded_input",
                    **degraded_metrics,
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
                    (
                        "improved_guided",
                        lambda image, kernel=case["kernel"]: guided_night_restoration(image, kernel),
                    ),
                ]
            )

            predictions: OrderedDict[str, object] = OrderedDict()
            for method_name, method in methods.items():
                restored = method(degraded_image)
                predictions[method_name] = restored
                metrics = evaluate_image(restored, gt_image)
                records.append(
                    {
                        "experiment": "synthetic",
                        "split": split,
                        "case": case_name,
                        "image": pair.name,
                        "method": method_name,
                        **metrics,
                    }
                )

            _save_synthetic_visualization(pair.name, case_name, degraded_image, gt_image, predictions, figures_root)

    df = pd.DataFrame(records)
    details_path = output_root / "05_人工退化逐图结果.csv"
    summary_by_case_path = output_root / "03_人工退化分类汇总.csv"
    summary_overall_path = output_root / "02_人工退化汇总.csv"
    df.to_csv(details_path, index=False, encoding="utf-8-sig")

    summary_by_case = (
        df.groupby(["case", "method"])[["psnr", "ssim", "rmse"]]
        .mean()
        .sort_values(by=["case", "ssim", "psnr"], ascending=[True, False, False])
        .round(4)
    )
    summary_by_case.to_csv(summary_by_case_path, encoding="utf-8-sig")

    summary_overall = (
        df.groupby("method")[["psnr", "ssim", "rmse"]]
        .mean()
        .sort_values(by=["ssim", "psnr"], ascending=[False, False])
        .round(4)
    )
    summary_overall.to_csv(summary_overall_path, encoding="utf-8-sig")
    return {
        "summary_path": summary_overall_path,
        "summary_by_case_path": summary_by_case_path,
        "details_path": details_path,
        "figures_dir": figures_root,
        "best_method": str(summary_overall.index[0]),
        "best_psnr": float(summary_overall.iloc[0]["psnr"]),
        "best_ssim": float(summary_overall.iloc[0]["ssim"]),
        "best_rmse": float(summary_overall.iloc[0]["rmse"]),
        "figure_count": len(pairs) * len(synthetic_cases),
    }


def write_simple_guide(
    output_root: Path,
    split: str,
    mode: str,
    real_report: dict[str, object] | None,
    synthetic_report: dict[str, object] | None,
) -> None:
    lines = [
        "先看这个文件",
        "",
        "建议按下面顺序看结果：",
        "1. 先看这个文件，知道结果分别放在哪。",
    ]

    if real_report is not None:
        lines.extend(
            [
                "2. 看 01_真实低照度汇总.csv。",
                "3. 看 06_真实低照度对比图 文件夹里的图片。",
                "",
                "真实低照度实验：",
                f"- 最优方法：{real_report['best_method']}",
                f"- 平均指标：PSNR={real_report['best_psnr']:.4f}，SSIM={real_report['best_ssim']:.4f}，RMSE={real_report['best_rmse']:.4f}",
                f"- 对比图数量：{real_report['figure_count']}",
            ]
        )

    if synthetic_report is not None:
        lines.extend(
            [
                "",
                "人工退化复原实验：",
                "4. 看 02_人工退化汇总.csv。",
                "5. 如果想分开看运动模糊和散焦模糊，再看 03_人工退化分类汇总.csv。",
                "6. 看 07_人工退化对比图 文件夹里的图片。",
                f"- 最优方法：{synthetic_report['best_method']}",
                f"- 平均指标：PSNR={synthetic_report['best_psnr']:.4f}，SSIM={synthetic_report['best_ssim']:.4f}，RMSE={synthetic_report['best_rmse']:.4f}",
                f"- 对比图数量：{synthetic_report['figure_count']}",
            ]
        )

    lines.extend(
        [
            "",
            "如果你只想看最终结论：",
            "- 直接看 01_真实低照度汇总.csv 和 02_人工退化汇总.csv。",
            "如果你想看图片效果：",
            "- 直接看 06_真实低照度对比图 和 07_人工退化对比图。",
            "04_真实低照度逐图结果.csv 和 05_人工退化逐图结果.csv 是详细明细，后面写论文时再看就行。",
            "",
            f"本次运行模式：{mode}",
            f"本次使用数据集：{split}",
        ]
    )

    guide_path = output_root / "00_先看这个.txt"
    guide_path.write_text("\n".join(lines), encoding="utf-8-sig")


def print_run_summary(
    output_root: Path,
    real_report: dict[str, object] | None,
    synthetic_report: dict[str, object] | None,
) -> None:
    print("\n运行完成。")
    print(f"结果目录：{output_root.resolve()}")
    print(f"先看这里：{(output_root / '00_先看这个.txt').resolve()}")
    if real_report is not None:
        print(f"真实低照度汇总：{real_report['summary_path'].resolve()}")
        print(f"真实低照度对比图：{real_report['figures_dir'].resolve()}")
    if synthetic_report is not None:
        print(f"人工退化汇总：{synthetic_report['summary_path'].resolve()}")
        print(f"人工退化分类汇总：{synthetic_report['summary_by_case_path'].resolve()}")
        print(f"人工退化对比图：{synthetic_report['figures_dir'].resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Night image restoration experiments")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("."),
        help="Project root that contains our485/ and eval15/",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="eval15",
        choices=["our485", "eval15"],
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "real", "synthetic"],
        help="Which experiment branch to run.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results"),
        help="Where to save metrics, restored images, and figures.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of image pairs to process.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_root = reset_dir(args.output_root)
    real_report = None
    synthetic_report = None
    if args.mode in {"all", "real"}:
        real_report = run_real_experiment(args.data_root, args.split, output_root, args.limit)
    if args.mode in {"all", "synthetic"}:
        synthetic_report = run_synthetic_experiment(args.data_root, args.split, output_root, args.limit)
    write_simple_guide(output_root, args.split, args.mode, real_report, synthetic_report)
    print_run_summary(output_root, real_report, synthetic_report)


if __name__ == "__main__":
    main()
