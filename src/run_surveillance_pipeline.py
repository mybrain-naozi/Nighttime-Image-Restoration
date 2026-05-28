from __future__ import annotations

import argparse
from pathlib import Path

from src.prepare_surveillance_samples import DEFAULT_OUTPUT_DIR, DEFAULT_SOURCE_DIR, prepare_surveillance_samples
from src.path_config import SURVEILLANCE_RESULT_DIR as DEFAULT_OUTPUT_ROOT, display_path, path_help_message, resolve_project_path
from src.run_surveillance_demo import run_surveillance_demo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare samples and run surveillance restoration")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR, help="LLVIP image folder")
    parser.add_argument("--sample-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Sample folder")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Result folder")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of sampled images for restoration")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source_dir = resolve_project_path(args.source_dir)
    sample_dir = resolve_project_path(args.sample_dir)
    output_root = resolve_project_path(args.output_root)
    if not source_dir.exists():
        raise FileNotFoundError(f"source_dir not found: {source_dir}\n{path_help_message()}")
    sample_report = prepare_surveillance_samples(source_dir, sample_dir)
    demo_report = run_surveillance_demo(sample_dir, output_root, args.limit)

    print(f"sample_path: {display_path(sample_report['output_dir'])}")
    print(f"result_path: {display_path(demo_report['output_root'])}")
    print(f"traditional_path: {display_path(demo_report['traditional_dir'])}")
    print(f"improved_path: {display_path(demo_report['improved_dir'])}")


if __name__ == "__main__":
    main()
