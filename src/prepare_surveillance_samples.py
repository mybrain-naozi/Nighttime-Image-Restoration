from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.utils import reset_dir

from src.path_config import (
    LLVIP_VISIBLE_DIR as DEFAULT_SOURCE_DIR,
    SURVEILLANCE_SAMPLE_DIR as DEFAULT_OUTPUT_DIR,
    display_path,
    path_help_message,
    resolve_project_path,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_SAMPLE_COUNT = 20


def _quick_brightness(image_path: Path) -> float:
    image_data = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(image_data, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")
    if max(image.shape[:2]) > 160:
        scale = 160.0 / max(image.shape[:2])
        image = cv2.resize(image, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return float(image.mean() / 255.0)


def prepare_surveillance_samples(
    source_dir: str | Path = DEFAULT_SOURCE_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
) -> dict[str, Path | int]:
    source_dir = resolve_project_path(source_dir)
    output_dir = reset_dir(resolve_project_path(output_dir))

    image_paths = sorted(
        [
            path
            for path in source_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=lambda path: str(path.relative_to(source_dir)).replace("\\", "/"),
    )[:sample_count]

    if not image_paths:
        raise FileNotFoundError(
            f"No images found in source folder: {source_dir}\n"
            f"{path_help_message()}"
        )

    rows: list[dict[str, object]] = []
    for image_path in image_paths:
        image_name = image_path.name
        target_path = output_dir / image_name
        target_path.write_bytes(image_path.read_bytes())
        rows.append({"image": image_name, "brightness": round(_quick_brightness(image_path), 4)})

    sample_list_path = output_dir / "sample_list.csv"
    pd.DataFrame(rows).to_csv(sample_list_path, index=False, encoding="utf-8-sig")

    return {
        "output_dir": output_dir,
        "sample_list_path": sample_list_path,
        "count": len(rows),
    }


def main() -> None:
    report = prepare_surveillance_samples()
    print(f"sample_path: {display_path(report['output_dir'])}")


if __name__ == "__main__":
    main()
