from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class ImagePair:
    name: str
    low_path: Path
    high_path: Path


def _natural_key(text: str) -> List[object]:
    key: List[object] = []
    current = ""
    is_digit = text[:1].isdigit()
    for char in text:
        if char.isdigit() == is_digit:
            current += char
            continue
        key.append(int(current) if is_digit else current.lower())
        current = char
        is_digit = char.isdigit()
    if current:
        key.append(int(current) if is_digit else current.lower())
    return key


def _image_map(folder: Path) -> dict[str, Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Missing image folder: {folder}")
    return {
        file_path.stem: file_path
        for file_path in folder.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS
    }


def scan_pairs(split_dir: str | Path) -> list[ImagePair]:
    split_dir = Path(split_dir)
    low_map = _image_map(split_dir / "low")
    high_map = _image_map(split_dir / "high")

    common_names = sorted(set(low_map) & set(high_map), key=_natural_key)
    if not common_names:
        raise RuntimeError(f"No paired images found in {split_dir}")

    return [
        ImagePair(name=name, low_path=low_map[name], high_path=high_map[name])
        for name in common_names
    ]


def limit_pairs(pairs: Iterable[ImagePair], limit: int | None) -> list[ImagePair]:
    pair_list = list(pairs)
    return pair_list if limit is None else pair_list[:limit]
