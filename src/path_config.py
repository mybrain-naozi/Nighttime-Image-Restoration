from __future__ import annotations

from pathlib import Path


# ============================================================
# Path configuration
# ============================================================
# If another user runs this project on a different computer, this is the
# first file they should check.
#
# All paths below support two formats:
# 1. Relative path: relative to the project root, e.g. "LLVIP/LLVIP/visible"
# 2. Absolute path: e.g. r"D:/datasets/LLVIP/LLVIP/visible"
#
# Recommended setup:
# Put the LLVIP folder directly inside this project folder. If you do that,
# the default LLVIP_VISIBLE_DIR below does not need to be changed.
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# >>> CHECK THIS FIRST: LLVIP visible image folder.
# Expected default structure:
#
# project_root/
#   LLVIP/
#     LLVIP/
#       visible/
#         200099.jpg
#         ...
#
# If the dataset is stored somewhere else, change only this line.
# Example:
# LLVIP_VISIBLE_DIR = Path(r"D:/datasets/LLVIP/LLVIP/visible")
LLVIP_VISIBLE_DIR = PROJECT_ROOT / "LLVIP" / "LLVIP" / "visible"


# >>> Optional: selected sample image folder.
# Usually no change is needed. The program creates this folder automatically.
SURVEILLANCE_SAMPLE_DIR = PROJECT_ROOT / "surveillance_samples"


# >>> Optional: final result folder.
# Usually no change is needed. The program creates this folder automatically.
SURVEILLANCE_RESULT_DIR = PROJECT_ROOT / "surveillance_results"


# >>> Optional: reference dataset folder used by run_surveillance_demo.py.
FINAL_DATASET_DIR = PROJECT_ROOT / "final_dataset" / "original"


# >>> Optional: number of selected surveillance images.
# Default is 100 for the thesis experiment. Use a smaller number for quick tests.
DEFAULT_DATASET_COUNT = 100


def resolve_project_path(path: str | Path) -> Path:
    """Resolve relative paths from the project root; keep absolute paths unchanged."""
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def display_path(path: str | Path) -> str:
    """Prefer short project-relative paths when printing output folders."""
    path = Path(path)
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def path_help_message() -> str:
    return (
        "Path error. Open src/path_config.py and check "
        "LLVIP_VISIBLE_DIR / SURVEILLANCE_SAMPLE_DIR / SURVEILLANCE_RESULT_DIR."
    )
