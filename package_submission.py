"""Zip this project into a submission archive.

Run once the five required screenshots exist in screenshots/. Not part of the
submission itself: excluded from its own zip.

Usage: python package_submission.py
"""

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "submission.zip"

EXCLUDE_NAMES = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    "submission.zip",
    "package_submission.py",
    "model_card_template.md",
    "sanitycheck.py",
    ".gitkeep",
}

REQUIRED_SCREENSHOTS = {
    "continuous_integration.png",
    "example.png",
    "continuous_deloyment.png",
    "continuous_deployment.png",
    "live_get.png",
    "live_post.png",
}


def should_include(path):
    for part in path.relative_to(ROOT).parts:
        if part in EXCLUDE_NAMES:
            return False
    return True


def check_screenshots():
    screenshots_dir = ROOT / "screenshots"
    present = {p.name for p in screenshots_dir.glob("*.png")}
    missing = REQUIRED_SCREENSHOTS - present
    if missing:
        print("Missing screenshots, fix these before packaging:")
        for name in sorted(missing):
            print(f"  - {name}")
        return False
    return True


def build_zip():
    files = [p for p in ROOT.rglob("*") if p.is_file() and should_include(p)]
    with zipfile.ZipFile(OUTPUT, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(path, path.relative_to(ROOT))
    return files


if __name__ == "__main__":
    if not check_screenshots():
        raise SystemExit(1)

    files = build_zip()
    print(f"Wrote {OUTPUT} with {len(files)} files:")
    for path in sorted(files):
        print(f"  {path.relative_to(ROOT)}")
