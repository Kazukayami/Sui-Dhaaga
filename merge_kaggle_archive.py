import argparse
import hashlib
import random
import re
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

from config import DATASET_DIR, SEED

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

KAGGLE_CLASS_MAP = {
    "defect free": "clean",
    "stain": "contamination",
    "hole": "hole",
    "lines": "line_defect",
    "horizontal": "horizontal_defect",
    "vertical": "vertical_defect",
}


def image_entries(zip_file: ZipFile):
    for entry in zip_file.infolist():
        if entry.is_dir():
            continue
        path = Path(entry.filename)
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        parts = entry.filename.replace("\\", "/").split("/")
        if len(parts) < 3:
            continue
        raw_class = parts[-2].strip().lower()
        class_name = KAGGLE_CLASS_MAP.get(raw_class)
        if class_name:
            yield entry, class_name


def family_key(entry_name: str) -> str:
    stem = Path(entry_name).stem.lower()
    stem = re.sub(r"_processed\s*\(?\d*\)?", "", stem)
    stem = re.sub(r"\s*processed\s*\(?\d*\)?", "", stem)
    stem = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    return stem or Path(entry_name).stem.lower()


def split_groups(groups: list[list], train_ratio: float, val_ratio: float) -> dict[str, list]:
    shuffled = list(groups)
    random.Random(SEED).shuffle(shuffled)
    train_end = int(len(shuffled) * train_ratio)
    val_end = train_end + int(len(shuffled) * val_ratio)
    return {
        "train": shuffled[:train_end],
        "validation": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def merge_archive(zip_path: Path, train_ratio: float = 0.70, val_ratio: float = 0.15) -> dict:
    summary = defaultdict(lambda: defaultdict(int))
    grouped = defaultdict(lambda: defaultdict(list))

    with ZipFile(zip_path) as zip_file:
        for entry, class_name in image_entries(zip_file):
            grouped[class_name][family_key(entry.filename)].append(entry)

        for class_name, families in grouped.items():
            split_map = split_groups(list(families.values()), train_ratio, val_ratio)
            for split, family_groups in split_map.items():
                out_dir = DATASET_DIR / split / class_name
                out_dir.mkdir(parents=True, exist_ok=True)
                for family in family_groups:
                    for entry in family:
                        digest = hashlib.sha256(entry.filename.encode("utf-8")).hexdigest()[:16]
                        target = out_dir / f"kaggle_{digest}_{Path(entry.filename).name}"
                        if target.exists():
                            continue
                        with zip_file.open(entry) as source, target.open("wb") as dest:
                            dest.write(source.read())
                        summary[split][class_name] += 1

    return {split: dict(counts) for split, counts in summary.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge the Kaggle fabric defects archive into this project.")
    parser.add_argument("zip_path", type=Path)
    args = parser.parse_args()
    summary = merge_archive(args.zip_path)
    for split, counts in sorted(summary.items()):
        print(split)
        for class_name, count in sorted(counts.items()):
            print(f"  {class_name}: {count}")


if __name__ == "__main__":
    main()
