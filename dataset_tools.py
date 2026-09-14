import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

import pandas as pd

from config import CLASS_ALIASES, DATASET_AUDIT_PATH, DATASET_DIR, REPORTS_DIR
from utils import ensure_project_dirs

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def standardize_class_name(name: str) -> str:
    normalized = name.strip().lower().replace("-", "_").replace(" ", "_")
    alias_key = normalized.replace("_", " ")
    return CLASS_ALIASES.get(normalized, CLASS_ALIASES.get(alias_key, normalized))


def file_hash(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def iter_images(root: Path):
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def audit_dataset(dataset_dir: Path = DATASET_DIR) -> dict:
    split_counts = {}
    hashes: dict[str, list[dict]] = defaultdict(list)
    class_names = set()

    for split_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
        class_counts = {}
        for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            standardized = standardize_class_name(class_dir.name)
            class_names.add(standardized)
            images = list(iter_images(class_dir))
            class_counts[standardized] = class_counts.get(standardized, 0) + len(images)
            for image_path in images:
                hashes[file_hash(image_path)].append(
                    {
                        "split": split_dir.name,
                        "class_name": standardized,
                        "path": str(image_path),
                    }
                )
        split_counts[split_dir.name] = class_counts

    duplicate_groups = [items for items in hashes.values() if len(items) > 1]
    leakage_groups = [
        items
        for items in duplicate_groups
        if len({item["split"] for item in items}) > 1
    ]
    label_conflict_groups = [
        items
        for items in duplicate_groups
        if len({item["class_name"] for item in items}) > 1
    ]

    payload = {
        "class_names": sorted(class_names),
        "split_counts": split_counts,
        "duplicate_group_count": len(duplicate_groups),
        "leakage_group_count": len(leakage_groups),
        "label_conflict_group_count": len(label_conflict_groups),
        "leakage_examples": leakage_groups[:20],
        "label_conflict_examples": label_conflict_groups[:20],
    }
    ensure_project_dirs()
    DATASET_AUDIT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    rows = []
    for split, counts in split_counts.items():
        for class_name, count in counts.items():
            rows.append({"split": split, "class_name": class_name, "count": count})
    pd.DataFrame(rows).to_csv(REPORTS_DIR / "dataset_distribution_audit.csv", index=False)
    return payload


def merge_folder_dataset(source_root: Path, target_dataset: Path, split: str = "train") -> None:
    """Merge a folder-classification dataset into the project dataset.

    Expected source format:
        source_root/class_name/image.ext

    Class folder names are standardized through CLASS_ALIASES. File names are
    prefixed by a content hash to avoid accidental overwrites.
    """
    for class_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
        class_name = standardize_class_name(class_dir.name)
        out_dir = target_dataset / split / class_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for image_path in iter_images(class_dir):
            digest = file_hash(image_path)[:16]
            target_path = out_dir / f"{digest}_{image_path.name}"
            if not target_path.exists():
                shutil.copy2(image_path, target_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and merge textile datasets.")
    parser.add_argument("--audit", action="store_true", help="Audit the current project dataset.")
    parser.add_argument("--merge-root", type=Path, help="Folder dataset root to merge.")
    parser.add_argument("--split", default="train", choices=["train", "validation", "test"])
    args = parser.parse_args()

    if args.merge_root:
        merge_folder_dataset(args.merge_root, DATASET_DIR, args.split)
    if args.audit or not args.merge_root:
        payload = audit_dataset(DATASET_DIR)
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
