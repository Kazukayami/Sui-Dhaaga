from collections import defaultdict
from pathlib import Path

from config import DATASET_DIR
from dataset_tools import file_hash, iter_images


def remove_leaked_eval_duplicates() -> dict:
    train_hashes = set()
    for image_path in iter_images(DATASET_DIR / "train"):
        train_hashes.add(file_hash(image_path))

    removed = defaultdict(int)
    seen_eval_hashes = set()
    for split in ["validation", "test"]:
        split_dir = DATASET_DIR / split
        if not split_dir.exists():
            continue
        for image_path in list(iter_images(split_dir)):
            digest = file_hash(image_path)
            if digest in train_hashes or digest in seen_eval_hashes:
                image_path.unlink()
                removed[split] += 1
            else:
                seen_eval_hashes.add(digest)
    return dict(removed)


if __name__ == "__main__":
    removed_counts = remove_leaked_eval_duplicates()
    if not removed_counts:
        print("No leaked validation/test duplicates found.")
    else:
        for split, count in sorted(removed_counts.items()):
            print(f"{split}: removed {count} leaked duplicate files")
