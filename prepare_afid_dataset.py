import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image

from config import DATASET_DIR, IMG_SIZE

RAW_ROOT = Path("dataset_raw/ds")
IMAGE_DIR = RAW_ROOT / "img"
ANNOTATION_DIR = RAW_ROOT / "ann"
SEED = 42
TARGET_CLASSES = {
    "broken end": "broken_end",
    "broken pick": "broken_pick",
    "broken yarn": "broken_yarn",
    "contamination": "contamination",
    "crease": "crease",
    "cut selvage": "cut_selvage",
    "fuzzyball": "fuzzyball",
    "knots": "knots",
    "nep": "nep",
    "warp ball": "warp_ball",
    "weft crack": "weft_crack",
    "weft curling": "weft_curling",
}
DEFECT_TRAIN_CROPS = [-144, -112, -80, -48, -24, 0, 24, 48, 80, 112]
DEFECT_EVAL_CROPS = [-96, -48, 0, 48]
CLEAN_TRAIN_CROPS = [0, 256, 512, 768, 1024]
CLEAN_EVAL_CROPS = [0, 512, 1024]


def clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(value, upper))


def crop_strip(image: Image.Image, x: int) -> Image.Image:
    left = clamp(x, 0, image.width - IMG_SIZE)
    return image.crop((left, 0, left + IMG_SIZE, IMG_SIZE))


def split_items(items: list[tuple[Path, int]], rng: random.Random):
    shuffled = items[:]
    rng.shuffle(shuffled)
    train_end = max(1, int(len(shuffled) * 0.7))
    val_end = max(train_end + 1, int(len(shuffled) * 0.85))
    return {
        "train": shuffled[:train_end],
        "validation": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def gather_samples() -> dict[str, list[tuple[Path, int]]]:
    samples: dict[str, list[tuple[Path, int]]] = defaultdict(list)
    for ann_path in ANNOTATION_DIR.glob("*.json"):
        data = json.loads(ann_path.read_text(encoding="utf-8"))
        image_path = IMAGE_DIR / ann_path.name.removesuffix(".json")
        objects = data.get("objects", [])
        if not objects:
            samples["clean"].append((image_path, 0))
            continue
        for obj in objects:
            class_name = TARGET_CLASSES.get(obj["classTitle"])
            if not class_name:
                continue
            origin_x = int(obj["bitmap"]["origin"][0])
            samples[class_name].append((image_path, origin_x))
    return samples


def write_dataset() -> None:
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)
    rng = random.Random(SEED)
    samples = gather_samples()

    for class_name in ["clean", *sorted(TARGET_CLASSES.values())]:
        items = samples.get(class_name, [])
        if not items:
            continue
        if class_name == "clean":
            rng.shuffle(items)
            items = items[:180]
        splits = split_items(items, rng)
        for split_name, split_items_for_class in splits.items():
            out_dir = DATASET_DIR / split_name / class_name
            out_dir.mkdir(parents=True, exist_ok=True)
            for index, (image_path, origin_x) in enumerate(split_items_for_class):
                with Image.open(image_path) as image:
                    if class_name == "clean":
                        crop_positions = (
                            CLEAN_TRAIN_CROPS
                            if split_name == "train"
                            else CLEAN_EVAL_CROPS
                        )
                    else:
                        offsets = (
                            DEFECT_TRAIN_CROPS
                            if split_name == "train"
                            else DEFECT_EVAL_CROPS
                        )
                        crop_positions = [origin_x + offset for offset in offsets]
                    for crop_index, crop_x in enumerate(crop_positions):
                        crop = crop_strip(image.convert("RGB"), crop_x)
                        filename = f"{image_path.stem}_{index}_{crop_index}.png"
                        crop.save(out_dir / filename)


if __name__ == "__main__":
    write_dataset()
