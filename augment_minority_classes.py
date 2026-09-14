import argparse
import random
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from config import SEED, TRAIN_DIR

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def image_files(path: Path) -> list[Path]:
    return sorted(
        file
        for file in path.iterdir()
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
        and "_aug_" not in file.stem
    )


def augment_image(image: Image.Image, rng: random.Random) -> Image.Image:
    image = image.convert("RGB")

    if rng.random() < 0.5:
        image = ImageOps.mirror(image)

    angle = rng.uniform(-8, 8)
    image = image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False)

    width, height = image.size
    crop_ratio = rng.uniform(0.88, 1.0)
    crop_w = int(width * crop_ratio)
    crop_h = int(height * crop_ratio)
    left = rng.randint(0, max(0, width - crop_w))
    top = rng.randint(0, max(0, height - crop_h))
    image = image.crop((left, top, left + crop_w, top + crop_h)).resize(
        (width, height),
        Image.Resampling.BICUBIC,
    )

    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.82, 1.18))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.82, 1.22))
    image = ImageEnhance.Sharpness(image).enhance(rng.uniform(0.85, 1.25))

    if rng.random() < 0.18:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.25, 0.75)))

    return image


def augment_class(class_dir: Path, target_count: int, rng: random.Random) -> int:
    originals = image_files(class_dir)
    if not originals:
        return 0

    current_count = len(list(class_dir.glob("*")))
    created = 0
    while current_count + created < target_count:
        source = rng.choice(originals)
        with Image.open(source) as image:
            augmented = augment_image(image, rng)
        output = class_dir / f"{source.stem}_aug_{created:04d}.png"
        augmented.save(output)
        created += 1
    return created


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Balance rare textile classes with controlled train-only augmentation."
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["contamination", "knots", "weft_crack"],
        help="Class folders under dataset/train to augment.",
    )
    parser.add_argument("--target-count", type=int, default=120)
    args = parser.parse_args()

    rng = random.Random(SEED)
    for class_name in args.classes:
        class_dir = TRAIN_DIR / class_name
        created = augment_class(class_dir, args.target_count, rng)
        print(f"{class_name}: created {created} augmented training images")


if __name__ == "__main__":
    main()
