from pathlib import Path
import random
import yaml

SOURCE = Path("data/yolo_dataset")
OUTPUT = Path("data/experiment1")
SEED = 42
EMPTY_FRACTION = 0.25

if OUTPUT.exists():
    raise SystemExit(
        "Experiment 1 folder already exists. Stopping to avoid overwriting."
    )

random.seed(SEED)

image_extensions = {".jpg", ".jpeg", ".png"}

train_images = {
    p.stem: p
    for p in (SOURCE / "images/train").iterdir()
    if p.is_file() and p.suffix.lower() in image_extensions
}

train_labels = sorted((SOURCE / "labels/train").glob("*.txt"))

annotated = []
empty = []

for label in train_labels:
    image = train_images.get(label.stem)

    if image is None:
        raise FileNotFoundError(f"Image missing for label: {label.name}")

    if label.stat().st_size == 0:
        empty.append((image, label))
    else:
        annotated.append((image, label))

empty_count = round(len(empty) * EMPTY_FRACTION)
selected_empty = random.sample(empty, empty_count)

selected = annotated + selected_empty

random.shuffle(selected)

def create_links(items, image_dir, label_dir):
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    for image, label in items:
        (image_dir / image.name).symlink_to(image.resolve())
        (label_dir / label.name).symlink_to(label.resolve())

create_links(
    selected,
    OUTPUT / "images/train",
    OUTPUT / "labels/train"
)

# Keep the entire original validation set unchanged.
val_images = sorted(
    p for p in (SOURCE / "images/val").iterdir()
    if p.is_file() and p.suffix.lower() in image_extensions
)

val_labels = sorted((SOURCE / "labels/val").glob("*.txt"))

if len(val_images) != len(val_labels):
    raise ValueError("Validation image and label counts do not match.")

create_links(
    list(zip(val_images, val_labels)),
    OUTPUT / "images/val",
    OUTPUT / "labels/val"
)

# Reuse the original class names and configuration.
with open(SOURCE / "data.yaml", "r") as f:
    config = yaml.safe_load(f)

config["path"] = str(OUTPUT.resolve())
config["train"] = "images/train"
config["val"] = "images/val"

with open(OUTPUT / "data.yaml", "w") as f:
    yaml.safe_dump(config, f, sort_keys=False)

print("\n===== EXPERIMENT 1 CREATED =====")
print("Annotated training images:", len(annotated))
print("Empty training images available:", len(empty))
print("Empty training images selected:", len(selected_empty))
print("Total training images:", len(selected))
print("Validation images:", len(val_images))
print("Random seed:", SEED)
print("Dataset configuration:", OUTPUT / "data.yaml")
