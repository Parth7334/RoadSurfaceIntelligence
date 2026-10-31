from pathlib import Path
import xml.etree.ElementTree as ET
import random

BASE = Path("data/processed/India/train")
XML_DIR = BASE / "annotations/xmls"
IMAGE_DIR = BASE / "images"

EXP1 = Path("data/experiment1")
OUTPUT = Path("data/experiment2")

CLASSES = {
    "D00": 0,
    "D10": 1,
    "D20": 2,
    "D40": 3,
}

random.seed(42)

if OUTPUT.exists():
    raise FileExistsError("Experiment 2 already exists. Refusing to overwrite.")

# Start with the complete Experiment 1 dataset.
for split in ["train", "val"]:
    for kind in ["images", "labels"]:
        destination = OUTPUT / kind / split
        destination.mkdir(parents=True, exist_ok=True)

        source = EXP1 / kind / split
        for item in source.iterdir():
            (destination / item.name).symlink_to(item.resolve())

# Map image filenames by stem.
images = {p.stem: p for p in IMAGE_DIR.glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]}

mixed_added = 0
target_objects_added = 0

for xml_file in XML_DIR.glob("*.xml"):
    root = ET.parse(xml_file).getroot()

    objects = root.findall("object")
    names = [obj.findtext("name", "").strip() for obj in objects]

    has_target = any(name in CLASSES for name in names)
    has_non_target = any(name not in CLASSES for name in names)

    if not (has_target and has_non_target):
        continue

    stem = xml_file.stem
    image = images.get(stem)

    if image is None:
        raise FileNotFoundError(f"Image missing for {xml_file.name}")

    size = root.find("size")
    width = float(size.findtext("width"))
    height = float(size.findtext("height"))

    yolo_lines = []

    for obj in objects:
        name = obj.findtext("name", "").strip()

        if name not in CLASSES:
            continue

        box = obj.find("bndbox")
        xmin = float(box.findtext("xmin"))
        ymin = float(box.findtext("ymin"))
        xmax = float(box.findtext("xmax"))
        ymax = float(box.findtext("ymax"))

        x_center = ((xmin + xmax) / 2) / width
        y_center = ((ymin + ymax) / 2) / height
        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height

        yolo_lines.append(
            f"{CLASSES[name]} {x_center:.6f} {y_center:.6f} "
            f"{box_width:.6f} {box_height:.6f}"
        )

    if not yolo_lines:
        continue

    image_dest = OUTPUT / "images/train" / image.name
    label_dest = OUTPUT / "labels/train" / f"{stem}.txt"

    if image_dest.exists() or label_dest.exists():
        raise FileExistsError(f"Duplicate image or label: {stem}")

    image_dest.symlink_to(image.resolve())
    label_dest.write_text("\n".join(yolo_lines) + "\n")

    mixed_added += 1
    target_objects_added += len(yolo_lines)

# Reuse the same class configuration and validation set.
import yaml

with open(EXP1 / "data.yaml") as f:
    config = yaml.safe_load(f)

config["path"] = str(OUTPUT.resolve())

with open(OUTPUT / "data.yaml", "w") as f:
    yaml.safe_dump(config, f, sort_keys=False)

print("\n===== EXPERIMENT 2 CREATED =====")
print("Mixed images added:", mixed_added)
print("Target objects added:", target_objects_added)
print("Training images:", len(list((OUTPUT / "images/train").iterdir())))
print("Validation images:", len(list((OUTPUT / "images/val").iterdir())))
print("Dataset configuration:", OUTPUT / "data.yaml")
