from pathlib import Path
from collections import Counter
import random
import shutil
import xml.etree.ElementTree as ET

from PIL import Image

SOURCE = Path("data/processed/India")
OUTPUT = Path("data/yolo_dataset")

IMAGE_DIR = SOURCE / "train/images"
XML_DIR = SOURCE / "train/annotations/xmls"

CLASSES = {
    "D00": 0,
    "D10": 1,
    "D20": 2,
    "D40": 3,
}

CLASS_NAMES = [
    "Longitudinal crack",
    "Transverse crack",
    "Alligator crack",
    "Pothole",
]

VAL_RATIO = 0.20
SEED = 42

# Prevent accidental overwriting of an existing prepared dataset.
if OUTPUT.exists():
    raise FileExistsError(
        f"{OUTPUT} already exists. Inspect it before replacing anything."
    )

# Collect eligible images.
eligible = []
excluded_mixed = 0
excluded_other = 0
empty_count = 0

for xml_path in sorted(XML_DIR.glob("*.xml")):
    root = ET.parse(xml_path).getroot()
    filename = root.findtext("filename", default=xml_path.stem)
    image_path = IMAGE_DIR / filename

    if not image_path.exists():
        raise FileNotFoundError(image_path)

    objects = root.findall("object")
    labels = [
        obj.findtext("name", default="UNKNOWN").strip()
        for obj in objects
    ]

    target_labels = set(labels) & set(CLASSES)
    other_labels = set(labels) - set(CLASSES)

    if target_labels and other_labels:
        excluded_mixed += 1
        continue

    if other_labels:
        excluded_other += 1
        continue

    if not labels:
        empty_count += 1

    eligible.append((image_path, xml_path))

print("Eligible images:", len(eligible))
print("Empty annotation images:", empty_count)
print("Excluded mixed images:", excluded_mixed)
print("Excluded other-only images:", excluded_other)

# Deterministic random 80:20 split.
random.seed(SEED)
random.shuffle(eligible)

split_index = int(len(eligible) * (1 - VAL_RATIO))
train_items = eligible[:split_index]
val_items = eligible[split_index:]

print("\nTraining images:", len(train_items))
print("Validation images:", len(val_items))

# Create YOLO directory structure.
for split in ("train", "val"):
    (OUTPUT / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUTPUT / "labels" / split).mkdir(parents=True, exist_ok=True)

def convert_annotation(xml_path, image_path, label_path):
    root = ET.parse(xml_path).getroot()

    with Image.open(image_path) as image:
        actual_width, actual_height = image.size

    size = root.find("size")
    width = int(size.findtext("width"))
    height = int(size.findtext("height"))

    if (width, height) != (actual_width, actual_height):
        raise ValueError(f"Dimension mismatch: {image_path.name}")

    lines = []

    for obj in root.findall("object"):
        label = obj.findtext("name", default="UNKNOWN").strip()

        if label not in CLASSES:
            raise ValueError(
                f"Unexpected label {label} in {xml_path.name}"
            )

        box = obj.find("bndbox")
        xmin = float(box.findtext("xmin"))
        ymin = float(box.findtext("ymin"))
        xmax = float(box.findtext("xmax"))
        ymax = float(box.findtext("ymax"))

        if not (0 <= xmin < xmax <= width and 0 <= ymin < ymax <= height):
            raise ValueError(f"Invalid bounding box in {xml_path.name}")

        x_center = ((xmin + xmax) / 2) / width
        y_center = ((ymin + ymax) / 2) / height
        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height

        lines.append(
            f"{CLASSES[label]} "
            f"{x_center:.6f} {y_center:.6f} "
            f"{box_width:.6f} {box_height:.6f}"
        )

    # An empty file represents an image with no target objects.
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""))

def process_split(items, split):
    counts = Counter()
    empty_labels = 0

    for image_path, xml_path in items:
        destination_image = OUTPUT / "images" / split / image_path.name
        destination_label = (
            OUTPUT / "labels" / split / f"{image_path.stem}.txt"
        )

        shutil.copy2(image_path, destination_image)
        convert_annotation(xml_path, image_path, destination_label)

        lines = destination_label.read_text().splitlines()

        if not lines:
            empty_labels += 1

        for line in lines:
            class_id = int(line.split()[0])
            counts[class_id] += 1

    return counts, empty_labels

train_counts, train_empty = process_split(train_items, "train")
val_counts, val_empty = process_split(val_items, "val")

# Create YOLO configuration.
yaml_content = f"""path: {OUTPUT.resolve()}
train: images/train
val: images/val

nc: 4
names:
"""

for index, name in enumerate(CLASS_NAMES):
    yaml_content += f"  {index}: {name}\n"

(OUTPUT / "data.yaml").write_text(yaml_content)

print("\n=== CONVERSION SUMMARY ===")
print("Training images:", len(train_items))
print("Validation images:", len(val_items))
print("Training empty labels:", train_empty)
print("Validation empty labels:", val_empty)

print("\n=== OBJECT COUNTS ===")
for class_id, name in enumerate(CLASS_NAMES):
    print(
        f"{class_id} - {name}: "
        f"Train={train_counts[class_id]}, "
        f"Validation={val_counts[class_id]}"
    )

print("\nYOLO configuration:", OUTPUT / "data.yaml")
print("Dataset preparation completed successfully.")
