from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path("data/yolo_dataset")
OUTPUT = Path("reports/eda")
OUTPUT.mkdir(parents=True, exist_ok=True)

SPLITS = ["train", "val"]

CLASS_NAMES = {
    0: "D00 - Longitudinal",
    1: "D10 - Transverse",
    2: "D20 - Alligator",
    3: "D40 - Pothole"
}

class_counts = Counter()
objects_per_image = []
box_widths = []
box_heights = []
box_areas = []
aspect_ratios = []

empty_images = Counter()
annotated_images = Counter()
total_images = Counter()

for split in SPLITS:
    image_dir = ROOT / "images" / split
    label_dir = ROOT / "labels" / split

    image_paths = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    for image_path in image_paths:
        label_path = label_dir / f"{image_path.stem}.txt"

        if not label_path.exists():
            raise FileNotFoundError(label_path)

        total_images[split] += 1
        lines = label_path.read_text().strip().splitlines()

        if not lines:
            empty_images[split] += 1
            objects_per_image.append(0)
            continue

        annotated_images[split] += 1
        image_objects = 0

        with Image.open(image_path) as image:
            image_width, image_height = image.size

        for line in lines:
            values = line.split()

            if len(values) != 5:
                raise ValueError(
                    f"Invalid YOLO annotation in {label_path}: {line}"
                )

            class_id = int(values[0])
            x_center, y_center, width, height = map(float, values[1:])

            if class_id not in CLASS_NAMES:
                raise ValueError(f"Unknown class ID: {class_id}")

            if not (
                0 <= x_center <= 1
                and 0 <= y_center <= 1
                and 0 < width <= 1
                and 0 < height <= 1
                and x_center - width / 2 >= -1e-6
                and y_center - height / 2 >= -1e-6
                and x_center + width / 2 <= 1 + 1e-6
                and y_center + height / 2 <= 1 + 1e-6
            ):
                raise ValueError(
                    f"Invalid normalized box in {label_path}: {line}"
                )

            class_counts[class_id] += 1
            image_objects += 1

            box_widths.append(width * image_width)
            box_heights.append(height * image_height)
            box_areas.append(width * height)
            aspect_ratios.append(width / height)

        objects_per_image.append(image_objects)

print("\n========== DATASET OVERVIEW ==========")

for split in SPLITS:
    print(f"\n{split.upper()}")
    print("Total images:", total_images[split])
    print("Images with annotations:", annotated_images[split])
    print("Empty images:", empty_images[split])

print("\n========== CLASS DISTRIBUTION ==========")

for class_id, name in CLASS_NAMES.items():
    print(f"{name}: {class_counts[class_id]}")

print("\nTotal objects:", sum(class_counts.values()))

print("\n========== OBJECT STATISTICS ==========")
print("Average objects per image:", round(np.mean(objects_per_image), 2))
print("Maximum objects in one image:", max(objects_per_image))
print("Median bounding-box width (pixels):", round(np.median(box_widths), 2))
print("Median bounding-box height (pixels):", round(np.median(box_heights), 2))
print("Median normalized box area:", round(np.median(box_areas), 5))
print("Median box aspect ratio:", round(np.median(aspect_ratios), 2))

# Chart 1: Class distribution
ids = list(CLASS_NAMES.keys())
names = [CLASS_NAMES[i] for i in ids]
counts = [class_counts[i] for i in ids]

plt.figure(figsize=(10, 6))
plt.bar(names, counts)
plt.title("Road Damage Class Distribution")
plt.ylabel("Number of Objects")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT / "class_distribution.png", dpi=150)
plt.close()

# Chart 2: Objects per image
plt.figure(figsize=(10, 6))
plt.hist(objects_per_image, bins=range(0, max(objects_per_image) + 2))
plt.title("Number of Objects per Image")
plt.xlabel("Objects per Image")
plt.ylabel("Number of Images")
plt.tight_layout()
plt.savefig(OUTPUT / "objects_per_image.png", dpi=150)
plt.close()

# Chart 3: Bounding-box dimensions
plt.figure(figsize=(10, 6))
plt.scatter(box_widths, box_heights, alpha=0.4)
plt.title("Bounding Box Width vs Height")
plt.xlabel("Width (pixels)")
plt.ylabel("Height (pixels)")
plt.tight_layout()
plt.savefig(OUTPUT / "bounding_box_dimensions.png", dpi=150)
plt.close()

# Chart 4: Bounding-box area
plt.figure(figsize=(10, 6))
plt.hist(box_areas, bins=30)
plt.title("Normalized Bounding Box Area Distribution")
plt.xlabel("Bounding Box Area / Image Area")
plt.ylabel("Number of Objects")
plt.tight_layout()
plt.savefig(OUTPUT / "bounding_box_area.png", dpi=150)
plt.close()

print("\n========== REPORT FILES ==========")
for path in sorted(OUTPUT.glob("*.png")):
    print(path)

print("\nEDA completed successfully.")
