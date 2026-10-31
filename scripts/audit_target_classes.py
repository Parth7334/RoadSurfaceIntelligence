from pathlib import Path
from collections import Counter
import xml.etree.ElementTree as ET

XML_DIR = Path("data/processed/India/train/annotations/xmls")
TARGET = {"D00", "D10", "D20", "D40"}

image_groups = Counter()
excluded_classes = Counter()
target_counts = Counter()
examples = {}

for path in sorted(XML_DIR.glob("*.xml")):
    root = ET.parse(path).getroot()
    labels = [
        obj.findtext("name", default="UNKNOWN").strip()
        for obj in root.findall("object")
    ]

    target_labels = [label for label in labels if label in TARGET]
    other_labels = [label for label in labels if label not in TARGET]

    if not labels:
        group = "No objects"
    elif target_labels and other_labels:
        group = "Target and non-target labels"
    elif target_labels:
        group = "Target labels only"
    else:
        group = "Non-target labels only"

    image_groups[group] += 1
    examples.setdefault(group, path.name)

    target_counts.update(target_labels)
    excluded_classes.update(other_labels)

print("=== IMAGE GROUPS ===")
for group in (
    "Target labels only",
    "Target and non-target labels",
    "Non-target labels only",
    "No objects",
):
    print(f"{group}: {image_groups[group]}")
    if group in examples:
        print(f"  Example: {examples[group]}")

print("\n=== TARGET CLASS INSTANCES ===")
for label, count in sorted(target_counts.items()):
    print(f"{label}: {count}")
print("Total target instances:", sum(target_counts.values()))

print("\n=== NON-TARGET LABEL INSTANCES ===")
for label, count in excluded_classes.most_common():
    print(f"{label}: {count}")
print("Total non-target instances:", sum(excluded_classes.values()))
