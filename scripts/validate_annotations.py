from pathlib import Path
from collections import Counter
import xml.etree.ElementTree as ET

ROOT = Path("data/processed/India")
IMAGE_DIR = ROOT / "train/images"
XML_DIR = ROOT / "train/annotations/xmls"

classes = Counter()
invalid_boxes = []
dimension_mismatches = []
empty_images = 0
parsed = 0
label_examples = {}

for xml_path in sorted(XML_DIR.glob("*.xml")):
    try:
        root = ET.parse(xml_path).getroot()
    except (ET.ParseError, OSError) as error:
        print("XML ERROR:", xml_path.name, error)
        continue

    parsed += 1
    filename = root.findtext("filename", default=xml_path.stem)
    image_path = IMAGE_DIR / filename

    size = root.find("size")
    width = int(size.findtext("width", "0")) if size is not None else 0
    height = int(size.findtext("height", "0")) if size is not None else 0

    objects = root.findall("object")
    if not objects:
        empty_images += 1

    for obj in objects:
        label = obj.findtext("name", default="UNKNOWN").strip()
        classes[label] += 1
        label_examples.setdefault(label, xml_path.name)

        box = obj.find("bndbox")
        if box is None:
            invalid_boxes.append((xml_path.name, label, "missing bndbox"))
            continue

        try:
            xmin = float(box.findtext("xmin"))
            ymin = float(box.findtext("ymin"))
            xmax = float(box.findtext("xmax"))
            ymax = float(box.findtext("ymax"))
        except (TypeError, ValueError):
            invalid_boxes.append((xml_path.name, label, "non-numeric coordinates"))
            continue

        if not (0 <= xmin < xmax <= width and 0 <= ymin < ymax <= height):
            invalid_boxes.append(
                (xml_path.name, label, (xmin, ymin, xmax, ymax, width, height))
            )

    if image_path.exists():
        try:
            from PIL import Image
            with Image.open(image_path) as image:
                actual_width, actual_height = image.size
            if (actual_width, actual_height) != (width, height):
                dimension_mismatches.append(
                    (filename, (width, height), (actual_width, actual_height))
                )
        except Exception as error:
            dimension_mismatches.append((filename, "image read error", str(error)))

print("\n=== ANNOTATION VALIDATION ===")
print("XML files parsed:", parsed)
print("Images with no objects:", empty_images)
print("Total object annotations:", sum(classes.values()))
print("Invalid bounding boxes:", len(invalid_boxes))
print("Image/XML dimension mismatches:", len(dimension_mismatches))

print("\n=== LABEL COUNTS ===")
for label, count in classes.most_common():
    print(f"{label}: {count} | example XML: {label_examples[label]}")

if invalid_boxes:
    print("\nFirst 10 invalid boxes:")
    for item in invalid_boxes[:10]:
        print(item)

if dimension_mismatches:
    print("\nFirst 10 dimension mismatches:")
    for item in dimension_mismatches[:10]:
        print(item)

print("\nValidation complete.")
