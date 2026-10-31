from pathlib import Path
from collections import Counter
import xml.etree.ElementTree as ET

ROOT = Path("data/processed/India")
TRAIN_IMAGES = ROOT / "train/images"
XML_DIR = ROOT / "train/annotations/xmls"
TEST_IMAGES = ROOT / "test/images"

def image_files(folder):
    return {
        p.stem: p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }

train_images = image_files(TRAIN_IMAGES)
test_images = image_files(TEST_IMAGES)
xml_files = {p.stem: p for p in XML_DIR.glob("*.xml")}

matched = set(train_images) & set(xml_files)
images_without_xml = set(train_images) - set(xml_files)
xml_without_image = set(xml_files) - set(train_images)

classes = Counter()
annotation_errors = []
object_counts = Counter()
sample_xml = None

for stem in sorted(matched):
    path = xml_files[stem]
    try:
        root = ET.parse(path).getroot()
        if sample_xml is None:
            sample_xml = root

        objects = root.findall("object")
        object_counts[stem] = len(objects)

        for obj in objects:
            name = obj.findtext("name", default="UNKNOWN").strip()
            classes[name] += 1
    except (ET.ParseError, OSError) as error:
        annotation_errors.append((path.name, str(error)))

print("\n=== DATASET SUMMARY ===")
print("Training images:", len(train_images))
print("Test images:", len(test_images))
print("XML annotation files:", len(xml_files))
print("Matching image/XML pairs:", len(matched))
print("Training images without XML:", len(images_without_xml))
print("XML files without matching image:", len(xml_without_image))
print("Annotation parsing errors:", len(annotation_errors))

print("\n=== DAMAGE CLASS COUNTS ===")
if classes:
    for name, count in classes.most_common():
        print(f"{name}: {count}")
else:
    print("No object labels found in matching XML files.")

print("\n=== OBJECTS PER ANNOTATED IMAGE ===")
if object_counts:
    print("Images with zero objects:", sum(n == 0 for n in object_counts.values()))
    print("Average objects per image:", round(sum(object_counts.values()) / len(object_counts), 2))

print("\n=== SAMPLE ANNOTATION ===")
if sample_xml is not None:
    print("Filename:", sample_xml.findtext("filename", default="not specified"))
    size = sample_xml.find("size")
    if size is not None:
        print("Width:", size.findtext("width"))
        print("Height:", size.findtext("height"))
    for obj in sample_xml.findall("object")[:3]:
        box = obj.find("bndbox")
        coords = {}
        if box is not None:
            coords = {tag: box.findtext(tag) for tag in ("xmin", "ymin", "xmax", "ymax")}
        print("Label:", obj.findtext("name", default="UNKNOWN"), "| Box:", coords)

if annotation_errors:
    print("\nFirst annotation errors:")
    for name, error in annotation_errors[:5]:
        print(name, "|", error)

print("\nAudit complete.")
