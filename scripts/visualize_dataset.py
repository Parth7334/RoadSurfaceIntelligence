from pathlib import Path
import random
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

ROOT = Path("data/processed/India")
IMAGE_DIR = ROOT / "train/images"
XML_DIR = ROOT / "train/annotations/xmls"
TARGET = {"D00", "D10", "D20", "D40"}

groups = {
    "Target damage only": [],
    "No damage labels": [],
    "Mixed target and excluded": [],
    "Excluded damage only": []
}

for path in XML_DIR.glob("*.xml"):
    root = ET.parse(path).getroot()
    labels = [o.findtext("name", default="UNKNOWN").strip()
              for o in root.findall("object")]

    target = set(labels) & TARGET
    other = set(labels) - TARGET

    if not labels:
        group = "No damage labels"
    elif target and other:
        group = "Mixed target and excluded"
    elif target:
        group = "Target damage only"
    else:
        group = "Excluded damage only"

    groups[group].append(path)

random.seed(42)
selected = []

for group, paths in groups.items():
    selected.extend(
        (group, p) for p in random.sample(paths, min(2, len(paths)))
    )

fig, axes = plt.subplots(4, 2, figsize=(16, 18))

for ax, (group, path) in zip(axes.flat, selected):
    root = ET.parse(path).getroot()
    filename = root.findtext("filename", default=path.stem)
    image = Image.open(IMAGE_DIR / filename).convert("RGB")

    ax.imshow(image)

    for obj in root.findall("object"):
        label = obj.findtext("name", default="UNKNOWN")
        box = obj.find("bndbox")
        if box is None:
            continue

        x1 = float(box.findtext("xmin"))
        y1 = float(box.findtext("ymin"))
        x2 = float(box.findtext("xmax"))
        y2 = float(box.findtext("ymax"))

        color = "lime" if label in TARGET else "red"
        ax.add_patch(patches.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            linewidth=2, edgecolor=color, facecolor="none"
        ))
        ax.text(x1, max(0, y1-5), label, color="white",
                bbox=dict(facecolor=color, alpha=0.8))

    ax.set_title(f"{group}\n{filename}")
    ax.axis("off")

fig.suptitle("RDD2022 India - Annotation Inspection", fontsize=18)
plt.tight_layout()
plt.savefig("data/processed/visual_samples.png", dpi=150)
print("Saved: data/processed/visual_samples.png")
plt.show()
