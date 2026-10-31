
from pathlib import Path
from ultralytics import YOLO
import argparse

# Model configuration
MODEL_PATH = Path("models/road_damage_yolo11n.pt")

CLASS_NAMES = {
    0: "Longitudinal crack",
    1: "Transverse crack",
    2: "Alligator crack",
    3: "Pothole",
}


def predict(image_path, confidence=0.25):
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    model = YOLO(MODEL_PATH)

    results = model.predict(
        source=str(image_path),
        conf=confidence,
        imgsz=512,
        device="mps",
        save=False,
        verbose=False
    )

    result = results[0]

    # Save annotated image
    output_dir = Path("predictions")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / f"{image_path.stem}_result.jpg"

    annotated_image = result.plot()
    from PIL import Image
    Image.fromarray(annotated_image[:, :, ::-1]).save(output_path)

    # Print detection summary
    print("\n===== ROAD DAMAGE ANALYSIS =====")
    print(f"Image: {image_path.name}")
    print(f"Confidence threshold: {confidence}")

    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        print("\nNo road damage detected.")
    else:
        print(f"\nTotal detections: {len(boxes)}")

        for index, box in enumerate(boxes, start=1):
            class_id = int(box.cls.item())
            score = float(box.conf.item())
            coordinates = box.xyxy[0].tolist()

            print(f"\nDetection {index}")
            print(f"Class: {CLASS_NAMES[class_id]}")
            print(f"Confidence: {score:.2%}")
            print(
                "Bounding box:",
                [round(value, 2) for value in coordinates]
            )

    print(f"\nAnnotated image saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Road Surface Intelligence"
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to road image"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)"
    )

    args = parser.parse_args()

    predict(args.image, args.conf)

