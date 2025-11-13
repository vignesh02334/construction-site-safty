import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 for PPE detection")
    parser.add_argument("--data", required=True, help="Path to dataset YAML (Ultralytics format)")
    parser.add_argument("--model", default="yolov8n.pt", help="Base model weights (e.g., yolov8n.pt)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--project", default="runs/ppe", help="Training output dir")
    parser.add_argument("--name", default="exp", help="Experiment name")
    args = parser.parse_args()

    yolo = YOLO(args.model)
    results = yolo.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
    )
    print("Training complete.")
    if hasattr(results, "save_dir"):
        print("Artifacts in:", results.save_dir)
    print("Tip: use the best.pt path with POST /model/reload to hot-swap the model.")


if __name__ == "__main__":
    main()

