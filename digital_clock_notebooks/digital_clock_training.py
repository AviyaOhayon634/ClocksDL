from ultralytics import YOLO

# ========= CONFIG =========
DATA_YAML = r"new_merged_dataset\data.yaml"  # שים r למניעת בעיות בנתיב
MODEL = "yolov8n.pt"
EPOCHS = 50
IMG_SIZE = 512


def main():
    # ========= LOAD MODEL =========
    model = YOLO(MODEL)

    # ========= TRAIN =========
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=32,
        workers=4,
        device=0,
        name="digit_detector4",
        patience=20,
        save=True,
        verbose=True
    )

    # ========= DONE =========
    print("Training finished 🚀")


if __name__ == "__main__":
    main()