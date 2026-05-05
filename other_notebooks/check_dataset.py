import csv
import os
import random
from PIL import Image, ImageDraw, ImageFont

DATA_DIR = "data"
TRAIN_LABELS = os.path.join(DATA_DIR, "train_labels.csv")
TEST_LABELS = os.path.join(DATA_DIR, "test_labels.csv")


def read_csv(path):
    with open(path, mode="r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def check_paths():
    print("Current folder:", os.getcwd())
    print("Train labels exists:", os.path.exists(TRAIN_LABELS))
    print("Test labels exists:", os.path.exists(TEST_LABELS))


def check_style_split():
    train_rows = read_csv(TRAIN_LABELS)
    test_rows = read_csv(TEST_LABELS)

    train_styles = {int(row["analog_style_id"]) for row in train_rows}
    test_styles = {int(row["analog_style_id"]) for row in test_rows}

    print("\nTrain styles:", sorted(train_styles))
    print("Test styles:", sorted(test_styles))

    overlap = train_styles.intersection(test_styles)

    if len(overlap) == 0:
        print("No overlap - correct style split")
    else:
        print("Error: overlap found:", sorted(overlap))


def check_angles(labels_path, num_samples=5):
    rows = read_csv(labels_path)
    samples = random.sample(rows, min(num_samples, len(rows)))

    for row in samples:
        h = int(row["hour"])
        m = int(row["minute"])
        s = int(row["second"])

        expected_minute = m * 6 + s * 0.1
        expected_hour = (h % 12) * 30 + m * 0.5 + s / 120
        expected_second = s * 6

        print("\nTime:", f"{h:02d}:{m:02d}:{s:02d}")
        print("Minute angle:", round(expected_minute, 2), "| Saved:", row["minute_angle"])
        print("Hour angle:", round(expected_hour, 2), "| Saved:", row["hour_angle"])
        print("Second angle:", round(expected_second, 2), "| Saved:", row["second_angle"])


def check_distribution(labels_path):
    rows = read_csv(labels_path)

    hour_counts = {}
    for row in rows:
        hour = int(row["hour"])
        hour_counts[hour] = hour_counts.get(hour, 0) + 1

    print("\nHour distribution:")
    for hour in sorted(hour_counts):
        print(f"{hour:02d}: {hour_counts[hour]}")


def make_preview(labels_path, num_samples=8, output_path="dataset_preview.png"):
    rows = read_csv(labels_path)
    samples = random.sample(rows, min(num_samples, len(rows)))

    cell_w = 256
    cell_h = 310
    cols = 3
    rows_count = len(samples)

    preview = Image.new("RGB", (cols * cell_w, rows_count * cell_h), "white")
    draw = ImageDraw.Draw(preview)

    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    for i, row in enumerate(samples):
        images = [
            ("Digital", row["digital_path"]),
            ("Analog", row["analog_path"]),
            ("Clean", row["analog_without_hands_path"]),
        ]

        time_text = f'{int(row["hour"]):02d}:{int(row["minute"]):02d}:{int(row["second"]):02d}'

        for j, (title, path) in enumerate(images):
            img = Image.open(path).convert("RGB").resize((224, 224))

            x = j * cell_w + 16
            y = i * cell_h + 55

            preview.paste(img, (x, y))

            title_text = f"{title} - {time_text}" if j == 0 else title
            draw.text((j * cell_w + 16, i * cell_h + 25), title_text, fill="black", font=font)

    preview.save(output_path)
    print(f"\nSaved preview to: {output_path}")


if __name__ == "__main__":
    print("CHECK DATASET")

    check_paths()

    print("\nChecking style split")
    check_style_split()

    print("\nChecking angles")
    check_angles(TRAIN_LABELS)

    print("\nChecking distribution")
    check_distribution(TRAIN_LABELS)

    print("\nCreating preview image")
    make_preview(TRAIN_LABELS)