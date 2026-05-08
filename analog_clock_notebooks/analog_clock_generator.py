"""
data_generator_analog_only.py

Synthetic data generator for ClocksDL (Analog Only).
Creates a dataset of 400 paired analog clock examples with 40 diverse styles.

For each sample we generate:
1. Analog clock with hands showing the time
2. The same analog clock without hands
3. CSV labels with hour, minute, second and hand angles

Dataset structure:
    data/
        train/
            analog_clock/
            analog_without_hands/
        test/
            analog_clock/
            analog_without_hands/
        train_labels.csv
        test_labels.csv
        labels.csv

Run:
    python data_generator_analog_only.py
"""

import csv
import math
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont, ImageFilter


# =========================
# Configuration
# =========================

DATASET_SIZE = 16000
IMAGE_SIZE = 256
OUTPUT_DIR = Path("data")
SEED = 42

TRAIN_RATIO = 0.8
SPLIT_BY_STYLE = True

NUM_ANALOG_STYLES = 40

SUBFOLDERS = ["analog_clock", "analog_without_hands"]

LABELS_PATH = OUTPUT_DIR / "labels.csv"
TRAIN_LABELS_PATH = OUTPUT_DIR / "train_labels.csv"
TEST_LABELS_PATH = OUTPUT_DIR / "test_labels.csv"


# =========================
# General utilities
# =========================

def reset_output_dir() -> None:
    """Remove old generated data and create a clean folder structure."""
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    for split in ["train", "test"]:
        for subfolder in SUBFOLDERS:
            (OUTPUT_DIR / split / subfolder).mkdir(parents=True, exist_ok=True)


def get_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    """Load a system font. Fall back to default if unavailable."""
    possible_fonts = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "arialbd.ttf" if bold else "arial.ttf",
        "Arial Bold.ttf" if bold else "Arial.ttf",
    ]

    for font_path in possible_fonts:
        try:
            return ImageFont.truetype(font_path, size)
        except OSError:
            continue

    return ImageFont.load_default()


def draw_centered_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont,
                       center: Tuple[int, int], fill) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    draw.text((center[0] - width / 2, center[1] - height / 2), text, font=font, fill=fill)


def angle_to_point(center: Tuple[int, int], length: float, angle_degrees: float) -> Tuple[int, int]:
    """
    Clock convention:
    0 degrees = 12 o'clock, 90 degrees = 3 o'clock.
    """
    angle_rad = math.radians(angle_degrees - 90)
    x = center[0] + length * math.cos(angle_rad)
    y = center[1] + length * math.sin(angle_rad)
    return int(round(x)), int(round(y))


def time_to_angles(hour: int, minute: int, second: int) -> Tuple[float, float, float]:
    """
    Convert HH:MM:SS to analog hand angles.
    """
    hour_12 = hour % 12
    second_angle = second * 6.0
    minute_angle = minute * 6.0 + second * 0.1
    hour_angle = hour_12 * 30.0 + minute * 0.5 + second / 120.0
    return hour_angle, minute_angle, second_angle


def random_time() -> Tuple[int, int, int]:
    return random.randint(0, 23), random.randint(0, 59), random.randint(0, 59)


def add_sensor_noise(img: Image.Image, amount: int = 6) -> Image.Image:
    """Add mild pixel noise to make synthetic images less perfect."""
    pixels = img.load()
    width, height = img.size

    for _ in range((width * height) // 80):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        r, g, b = pixels[x, y]
        delta = random.randint(-amount, amount)
        pixels[x, y] = (
            max(0, min(255, r + delta)),
            max(0, min(255, g + delta)),
            max(0, min(255, b + delta)),
        )

    return img


# =========================
# Analog clocks (40 Styles)
# =========================

def analog_style_config(style_id: int) -> Dict:
    styles = [
        # --- ORIGINAL STYLES (1-16) ---
        {"name": "teacher_blue_learning", "bg": (250, 250, 248), "face": (145, 215, 235), "inner": (230, 245, 170), "outline": (170, 155, 120), "marker": (40, 90, 110), "numbers": True, "dots": True, "rim": 5, "playful": True},
        {"name": "teacher_yellow_kids", "bg": (250, 250, 248), "face": (250, 220, 20), "inner": None, "outline": (240, 190, 20), "marker": (235, 70, 25), "numbers": True, "dots": False, "rim": 4, "playful": True},
        {"name": "teacher_lcd_pair_simple", "bg": (250, 250, 250), "face": (255, 255, 252), "inner": None, "outline": (30, 30, 30), "marker": (5, 5, 5), "numbers": True, "dots": False, "rim": 7, "playful": False},
        {"name": "classic_white_numbers", "bg": (248, 248, 245), "face": (255, 255, 252), "inner": None, "outline": (25, 25, 25), "marker": (10, 10, 10), "numbers": True, "dots": False, "rim": 5, "playful": False},
        {"name": "black_white_numbers", "bg": (245, 245, 245), "face": (255, 255, 255), "inner": None, "outline": (15, 15, 15), "marker": (5, 5, 5), "numbers": True, "dots": False, "rim": 8, "playful": False},
        {"name": "minimal_white", "bg": (245, 245, 242), "face": (252, 252, 248), "inner": None, "outline": (60, 60, 60), "marker": (50, 50, 50), "numbers": False, "dots": False, "rim": 3, "playful": False},
        {"name": "brown_dot_minimal", "bg": (174, 174, 168), "face": (128, 65, 45), "inner": None, "outline": (128, 65, 45), "marker": (195, 195, 185), "numbers": False, "dots": True, "rim": 0, "playful": False},
        {"name": "simple_beige", "bg": (225, 204, 175), "face": (235, 210, 175), "inner": None, "outline": (120, 90, 60), "marker": (120, 90, 60), "numbers": False, "dots": True, "rim": 2, "playful": False},
        {"name": "cream_gold", "bg": (236, 228, 210), "face": (252, 246, 226), "inner": None, "outline": (180, 145, 70), "marker": (160, 120, 60), "numbers": True, "dots": False, "rim": 4, "playful": False},
        {"name": "dark_luxury", "bg": (30, 30, 30), "face": (18, 18, 20), "inner": None, "outline": (210, 170, 80), "marker": (220, 190, 100), "numbers": False, "dots": False, "rim": 4, "playful": False},
        {"name": "blue_clean", "bg": (224, 236, 246), "face": (232, 244, 255), "inner": None, "outline": (50, 100, 150), "marker": (50, 100, 150), "numbers": True, "dots": False, "rim": 3, "playful": False},
        {"name": "gray_industrial", "bg": (210, 210, 210), "face": (235, 235, 232), "inner": None, "outline": (70, 70, 70), "marker": (30, 30, 30), "numbers": False, "dots": True, "rim": 6, "playful": False},
        {"name": "pastel_kids", "bg": (245, 236, 240), "face": (255, 245, 250), "inner": None, "outline": (210, 120, 160), "marker": (120, 80, 120), "numbers": True, "dots": True, "rim": 4, "playful": True},
        {"name": "green_classroom", "bg": (235, 245, 235), "face": (220, 250, 215), "inner": None, "outline": (70, 130, 70), "marker": (40, 100, 45), "numbers": True, "dots": False, "rim": 4, "playful": True},
        {"name": "red_modern", "bg": (245, 235, 235), "face": (255, 238, 238), "inner": None, "outline": (160, 40, 40), "marker": (120, 30, 30), "numbers": False, "dots": True, "rim": 5, "playful": False},
        {"name": "navy_minimal", "bg": (225, 230, 238), "face": (25, 35, 55), "inner": None, "outline": (230, 230, 220), "marker": (230, 230, 220), "numbers": False, "dots": False, "rim": 4, "playful": False},

        # --- NEW STYLES (17-40) ---
        {"name": "wooden_oak", "bg": (220, 210, 200), "face": (180, 130, 80), "inner": None, "outline": (100, 60, 30), "marker": (60, 30, 10), "numbers": True, "dots": False, "rim": 6, "playful": False},
        {"name": "wooden_mahogany", "bg": (240, 235, 230), "face": (100, 30, 10), "inner": None, "outline": (60, 10, 0), "marker": (240, 200, 150), "numbers": False, "dots": True, "rim": 8, "playful": False},
        {"name": "neon_pink", "bg": (10, 10, 15), "face": (20, 20, 25), "inner": None, "outline": (255, 20, 147), "marker": (255, 105, 180), "numbers": True, "dots": False, "rim": 3, "playful": False},
        {"name": "neon_cyan", "bg": (15, 15, 15), "face": (25, 25, 25), "inner": None, "outline": (0, 255, 255), "marker": (0, 200, 200), "numbers": False, "dots": True, "rim": 3, "playful": False},
        {"name": "steampunk_copper", "bg": (180, 170, 160), "face": (184, 115, 51), "inner": (140, 70, 20), "outline": (90, 40, 10), "marker": (50, 20, 5), "numbers": True, "dots": True, "rim": 5, "playful": False},
        {"name": "steampunk_brass", "bg": (190, 180, 170), "face": (225, 193, 110), "inner": None, "outline": (130, 100, 40), "marker": (80, 60, 20), "numbers": False, "dots": True, "rim": 7, "playful": False},
        {"name": "vintage_parchment", "bg": (245, 240, 230), "face": (238, 224, 193), "inner": None, "outline": (120, 90, 50), "marker": (70, 40, 20), "numbers": True, "dots": False, "rim": 3, "playful": False},
        {"name": "scandi_mint", "bg": (250, 250, 250), "face": (240, 250, 245), "inner": None, "outline": (150, 200, 180), "marker": (100, 150, 130), "numbers": False, "dots": False, "rim": 2, "playful": False},
        {"name": "scandi_blush", "bg": (250, 250, 250), "face": (255, 240, 240), "inner": None, "outline": (220, 180, 180), "marker": (180, 140, 140), "numbers": False, "dots": True, "rim": 2, "playful": False},
        {"name": "pop_art_yellow", "bg": (200, 230, 255), "face": (255, 220, 0), "inner": None, "outline": (0, 0, 255), "marker": (255, 0, 0), "numbers": True, "dots": True, "rim": 6, "playful": True},
        {"name": "pop_art_pink", "bg": (220, 255, 220), "face": (255, 100, 150), "inner": None, "outline": (0, 200, 0), "marker": (255, 255, 0), "numbers": True, "dots": False, "rim": 5, "playful": True},
        {"name": "space_dark", "bg": (10, 10, 20), "face": (5, 5, 15), "inner": None, "outline": (100, 150, 255), "marker": (255, 255, 255), "numbers": False, "dots": True, "rim": 2, "playful": False},
        {"name": "sunset_orange", "bg": (255, 230, 200), "face": (255, 150, 50), "inner": (255, 100, 0), "outline": (200, 50, 0), "marker": (100, 20, 0), "numbers": True, "dots": False, "rim": 4, "playful": False},
        {"name": "ocean_deep", "bg": (200, 220, 240), "face": (10, 60, 100), "inner": None, "outline": (0, 30, 60), "marker": (150, 200, 255), "numbers": False, "dots": True, "rim": 5, "playful": False},
        {"name": "forest_green", "bg": (230, 240, 220), "face": (30, 80, 40), "inner": None, "outline": (10, 40, 15), "marker": (180, 220, 170), "numbers": True, "dots": False, "rim": 4, "playful": False},
        {"name": "retro_70s", "bg": (240, 220, 190), "face": (150, 80, 30), "inner": (200, 120, 40), "outline": (100, 40, 10), "marker": (255, 220, 150), "numbers": True, "dots": True, "rim": 6, "playful": False},
        {"name": "cyberpunk_purple", "bg": (20, 10, 30), "face": (40, 10, 60), "inner": None, "outline": (180, 50, 255), "marker": (0, 255, 200), "numbers": False, "dots": False, "rim": 3, "playful": False},
        {"name": "royal_crimson", "bg": (245, 235, 225), "face": (140, 10, 20), "inner": None, "outline": (210, 170, 80), "marker": (210, 170, 80), "numbers": True, "dots": False, "rim": 5, "playful": False},
        {"name": "ice_blue", "bg": (240, 250, 255), "face": (200, 235, 255), "inner": (150, 210, 255), "outline": (100, 180, 255), "marker": (20, 80, 150), "numbers": False, "dots": True, "rim": 3, "playful": False},
        {"name": "desert_sand", "bg": (245, 235, 220), "face": (210, 180, 140), "inner": None, "outline": (139, 69, 19), "marker": (139, 69, 19), "numbers": True, "dots": False, "rim": 4, "playful": False},
        {"name": "monochrome_dark", "bg": (50, 50, 50), "face": (80, 80, 80), "inner": None, "outline": (20, 20, 20), "marker": (200, 200, 200), "numbers": False, "dots": True, "rim": 6, "playful": False},
        {"name": "monochrome_light", "bg": (220, 220, 220), "face": (200, 200, 200), "inner": None, "outline": (150, 150, 150), "marker": (50, 50, 50), "numbers": True, "dots": False, "rim": 2, "playful": False},
        {"name": "kid_candy_floss", "bg": (255, 240, 250), "face": (255, 200, 230), "inner": (255, 220, 240), "outline": (255, 100, 180), "marker": (150, 50, 100), "numbers": True, "dots": True, "rim": 5, "playful": True},
        {"name": "kid_bubblegum_blue", "bg": (230, 245, 255), "face": (150, 210, 255), "inner": None, "outline": (50, 150, 255), "marker": (255, 100, 150), "numbers": True, "dots": False, "rim": 5, "playful": True},
    ]
    return styles[style_id % len(styles)]


def create_analog_clean(style_id: int, image_size: int = IMAGE_SIZE) -> Image.Image:
    """Create analog clock face without hands."""
    cfg = analog_style_config(style_id)
    img = Image.new("RGB", (image_size, image_size), color=cfg["bg"])
    draw = ImageDraw.Draw(img)

    center = (image_size // 2, image_size // 2)
    radius = image_size // 2 - random.randint(20, 28)

    shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    shadow_offset = random.randint(4, 8)
    shadow_box = [
        center[0] - radius + shadow_offset,
        center[1] - radius + shadow_offset,
        center[0] + radius + shadow_offset,
        center[1] + radius + shadow_offset,
    ]
    shadow_draw.ellipse(shadow_box, fill=(0, 0, 0, 45))
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=8))
    img = Image.alpha_composite(img.convert("RGBA"), shadow).convert("RGB")
    draw = ImageDraw.Draw(img)

    face_box = [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius]
    draw.ellipse(face_box, fill=cfg["face"], outline=cfg["outline"], width=max(1, cfg["rim"]))

    if cfg.get("inner") is not None:
        inner_radius = int(radius * 0.42)
        inner_box = [
            center[0] - inner_radius,
            center[1] - inner_radius,
            center[0] + inner_radius,
            center[1] + inner_radius,
        ]
        draw.ellipse(inner_box, fill=cfg["inner"], outline=tuple(max(0, c - 25) for c in cfg["inner"]), width=2)

    for i in range(60):
        angle = i * 6
        if cfg["dots"] and i % 5 == 0:
            point = angle_to_point(center, radius - 18, angle)
            dot_radius = 4 if i % 15 == 0 else 3
            draw.ellipse(
                [point[0] - dot_radius, point[1] - dot_radius, point[0] + dot_radius, point[1] + dot_radius],
                fill=cfg["marker"],
            )
        elif not cfg["dots"]:
            outer = angle_to_point(center, radius - 8, angle)
            inner = angle_to_point(center, radius - (22 if i % 5 == 0 else 15), angle)
            width = 3 if i % 5 == 0 else 1
            draw.line([inner, outer], fill=cfg["marker"], width=width)

    if cfg["numbers"]:
        font = get_font(29 if cfg.get("playful") else 22, bold=True)
        for number in range(1, 13):
            angle = number * 30
            position = angle_to_point(center, radius - 42, angle)
            draw_centered_text(draw, str(number), font, position, cfg["marker"])

    if random.random() < 0.2:
        brand_font = get_font(10, bold=False)
        draw_centered_text(
            draw,
            random.choice(["QUARTZ", "CLASSIC", "TIME", "CHRONOS"]),
            brand_font,
            (center[0], center[1] + 42),
            cfg["marker"],
        )

    return img


def add_hands_to_clock(clean_clock: Image.Image, hour: int, minute: int, second: int,
                       style_id: int) -> Image.Image:
    """Draw hour, minute and second hands on a clean analog clock."""
    img = clean_clock.copy()
    draw = ImageDraw.Draw(img)
    cfg = analog_style_config(style_id)

    image_size = img.size[0]
    center = (image_size // 2, image_size // 2)
    radius = image_size // 2 - 26

    hour_angle, minute_angle, second_angle = time_to_angles(hour, minute, second)

    hour_end = angle_to_point(center, radius * 0.50, hour_angle)
    minute_end = angle_to_point(center, radius * 0.75, minute_angle)
    second_end = angle_to_point(center, radius * 0.85, second_angle)

    # Dynamic hand colors based on style keywords
    style_name = cfg["name"]
    dark_bg_keywords = ["dark", "navy", "space", "cyberpunk", "monochrome_dark", "ocean", "mahogany"]
    colorful_keywords = ["teacher_yellow", "pop_art", "kid"]

    if any(kw in style_name for kw in dark_bg_keywords):
        hour_color = (230, 230, 230)
        minute_color = (230, 230, 230)
        second_color = (230, 50, 45)
    elif "neon" in style_name:
        hour_color = (255, 255, 255)
        minute_color = (255, 255, 255)
        second_color = cfg["outline"] # Match the neon outline
    elif any(kw in style_name for kw in colorful_keywords):
        hour_color = (30, 80, 180)
        minute_color = (35, 160, 90)
        second_color = (220, 30, 30)
    elif "vintage" in style_name or "steampunk" in style_name or "retro" in style_name:
        hour_color = (40, 20, 10)
        minute_color = (40, 20, 10)
        second_color = (130, 30, 20)
    else:
        hour_color = (10, 10, 10)
        minute_color = (15, 15, 15)
        second_color = (205, 40, 35)

    draw.line([center, hour_end], fill=hour_color, width=7)
    draw.line([center, minute_end], fill=minute_color, width=4)
    draw.line([center, second_end], fill=second_color, width=2)

    tail = angle_to_point(center, radius * 0.18, (second_angle + 180) % 360)
    draw.line([center, tail], fill=second_color, width=1)

    cap_radius = 6
    draw.ellipse(
        [
            center[0] - cap_radius,
            center[1] - cap_radius,
            center[0] + cap_radius,
            center[1] + cap_radius,
        ],
        fill=hour_color,
        outline=(180, 150, 80) if "steampunk" not in style_name else (80, 40, 10),
        width=2,
    )

    return img


# =========================
# Split logic
# =========================

def choose_split(analog_style_id: int, train_style_ids: List[int], test_style_ids: List[int]) -> str:
    if analog_style_id in train_style_ids:
        return "train"
    if analog_style_id in test_style_ids:
        return "test"
    raise ValueError(f"Style {analog_style_id} was not assigned to train or test")


def get_style_split() -> Tuple[List[int], List[int]]:
    """
    Split by analog style.
    This tests whether the model generalizes to unseen analog clock designs.
    """
    style_ids = list(range(NUM_ANALOG_STYLES))
    random.shuffle(style_ids)

    split_idx = int(TRAIN_RATIO * len(style_ids))
    train_style_ids = style_ids[:split_idx]
    test_style_ids = style_ids[split_idx:]

    return train_style_ids, test_style_ids


# =========================
# Dataset generation
# =========================

def generate_dataset(num_examples: int = DATASET_SIZE) -> None:
    random.seed(SEED)
    reset_output_dir()

    all_rows = []
    train_rows = []
    test_rows = []

    train_style_ids, test_style_ids = get_style_split()

    for idx in range(num_examples):
        hour, minute, second = random_time()

        analog_style_id = idx % NUM_ANALOG_STYLES

        if SPLIT_BY_STYLE:
            split = choose_split(analog_style_id, train_style_ids, test_style_ids)
        else:
            split = "train" if random.random() < TRAIN_RATIO else "test"

        hour_angle, minute_angle, second_angle = time_to_angles(hour, minute, second)
        sample_id = f"sample_{idx:04d}"

        clean_analog_img = create_analog_clean(analog_style_id)
        analog_with_hands_img = add_hands_to_clock(clean_analog_img, hour, minute, second, analog_style_id)

        analog_path = OUTPUT_DIR / split / "analog_clock" / f"{sample_id}.png"
        clean_path = OUTPUT_DIR / split / "analog_without_hands" / f"{sample_id}.png"

        analog_with_hands_img.save(analog_path)
        clean_analog_img.save(clean_path)

        row = {
            "id": sample_id,
            "split": split,
            "hour": hour,
            "minute": minute,
            "second": second,
            "hour_angle": round(hour_angle, 4),
            "minute_angle": round(minute_angle, 4),
            "second_angle": round(second_angle, 4),
            "analog_style_id": analog_style_id,
            "analog_path": str(analog_path),
            "analog_without_hands_path": str(clean_path),
        }

        all_rows.append(row)
        if split == "train":
            train_rows.append(row)
        else:
            test_rows.append(row)

    write_csv(LABELS_PATH, all_rows)
    write_csv(TRAIN_LABELS_PATH, train_rows)
    write_csv(TEST_LABELS_PATH, test_rows)

    print(f"Generated {num_examples} paired samples (Analog Only)")
    print(f"Train samples: {len(train_rows)}")
    print(f"Test samples: {len(test_rows)}")
    print(f"Train analog styles: {sorted(train_style_ids)}")
    print(f"Test analog styles: {sorted(test_style_ids)}")
    print(f"Output directory: {OUTPUT_DIR}")


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write into {path}")

    with open(path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    generate_dataset()