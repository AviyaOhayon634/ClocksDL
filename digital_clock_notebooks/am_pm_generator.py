import os
import random
import shutil
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# ========= CONFIG =========
SOURCE_PATH = "new_merged_dataset/valid"  # הנתיב לקבצים המקוריים
OUT_PATH = "new_merged_with_ampm/valid"

SPLIT = "valid"  # או "valid"

if SPLIT == "train":
    ADD_AMPM_PROB = 0.4
    ADD_NOISE_TEXT_PROB = 0.1
    BLUR_PROB = 0.3
else:  # valid
    ADD_AMPM_PROB = 0.1
    ADD_NOISE_TEXT_PROB = 0.1
    BLUR_PROB = 0.3

IMG_DIR = os.path.join(SOURCE_PATH, "images")
LBL_DIR = os.path.join(SOURCE_PATH, "labels")

OUT_IMG = os.path.join(OUT_PATH, "images")
OUT_LBL = os.path.join(OUT_PATH, "labels")

os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(OUT_LBL, exist_ok=True)

FONTS = [
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "fonts/ledsled.ttf",
    "fonts/ledsled3d.ttf",
]

AM_CLASS = 10
PM_CLASS = 11
MARGIN = 10

# 🔥 טקסטים לא רלוונטיים
DISTRACTOR_TEXTS = [
    "DAY", "TEMP", "DST", "MON",
    "MINUTE", "HOUR", "SAT", "WED", "TUE", "SEC",
]


# ========= HELPERS =========
def yolo_to_pixel(box, img_w, img_h):
    xc, yc, w, h = box
    x1 = (xc - w/2) * img_w
    y1 = (yc - h/2) * img_h
    x2 = (xc + w/2) * img_w
    y2 = (yc + h/2) * img_h
    return (x1, y1, x2, y2)


def pixel_to_yolo(x1, y1, x2, y2, img_w, img_h):
    xc = (x1 + x2) / 2 / img_w
    yc = (y1 + y2) / 2 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return xc, yc, w, h


def overlaps(a, b):
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


def find_random_safe_position(existing_boxes, text_w, text_h, img_w, img_h):
    for _ in range(50):
        x = random.randint(MARGIN, img_w - text_w - MARGIN)
        y = random.randint(MARGIN, img_h - text_h - MARGIN)

        new_box = (x, y, x + text_w, y + text_h)

        if all(not overlaps(new_box, b) for b in existing_boxes):
            return x, y, new_box

    return None, None, None


# ========= PROCESS =========
files = [f for f in os.listdir(IMG_DIR) if f.endswith((".jpg", ".png"))]

print(f"Processing {len(files)} images...")

for file in files:

    img_path = os.path.join(IMG_DIR, file)
    lbl_name = file.replace(".jpg", ".txt").replace(".png", ".txt")
    lbl_path = os.path.join(LBL_DIR, lbl_name)

    if not os.path.exists(lbl_path):
        continue

    shutil.copy(img_path, os.path.join(OUT_IMG, file))
    shutil.copy(lbl_path, os.path.join(OUT_LBL, lbl_name))

    img = Image.open(os.path.join(OUT_IMG, file)).convert("RGBA")
    img_w, img_h = img.size

    # ===== READ EXISTING BOXES =====
    existing_boxes = []
    with open(os.path.join(OUT_LBL, lbl_name), "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        _, xc, yc, w, h = map(float, parts)
        existing_boxes.append(yolo_to_pixel((xc, yc, w, h), img_w, img_h))

    # ===== החלטה מה להוסיף =====
    r = random.random()

    if r < ADD_AMPM_PROB:
        mode = "ampm"
    elif r < ADD_AMPM_PROB + ADD_NOISE_TEXT_PROB:
        mode = "noise"
    else:
        img.convert("RGB").save(os.path.join(OUT_IMG, file))
        continue

    # ===== טקסט =====
    if mode == "ampm":
        label_text = random.choice(["AM", "PM"])
        class_id = AM_CLASS if label_text == "AM" else PM_CLASS
    else:
        label_text = random.choice(DISTRACTOR_TEXTS)
        class_id = None  # ❌ לא מוסיפים label

    # ===== גודל =====
    avg_height = sum((b[3]-b[1]) for b in existing_boxes) / len(existing_boxes)
    font_size = int(avg_height * random.uniform(0.5, 0.7))
    font_size = min(font_size, int(img_h * 0.15))
    font_size = max(font_size, 15)

    font = ImageFont.truetype(random.choice(FONTS), font_size)

    dummy_draw = ImageDraw.Draw(Image.new("RGB", (1,1)))
    temp_bbox = dummy_draw.textbbox((0,0), label_text, font=font)
    text_w = temp_bbox[2] - temp_bbox[0]
    text_h = temp_bbox[3] - temp_bbox[1]

    x, y, new_box = find_random_safe_position(
        existing_boxes, text_w, text_h, img_w, img_h
    )

    if new_box is None:
        img.convert("RGB").save(os.path.join(OUT_IMG, file))
        continue

    # ===== צבעים =====
    COLORS = [
        (255,255,255),
        (0,255,0),
        (255,0,0),
        (0,200,255),
        (200,200,200)
    ]

    text_color = random.choice(COLORS)
    outline_color = (0,0,0) if sum(text_color) > 400 else (255,255,255)

    text_layer = Image.new("RGBA", img.size, (0,0,0,0))
    draw = ImageDraw.Draw(text_layer)

    for dx in [-1,0,1]:
        for dy in [-1,0,1]:
            if dx != 0 or dy != 0:
                draw.text((x+dx, y+dy), label_text, fill=outline_color, font=font)

    draw.text((x,y), label_text, fill=text_color, font=font)

    if random.random() < BLUR_PROB:
        text_layer = text_layer.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5,1.2)))

    img = Image.alpha_composite(img, text_layer)

    # ===== רק אם AM/PM מוסיפים label =====
    if class_id is not None:
        xc, yc, bw, bh = pixel_to_yolo(*new_box, img_w, img_h)

        with open(os.path.join(OUT_LBL, lbl_name), "a") as f:
            f.write(f"\n{class_id} {xc} {yc} {bw} {bh}")

    img.convert("RGB").save(os.path.join(OUT_IMG, file))

print("✅ DONE - AM/PM + DISTRACTOR DATASET CREATED")