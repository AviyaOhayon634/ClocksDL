"""
data_generator_analog_only.py

Synthetic data generator for ClocksDL (Analog Only).
Includes dynamic layouts to prevent overlap, realistic shadows, gradients,
multiple fonts, and complex toy clocks (like the yellow learning clock).
"""

import csv
import math
import random
import shutil
import os
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont, ImageFilter

# =========================
# Configuration
# =========================

DATASET_SIZE = 25000
IMAGE_SIZE = 512
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
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    for split in ["train", "test"]:
        for subfolder in SUBFOLDERS:
            (OUTPUT_DIR / split / subfolder).mkdir(parents=True, exist_ok=True)

def get_font(size: int, bold: bool = True) -> ImageFont.ImageFont:
    """Load a custom font randomly from a predefined list of fonts."""
    font_options = [
        "Roboto-VariableFont_wdth,wght.ttf",
        "Boldonse-Regular.ttf",
        "arialbd.ttf",
        "Oswald-VariableFont_wght.ttf"
    ]
    font_filename = random.choice(font_options)
    
    try:
        return ImageFont.truetype(font_filename, size)
    except OSError:
        # Fallback to default if no TTF files are found in the directory
        return ImageFont.load_default()

def draw_centered_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont,
                       center: Tuple[int, int], fill) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    draw.text((center[0] - width / 2, center[1] - height / 2), text, font=font, fill=fill)

def angle_to_point(center: Tuple[int, int], length: float, angle_degrees: float) -> Tuple[int, int]:
    angle_rad = math.radians(angle_degrees - 90)
    x = center[0] + length * math.cos(angle_rad)
    y = center[1] + length * math.sin(angle_rad)
    return int(round(x)), int(round(y))

def time_to_angles(hour: int, minute: int, second: int) -> Tuple[float, float, float]:
    hour_12 = hour % 12
    second_angle = second * 6.0
    minute_angle = minute * 6.0 + second * 0.1
    hour_angle = hour_12 * 30.0 + minute * 0.5 + second / 120.0
    return hour_angle, minute_angle, second_angle

def random_time() -> Tuple[int, int, int]:
    return random.randint(0, 23), random.randint(0, 59), random.randint(0, 59)

def add_sensor_noise(img: Image.Image, amount: int = 5) -> Image.Image:
    pixels = img.load()
    width, height = img.size
    for _ in range((width * height) // 80):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        r, g, b = pixels[x, y]
        delta = random.randint(-amount, amount)
        pixels[x, y] = (max(0, min(255, r+delta)), max(0, min(255, g+delta)), max(0, min(255, b+delta)))
    return img

def get_hand_polygon(center: Tuple[int, int], length: float, angle_degrees: float, 
                     base_width: float, tip_width: float, tail_length: float = 0) -> List[Tuple[float, float]]:
    angle_rad = math.radians(angle_degrees - 90)
    perp_rad = angle_rad + math.pi / 2

    tip_x = center[0] + length * math.cos(angle_rad)
    tip_y = center[1] + length * math.sin(angle_rad)
    tip_p1 = (tip_x + (tip_width / 2) * math.cos(perp_rad), tip_y + (tip_width / 2) * math.sin(perp_rad))
    tip_p2 = (tip_x - (tip_width / 2) * math.cos(perp_rad), tip_y - (tip_width / 2) * math.sin(perp_rad))

    base_x = center[0] - tail_length * math.cos(angle_rad)
    base_y = center[1] - tail_length * math.sin(angle_rad)
    base_p1 = (base_x + (base_width / 2) * math.cos(perp_rad), base_y + (base_width / 2) * math.sin(perp_rad))
    base_p2 = (base_x - (base_width / 2) * math.cos(perp_rad), base_y - (base_width / 2) * math.sin(perp_rad))

    return [base_p1, tip_p1, tip_p2, base_p2]

def draw_toy_hand(draw: ImageDraw.ImageDraw, center: Tuple[int, int], length: float, angle_degrees: float, color):
    """Draws a specific chunky toy hand with a bulbous tip (like the yellow learning clock)."""
    angle_rad = math.radians(angle_degrees - 90)
    
    # Main shaft
    base_width = 16
    shaft_end_length = length * 0.75
    shaft_poly = get_hand_polygon(center, shaft_end_length, angle_degrees, base_width, 8, tail_length=15)
    draw.polygon(shaft_poly, fill=color)
    
    # Bulbous part near tip
    bulb_center = angle_to_point(center, length * 0.75, angle_degrees)
    bulb_radius = 8
    draw.ellipse([bulb_center[0]-bulb_radius, bulb_center[1]-bulb_radius, bulb_center[0]+bulb_radius, bulb_center[1]+bulb_radius], fill=color)
    
    # Pointy tip
    tip_poly = get_hand_polygon(bulb_center, length * 0.25, angle_degrees, bulb_radius * 1.5, 1, tail_length=0)
    draw.polygon(tip_poly, fill=color)


# =========================
# Analog clocks Styles
# =========================

def analog_style_config(style_id: int) -> Dict:
    styles = [
        # --- THE SPECIFIC TOY CLOCK FROM YOUR IMAGE ---
        {"name": "exact_teacher_yellow", "bg": (250, 250, 250), "face": (255, 230, 0), "inner": None, "outline": (240, 200, 0), "marker": (0, 100, 220), "numbers": True, "dots": True, "rim": 5, "complex_numbers": True},
        
        # --- ORIGINAL STYLES ---
        {"name": "teacher_blue_learning", "bg": (250, 250, 248), "face": (145, 215, 235), "inner": (230, 245, 170), "outline": (170, 155, 120), "marker": (40, 90, 110), "numbers": True, "dots": True, "rim": 5},
        {"name": "classic_white_numbers", "bg": (248, 248, 245), "face": (255, 255, 252), "inner": None, "outline": (25, 25, 25), "marker": (10, 10, 10), "numbers": True, "dots": False, "rim": 5},
        {"name": "black_white_numbers", "bg": (245, 245, 245), "face": (255, 255, 255), "inner": None, "outline": (15, 15, 15), "marker": (5, 5, 5), "numbers": True, "dots": False, "rim": 8},
        {"name": "minimal_white", "bg": (245, 245, 242), "face": (252, 252, 248), "inner": None, "outline": (60, 60, 60), "marker": (50, 50, 50), "numbers": False, "dots": False, "rim": 3},
        {"name": "brown_dot_minimal", "bg": (174, 174, 168), "face": (128, 65, 45), "inner": None, "outline": (128, 65, 45), "marker": (195, 195, 185), "numbers": False, "dots": True, "rim": 0},
        {"name": "cream_gold", "bg": (236, 228, 210), "face": (252, 246, 226), "inner": None, "outline": (180, 145, 70), "marker": (160, 120, 60), "numbers": True, "dots": False, "rim": 4},
        {"name": "dark_luxury", "bg": (30, 30, 30), "face": (18, 18, 20), "inner": None, "outline": (210, 170, 80), "marker": (220, 190, 100), "numbers": False, "dots": False, "rim": 4},
        {"name": "blue_clean", "bg": (224, 236, 246), "face": (232, 244, 255), "inner": None, "outline": (50, 100, 150), "marker": (50, 100, 150), "numbers": True, "dots": False, "rim": 3},
        {"name": "gray_industrial", "bg": (210, 210, 210), "face": (235, 235, 232), "inner": None, "outline": (70, 70, 70), "marker": (30, 30, 30), "numbers": False, "dots": True, "rim": 6},
        {"name": "pastel_kids", "bg": (245, 236, 240), "face": (255, 245, 250), "inner": None, "outline": (210, 120, 160), "marker": (120, 80, 120), "numbers": True, "dots": True, "rim": 4},
        {"name": "navy_minimal", "bg": (225, 230, 238), "face": (25, 35, 55), "inner": None, "outline": (230, 230, 220), "marker": (230, 230, 220), "numbers": False, "dots": False, "rim": 4},
        {"name": "wooden_oak", "bg": (220, 210, 200), "face": (180, 130, 80), "inner": None, "outline": (100, 60, 30), "marker": (60, 30, 10), "numbers": True, "dots": False, "rim": 6},
        {"name": "neon_pink", "bg": (10, 10, 15), "face": (20, 20, 25), "inner": None, "outline": (255, 20, 147), "marker": (255, 105, 180), "numbers": True, "dots": False, "rim": 3},
        {"name": "steampunk_copper", "bg": (180, 170, 160), "face": (184, 115, 51), "inner": (140, 70, 20), "outline": (90, 40, 10), "marker": (50, 20, 5), "numbers": True, "dots": True, "rim": 5},
        {"name": "vintage_parchment", "bg": (245, 240, 230), "face": (238, 224, 193), "inner": None, "outline": (120, 90, 50), "marker": (70, 40, 20), "numbers": True, "dots": False, "rim": 3},
        {"name": "pop_art_yellow", "bg": (200, 230, 255), "face": (255, 220, 0), "inner": None, "outline": (0, 0, 255), "marker": (255, 0, 0), "numbers": True, "dots": True, "rim": 6},
        {"name": "ocean_deep", "bg": (200, 220, 240), "face": (10, 60, 100), "inner": None, "outline": (0, 30, 60), "marker": (150, 200, 255), "numbers": False, "dots": True, "rim": 5},
        {"name": "retro_70s", "bg": (240, 220, 190), "face": (150, 80, 30), "inner": (200, 120, 40), "outline": (100, 40, 10), "marker": (255, 220, 150), "numbers": True, "dots": True, "rim": 6},
        {"name": "monochrome_dark", "bg": (50, 50, 50), "face": (80, 80, 80), "inner": None, "outline": (20, 20, 20), "marker": (200, 200, 200), "numbers": False, "dots": True, "rim": 6},
    ]
    # Multiply the styles to reach 40 (or just return modulus)
    return styles[style_id % len(styles)]

def create_analog_clean(style_id: int, image_size: int = IMAGE_SIZE) -> Image.Image:
    cfg = analog_style_config(style_id)
    img = Image.new("RGB", (image_size, image_size), color=cfg["bg"])
    draw = ImageDraw.Draw(img)

    center = (image_size // 2, image_size // 2)
    radius = image_size // 2 - random.randint(12, 18)

    # Shadow
    shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    shadow_offset = random.randint(4, 8)
    shadow_draw.ellipse([center[0] - radius + shadow_offset, center[1] - radius + shadow_offset,
                         center[0] + radius + shadow_offset, center[1] + radius + shadow_offset], fill=(0, 0, 0, 45))
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=8))
    img = Image.alpha_composite(img.convert("RGBA"), shadow).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Face and Gradient
    face_color = cfg["face"]
    rim_width = max(3, cfg.get("rim", 4) + random.randint(0, 4))
    
    if random.random() < 0.3 and not cfg.get("complex_numbers"): 
        for r in range(radius, 0, -1):
            factor = r / radius
            grad_color = (
                int(face_color[0] * (1 - factor) + min(255, face_color[0] + 50) * factor),
                int(face_color[1] * (1 - factor) + min(255, face_color[1] + 50) * factor),
                int(face_color[2] * (1 - factor) + min(255, face_color[2] + 50) * factor)
            )
            draw.ellipse([center[0] - r, center[1] - r, center[0] + r, center[1] + r], fill=grad_color)
        draw.ellipse([center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius], outline=cfg["outline"], width=rim_width)
    else:
        draw.ellipse([center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius], fill=cfg["face"], outline=cfg["outline"], width=rim_width)

    # ====== SPECIAL COMPLEX TOY CLOCK LOGIC ======
    if cfg.get("complex_numbers"):
        # 1. Outer Blue Dots & Numbers (Minutes)
        font_min = get_font(12, bold=True)
        for i in range(60):
            angle = i * 6
            if i % 5 == 0:
                pos = angle_to_point(center, radius - 15, angle)
                min_str = str(i) if i != 0 else "60"
                draw_centered_text(draw, min_str, font_min, pos, (0, 100, 220))
            else:
                point = angle_to_point(center, radius - 6, angle)
                draw.ellipse([point[0]-2, point[1]-2, point[0]+2, point[1]+2], fill=(0, 100, 220))
        
        # 2. Main Red Numbers (1-12)
        font_main = get_font(40, bold=True)
        for i in range(1, 13):
            pos = angle_to_point(center, radius - 45, i * 30)
            draw_centered_text(draw, str(i), font_main, pos, (230, 30, 30))
            
        # 3. Inner Red Numbers (13-24)
        font_inner = get_font(14, bold=True)
        for i in range(13, 25):
            angle = (i - 12) * 30
            pos = angle_to_point(center, radius - 75, angle)
            draw_centered_text(draw, str(i), font_inner, pos, (230, 30, 30))
            
        return img

    # ====== REGULAR SMART LAYOUT LOGIC ======
    layout_style = random.choice(["numbers_outer", "numbers_inner"])

    if cfg["numbers"]:
        if layout_style == "numbers_outer":
            base_font_size = random.randint(35, 45) 
            number_radius = radius - 35
            tick_outer_radius = radius - 70 
            tick_inner_radius_5min = tick_outer_radius - 12
            tick_inner_radius_1min = tick_outer_radius - 5
        else:
            base_font_size = random.randint(30, 40)
            tick_outer_radius = radius - 5
            tick_inner_radius_5min = tick_outer_radius - 15
            tick_inner_radius_1min = tick_outer_radius - 5
            number_radius = radius - 45
    else:
        tick_outer_radius = radius - 8
        tick_inner_radius_5min = tick_outer_radius - 15
        tick_inner_radius_1min = tick_outer_radius - 5
        number_radius = 0

    # Draw Ticks
    for i in range(60):
        angle = i * 6
        is_5_min = (i % 5 == 0)
        
        # Overlap prevention trick
        skip_tick = False
        if cfg["numbers"] and is_5_min and layout_style == "numbers_inner":
            skip_tick = True

        if not skip_tick:
            if cfg["dots"] and is_5_min:
                point = angle_to_point(center, tick_outer_radius, angle)
                dot_radius = 4 if i % 15 == 0 else 3
                draw.ellipse([point[0]-dot_radius, point[1]-dot_radius, point[0]+dot_radius, point[1]+dot_radius], fill=cfg["marker"])
            elif not cfg["dots"]:
                outer = angle_to_point(center, tick_outer_radius, angle)
                inner = angle_to_point(center, tick_inner_radius_5min if is_5_min else tick_inner_radius_1min, angle)
                width = 3 if is_5_min else 1
                draw.line([inner, outer], fill=cfg["marker"], width=width)

    # Draw Numbers
    if cfg["numbers"]:
        font = get_font(base_font_size, bold=True)
        for number in range(1, 13):
            angle = number * 30
            position = angle_to_point(center, number_radius, angle)
            draw_centered_text(draw, str(number), font, position, cfg["marker"])

    # Brand text
    if random.random() < 0.2:
        brand_font = get_font(10, bold=False)
        brand_y_pos = center[1] + (radius * 0.45)
        draw_centered_text(draw, random.choice(["QUARTZ", "CLASSIC", "TIME"]), brand_font, (center[0], int(brand_y_pos)), cfg["marker"])

    return img

def get_diamond_hand_polygon(center: Tuple[int, int], length: float, angle_degrees: float, 
                             max_width: float) -> List[Tuple[float, float]]:
    """מייצר מחוג בצורת מעוין/יהלום, כמו בשעון הכחול"""
    angle_rad = math.radians(angle_degrees - 90)
    perp_rad = angle_rad + math.pi / 2

    # קצה המחוג (השפיץ)
    tip = (center[0] + length * math.cos(angle_rad), center[1] + length * math.sin(angle_rad))
    
    # הנקודה הרחבה ביותר (בערך בשליש הדרך)
    mid_dist = length * 0.25
    mid_x = center[0] + mid_dist * math.cos(angle_rad)
    mid_y = center[1] + mid_dist * math.sin(angle_rad)
    
    mid_p1 = (mid_x + (max_width / 2) * math.cos(perp_rad), mid_y + (max_width / 2) * math.sin(perp_rad))
    mid_p2 = (mid_x - (max_width / 2) * math.cos(perp_rad), mid_y - (max_width / 2) * math.sin(perp_rad))
    
    # הזנב (קצת מאחורי המרכז)
    tail = (center[0] - (length * 0.15) * math.cos(angle_rad), center[1] - (length * 0.15) * math.sin(angle_rad))
    
    return [tail, mid_p1, tip, mid_p2]

def add_hands_to_clock(clean_clock: Image.Image, hour: int, minute: int, second: int,
                       style_id: int) -> Image.Image:
    img = clean_clock.copy()
    draw = ImageDraw.Draw(img)
    cfg = analog_style_config(style_id)

    image_size = img.size[0]
    center = (image_size // 2, image_size // 2)
    radius = image_size // 2 - 26

    hour_angle, minute_angle, second_angle = time_to_angles(hour, minute, second)

    # --- יצירת מגוון צבעי מחוגים (כולל זהב, לבן וכסף) ---
    color_palettes = [
        {"h": (10, 10, 10), "m": (15, 15, 15), "s": (205, 40, 35)}, # קלאסי (שחור ואדום)
        {"h": (218, 165, 32), "m": (218, 165, 32), "s": (218, 165, 32)}, # זהב! (לשעון הכחול)
        {"h": (240, 240, 240), "m": (240, 240, 240), "s": (200, 200, 200)}, # לבן/כסוף
        {"h": (40, 40, 40), "m": (40, 40, 40), "s": (40, 40, 40)} # הכל שחור
    ]
    
    if cfg.get("complex_numbers"):
        hour_color, minute_color, second_color = (230, 30, 30), (0, 120, 240), (0, 0, 0, 0)
        hand_style_type = "toy_arrow"
    else:
        palette = random.choice(color_palettes)
        hour_color, minute_color, second_color = palette["h"], palette["m"], palette["s"]
        
        # סגנונות חדשים: כולל ענק (massive) ויהלום (diamond)
        hand_style_type = random.choice(["straight", "tapered", "chunky", "massive", "diamond"])

    h_length = radius * 0.50
    m_length = radius * 0.75
    s_length = radius * 0.85

    # 1. ציור הצלליות (מודגשות יותר כדי לעזור למודל עם הקצוות)
    shadow_offset = random.choice([3, 5])
    shadow_color = (0, 0, 0, random.randint(60, 100))
    
    if hand_style_type == "toy_arrow":
        draw_toy_hand(draw, (center[0]+shadow_offset, center[1]+shadow_offset), h_length, hour_angle, shadow_color)
        draw_toy_hand(draw, (center[0]+shadow_offset, center[1]+shadow_offset), m_length, minute_angle, shadow_color)
    else:
        if hand_style_type == "massive":
            h_poly = get_hand_polygon(center, h_length, hour_angle, 24, 18, 20) # עובי מטורף לשעון הלבן
            m_poly = get_hand_polygon(center, m_length, minute_angle, 18, 12, 25)
        elif hand_style_type == "diamond":
            h_poly = get_diamond_hand_polygon(center, h_length, hour_angle, 16) # יהלום לשעון הכחול
            m_poly = get_diamond_hand_polygon(center, m_length, minute_angle, 12)
        elif hand_style_type == "chunky":
            h_poly = get_hand_polygon(center, h_length, hour_angle, 16, 8, 15)
            m_poly = get_hand_polygon(center, m_length, minute_angle, 12, 6, 20)
        elif hand_style_type == "tapered":
            h_poly = get_hand_polygon(center, h_length, hour_angle, 12, 1, 5)
            m_poly = get_hand_polygon(center, m_length, minute_angle, 8, 1, 10)
        else:
            h_poly = get_hand_polygon(center, h_length, hour_angle, 8, 8, 10)
            m_poly = get_hand_polygon(center, m_length, minute_angle, 5, 5, 15)
        
        draw.polygon([(x+shadow_offset, y+shadow_offset) for x,y in h_poly], fill=shadow_color)
        draw.polygon([(x+shadow_offset, y+shadow_offset) for x,y in m_poly], fill=shadow_color)

    # 2. ציור המחוגים עצמם
    if hand_style_type == "toy_arrow":
        draw_toy_hand(draw, center, h_length, hour_angle, hour_color)
        draw_toy_hand(draw, center, m_length, minute_angle, minute_color)
        cap_radius = 12
        draw.ellipse([center[0]-cap_radius, center[1]-cap_radius, center[0]+cap_radius, center[1]+cap_radius], fill=minute_color)
    else:
        draw.polygon(h_poly, fill=hour_color)
        draw.polygon(m_poly, fill=minute_color)
        
        # מחוג שניות (לא לכל השעונים יש)
        if random.random() > 0.3:
            s_poly = get_hand_polygon(center, s_length, second_angle, 3, 1, 25)
            draw.polygon(s_poly, fill=second_color)
        
        # פקק מרכזי (קצת יותר גדול כדי לכסות הכל)
        cap_radius = 10 if hand_style_type in ["chunky", "massive"] else 6
        draw.ellipse([center[0]-cap_radius, center[1]-cap_radius, center[0]+cap_radius, center[1]+cap_radius], 
                     fill=hour_color, outline=(150, 150, 150), width=1)

    return img

# =========================
# Split & Generation Logic
# =========================

def get_style_split() -> Tuple[List[int], List[int]]:
    style_ids = list(range(NUM_ANALOG_STYLES))
    random.shuffle(style_ids)
    split_idx = int(TRAIN_RATIO * len(style_ids))
    return style_ids[:split_idx], style_ids[split_idx:]

def generate_dataset(num_examples: int = DATASET_SIZE) -> None:
    random.seed(SEED)
    reset_output_dir()

    all_rows, train_rows, test_rows = [], [], []
    train_style_ids, test_style_ids = get_style_split()

    print(f"Starting generation of {num_examples} clocks...")

    for idx in range(num_examples):
        hour, minute, second = random_time()
        analog_style_id = idx % NUM_ANALOG_STYLES
        split = "train" if analog_style_id in train_style_ids else "test"

        hour_angle, minute_angle, second_angle = time_to_angles(hour, minute, second)
        sample_id = f"sample_{idx:05d}"

        clean_analog_img = create_analog_clean(analog_style_id)
        analog_with_hands_img = add_hands_to_clock(clean_analog_img, hour, minute, second, analog_style_id)
        
        # Add realistic sensor noise
        analog_with_hands_img = add_sensor_noise(analog_with_hands_img)

        analog_path = OUTPUT_DIR / split / "analog_clock" / f"{sample_id}.png"
        clean_path = OUTPUT_DIR / split / "analog_without_hands" / f"{sample_id}.png"

        analog_with_hands_img.save(analog_path)
        clean_analog_img.save(clean_path)

        row = {
            "id": sample_id,
            "split": split,
            "hour": hour, "minute": minute, "second": second,
            "hour_angle": round(hour_angle, 4), "minute_angle": round(minute_angle, 4), "second_angle": round(second_angle, 4),
            "analog_style_id": analog_style_id,
            "analog_path": str(analog_path), "analog_without_hands_path": str(clean_path),
        }

        all_rows.append(row)
        if split == "train": train_rows.append(row)
        else: test_rows.append(row)

        # === Progress Printing ===
        if (idx + 1) % 1000 == 0:
            print(f"Progress: Generated {idx + 1} / {num_examples} clocks...")

    write_csv(LABELS_PATH, all_rows)
    write_csv(TRAIN_LABELS_PATH, train_rows)
    write_csv(TEST_LABELS_PATH, test_rows)

    print(f"Done! Generated {num_examples} paired samples.")
    print(f"Train samples: {len(train_rows)} | Test samples: {len(test_rows)}")

def write_csv(path: Path, rows: List[Dict]) -> None:
    with open(path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

if __name__ == "__main__":
    generate_dataset()