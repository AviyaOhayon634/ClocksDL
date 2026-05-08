"""
Synthetic data generator for ClocksDL — Analog Only.
Designed to closely match real-world analog clock photos from the internet.
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
DATASET_SIZE = 27000
IMAGE_SIZE   = 512
OUTPUT_DIR   = Path("data")
SEED         = 42

TRAIN_RATIO      = 0.8
NUM_ANALOG_STYLES = 50

SUBFOLDERS = ["analog_clock", "analog_without_hands"]

LABELS_PATH       = OUTPUT_DIR / "labels.csv"
TRAIN_LABELS_PATH = OUTPUT_DIR / "train_labels.csv"
TEST_LABELS_PATH  = OUTPUT_DIR / "test_labels.csv"

ROMAN = ["I","II","III","IV","V","VI","VII","VIII","IX","X","XI","XII"]

# =========================
# General utilities
# =========================
def reset_output_dir() -> None:
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    for split in ["train", "test"]:
        for sub in SUBFOLDERS:
            (OUTPUT_DIR / split / sub).mkdir(parents=True, exist_ok=True)

def get_font(size: int) -> ImageFont.ImageFont:
    font_options = [
        "Roboto-VariableFont_wdth,wght.ttf",
        "Boldonse-Regular.ttf",
        "arialbd.ttf",
        "Oswald-VariableFont_wght.ttf",
    ]
    for name in random.sample(font_options, len(font_options)):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()

def draw_centered_text(draw, text, font, center, fill):
    bbox = draw.textbbox((0, 0), text, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text((center[0] - w / 2, center[1] - h / 2), text, font=font, fill=fill)

def angle_to_point(center, length, angle_deg):
    rad = math.radians(angle_deg - 90)
    return (int(round(center[0] + length * math.cos(rad))),
            int(round(center[1] + length * math.sin(rad))))

def time_to_angles(h, m, s):
    h12 = h % 12
    return (h12 * 30.0 + m * 0.5 + s / 120.0,
            m * 6.0 + s * 0.1,
            s * 6.0)

def random_time():
    return random.randint(0, 23), random.randint(0, 59), random.randint(0, 59)

def add_sensor_noise(img: Image.Image, amount: int = 4) -> Image.Image:
    pixels = img.load()
    w, h = img.size
    for _ in range((w * h) // 100):
        x, y = random.randint(0, w-1), random.randint(0, h-1)
        r, g, b = pixels[x, y]
        d = random.randint(-amount, amount)
        pixels[x, y] = (max(0,min(255,r+d)), max(0,min(255,g+d)), max(0,min(255,b+d)))
    return img

# =========================
# Hand polygon helpers
# =========================
def get_hand_polygon(center, length, angle_deg, base_w, tip_w, tail=0):
    rad  = math.radians(angle_deg - 90)
    perp = rad + math.pi / 2
    tx = center[0] + length * math.cos(rad)
    ty = center[1] + length * math.sin(rad)
    bx = center[0] - tail * math.cos(rad)
    by = center[1] - tail * math.sin(rad)
    return [
        (bx + base_w/2 * math.cos(perp), by + base_w/2 * math.sin(perp)),
        (tx + tip_w/2  * math.cos(perp), ty + tip_w/2  * math.sin(perp)),
        (tx - tip_w/2  * math.cos(perp), ty - tip_w/2  * math.sin(perp)),
        (bx - base_w/2 * math.cos(perp), by - base_w/2 * math.sin(perp)),
    ]

def get_diamond_polygon(center, length, angle_deg, max_w):
    rad  = math.radians(angle_deg - 90)
    perp = rad + math.pi / 2
    tip  = (center[0] + length * math.cos(rad), center[1] + length * math.sin(rad))
    md   = length * 0.25
    mx, my = center[0] + md * math.cos(rad), center[1] + md * math.sin(rad)
    tail = (center[0] - length * 0.15 * math.cos(rad),
            center[1] - length * 0.15 * math.sin(rad))
    return [
        tail,
        (mx + max_w/2 * math.cos(perp), my + max_w/2 * math.sin(perp)),
        tip,
        (mx - max_w/2 * math.cos(perp), my - max_w/2 * math.sin(perp)),
    ]

def get_sword_polygon(center, length, angle_deg, max_w):
    """Wide near base, tapers to a long thin point — like classic clock hands."""
    rad  = math.radians(angle_deg - 90)
    perp = rad + math.pi / 2
    tip  = (center[0] + length * math.cos(rad), center[1] + length * math.sin(rad))
    # wide point at 30% from base
    mid_d = length * 0.30
    mx, my = center[0] + mid_d * math.cos(rad), center[1] + mid_d * math.sin(rad)
    tail_d = length * 0.15
    bx, by = center[0] - tail_d * math.cos(rad), center[1] - tail_d * math.sin(rad)
    return [
        (bx + max_w * 0.3 * math.cos(perp), by + max_w * 0.3 * math.sin(perp)),
        (mx + max_w / 2  * math.cos(perp), my + max_w / 2  * math.sin(perp)),
        tip,
        (mx - max_w / 2  * math.cos(perp), my - max_w / 2  * math.sin(perp)),
        (bx - max_w * 0.3 * math.cos(perp), by - max_w * 0.3 * math.sin(perp)),
    ]

def draw_toy_hand(draw, center, length, angle_deg, color):
    angle_rad = math.radians(angle_deg - 90)
    shaft = get_hand_polygon(center, length * 0.75, angle_deg, 16, 8, tail=15)
    draw.polygon(shaft, fill=color)
    bc = angle_to_point(center, length * 0.75, angle_deg)
    draw.ellipse([bc[0]-8, bc[1]-8, bc[0]+8, bc[1]+8], fill=color)
    tip = get_hand_polygon(bc, length * 0.25, angle_deg, 12, 1, tail=0)
    draw.polygon(tip, fill=color)

# =========================
# Clock styles  (50 styles)
# =========================
def analog_style_config(style_id: int) -> Dict:
    styles = [
        # ---- Realistic white/light face clocks (most common on internet) ----
        {"name": "modern_white_black",    "bg": (240,240,240), "face": (252,252,252), "outline": (20,20,20),      "marker": (15,15,15),     "numbers": True,  "roman": False, "dots": False, "rim": 6,  "rim_color": (20,20,20)},
        {"name": "modern_white_silver",   "bg": (220,225,230), "face": (250,250,252), "outline": (160,165,170),   "marker": (30,30,30),     "numbers": True,  "roman": False, "dots": False, "rim": 5,  "rim_color": (160,165,170)},
        {"name": "classic_roman",         "bg": (245,242,235), "face": (252,248,240), "outline": (60,50,30),      "marker": (50,40,25),     "numbers": True,  "roman": True,  "dots": False, "rim": 5,  "rim_color": (60,50,30)},
        {"name": "roman_gold_rim",        "bg": (240,235,220), "face": (255,250,235), "outline": (180,145,60),    "marker": (80,60,20),     "numbers": True,  "roman": True,  "dots": False, "rim": 7,  "rim_color": (180,145,60)},
        {"name": "minimal_ticks_only",    "bg": (245,245,245), "face": (255,255,255), "outline": (80,80,80),      "marker": (40,40,40),     "numbers": False, "roman": False, "dots": False, "rim": 4,  "rim_color": (80,80,80)},
        {"name": "minimal_dots",          "bg": (248,248,248), "face": (255,255,255), "outline": (100,100,100),   "marker": (60,60,60),     "numbers": False, "roman": False, "dots": True,  "rim": 3,  "rim_color": (100,100,100)},
        {"name": "silver_face",           "bg": (210,215,220), "face": (200,205,210), "outline": (120,125,130),   "marker": (40,40,40),     "numbers": True,  "roman": False, "dots": False, "rim": 5,  "rim_color": (130,135,140)},
        {"name": "light_blue",            "bg": (220,235,245), "face": (230,242,255), "outline": (60,100,140),    "marker": (40,80,120),    "numbers": True,  "roman": False, "dots": False, "rim": 4,  "rim_color": (60,100,140)},
        {"name": "cream_classic",         "bg": (242,235,218), "face": (252,245,228), "outline": (120,95,55),     "marker": (80,60,25),     "numbers": True,  "roman": False, "dots": False, "rim": 5,  "rim_color": (120,95,55)},
        {"name": "wood_brown",            "bg": (210,195,175), "face": (235,215,185), "outline": (100,65,30),     "marker": (60,35,10),     "numbers": True,  "roman": False, "dots": False, "rim": 8,  "rim_color": (100,65,30)},
        # ---- Dark face clocks ----
        {"name": "black_face_white_num",  "bg": (40,40,40),   "face": (18,18,22),   "outline": (200,200,200),   "marker": (210,210,210),  "numbers": True,  "roman": False, "dots": False, "rim": 5,  "rim_color": (180,180,180)},
        {"name": "black_gold",            "bg": (30,25,20),   "face": (15,12,10),   "outline": (210,170,70),    "marker": (220,185,90),   "numbers": False, "roman": False, "dots": False, "rim": 4,  "rim_color": (200,160,60)},
        {"name": "navy_white",            "bg": (20,30,55),   "face": (22,35,65),   "outline": (210,215,225),   "marker": (220,225,235),  "numbers": False, "roman": False, "dots": False, "rim": 5,  "rim_color": (180,185,200)},
        {"name": "dark_roman",            "bg": (35,30,25),   "face": (25,20,18),   "outline": (200,180,120),   "marker": (210,185,130),  "numbers": True,  "roman": True,  "dots": False, "rim": 4,  "rim_color": (190,160,100)},
        {"name": "slate_minimal",         "bg": (70,75,80),   "face": (85,90,95),   "outline": (30,30,32),      "marker": (200,200,200),  "numbers": False, "roman": False, "dots": True,  "rim": 6,  "rim_color": (30,30,32)},
        # ---- Colored face clocks ----
        {"name": "pastel_blue_kids",      "bg": (215,235,250), "face": (225,245,255), "outline": (80,140,200),   "marker": (50,110,170),   "numbers": True,  "roman": False, "dots": True,  "rim": 5,  "rim_color": (80,140,200)},
        {"name": "pastel_pink",           "bg": (245,230,235), "face": (255,240,245), "outline": (190,110,140),  "marker": (160,80,110),   "numbers": True,  "roman": False, "dots": True,  "rim": 4,  "rim_color": (190,110,140)},
        {"name": "pastel_green",          "bg": (225,242,228), "face": (235,250,238), "outline": (70,140,80),    "marker": (40,110,55),    "numbers": True,  "roman": False, "dots": False, "rim": 4,  "rim_color": (70,140,80)},
        {"name": "yellow_teacher",        "bg": (250,250,248), "face": (255,230,0),   "outline": (240,200,0),    "marker": (0,100,220),    "numbers": True,  "roman": False, "dots": True,  "rim": 5,  "rim_color": (240,200,0), "complex_numbers": True},
        {"name": "multicolor_kids",       "bg": (250,250,248), "face": (255,255,255), "outline": (30,30,30),     "marker": "multi",        "numbers": True,  "roman": False, "dots": False, "rim": 5,  "rim_color": (30,30,30)},
        # ---- Luxury / vintage ----
        {"name": "vintage_cream",         "bg": (242,238,225), "face": (238,228,200), "outline": (110,85,45),    "marker": (70,45,15),     "numbers": True,  "roman": True,  "dots": False, "rim": 4,  "rim_color": (110,85,45)},
        {"name": "copper_steampunk",      "bg": (175,160,145), "face": (185,115,50),  "outline": (90,45,10),     "marker": (50,22,5),      "numbers": True,  "roman": False, "dots": True,  "rim": 6,  "rim_color": (90,45,10)},
        {"name": "antique_brass",         "bg": (200,185,155), "face": (210,175,110), "outline": (130,100,50),   "marker": (80,55,15),     "numbers": True,  "roman": True,  "dots": False, "rim": 5,  "rim_color": (130,100,50)},
        {"name": "rose_gold",             "bg": (240,225,220), "face": (248,220,210), "outline": (195,130,115),  "marker": (160,90,80),    "numbers": False, "roman": False, "dots": True,  "rim": 4,  "rim_color": (195,130,115)},
        {"name": "marble_white",          "bg": (230,230,230), "face": (245,243,240), "outline": (140,135,130),  "marker": (60,55,50),     "numbers": True,  "roman": False, "dots": False, "rim": 5,  "rim_color": (140,135,130)},
        # ---- More variety ----
        {"name": "red_accent",            "bg": (245,240,240), "face": (255,252,252), "outline": (180,20,20),    "marker": (15,15,15),     "numbers": True,  "roman": False, "dots": False, "rim": 5,  "rim_color": (180,20,20)},
        {"name": "teal_modern",           "bg": (210,235,235), "face": (220,245,245), "outline": (30,110,110),   "marker": (20,80,80),     "numbers": True,  "roman": False, "dots": False, "rim": 4,  "rim_color": (30,110,110)},
        {"name": "gunmetal",              "bg": (55,58,62),   "face": (70,73,78),    "outline": (160,165,170),   "marker": (180,185,190),  "numbers": False, "roman": False, "dots": False, "rim": 5,  "rim_color": (160,165,170)},
        {"name": "ivory_roman_thin",      "bg": (245,242,234), "face": (254,251,243), "outline": (80,72,55),     "marker": (60,50,35),     "numbers": True,  "roman": True,  "dots": False, "rim": 3,  "rim_color": (80,72,55)},
        {"name": "white_red_seconds",     "bg": (240,240,240), "face": (255,255,255), "outline": (30,30,30),     "marker": (15,15,15),     "numbers": True,  "roman": False, "dots": False, "rim": 5,  "rim_color": (30,30,30)},
        {"name": "ocean_blue",            "bg": (195,215,235), "face": (10,60,105),   "outline": (5,35,65),      "marker": (145,195,255),  "numbers": False, "roman": False, "dots": True,  "rim": 5,  "rim_color": (5,35,65)},
        {"name": "green_retro",           "bg": (200,220,195), "face": (140,180,140), "outline": (50,90,50),     "marker": (20,50,20),     "numbers": True,  "roman": False, "dots": True,  "rim": 5,  "rim_color": (50,90,50)},
        {"name": "white_dots_only",       "bg": (235,235,235), "face": (255,255,255), "outline": (50,50,50),     "marker": (30,30,30),     "numbers": False, "roman": False, "dots": True,  "rim": 4,  "rim_color": (50,50,50)},
        {"name": "dark_blue_gold",        "bg": (20,30,55),   "face": (15,25,50),    "outline": (200,170,80),    "marker": (210,180,90),   "numbers": True,  "roman": False, "dots": False, "rim": 5,  "rim_color": (200,170,80)},
        {"name": "grey_minimal",          "bg": (200,200,200), "face": (230,230,230), "outline": (60,60,60),     "marker": (30,30,30),     "numbers": False, "roman": False, "dots": False, "rim": 4,  "rim_color": (60,60,60)},
        {"name": "white_roman_ticks",     "bg": (242,242,242), "face": (255,255,255), "outline": (25,25,25),     "marker": (20,20,20),     "numbers": True,  "roman": True,  "dots": False, "rim": 5,  "rim_color": (25,25,25)},
        {"name": "sand_beige",            "bg": (230,220,205), "face": (242,232,218), "outline": (130,105,75),   "marker": (90,65,35),     "numbers": True,  "roman": False, "dots": False, "rim": 4,  "rim_color": (130,105,75)},
        {"name": "neon_pink_dark",        "bg": (10,10,15),   "face": (18,18,22),    "outline": (255,20,147),    "marker": (255,100,175),  "numbers": True,  "roman": False, "dots": False, "rim": 3,  "rim_color": (255,20,147)},
        {"name": "warm_white_wood",       "bg": (215,200,180), "face": (250,245,238), "outline": (105,70,35),    "marker": (65,40,15),     "numbers": True,  "roman": False, "dots": False, "rim": 8,  "rim_color": (105,70,35)},
        {"name": "deep_charcoal_roman",   "bg": (45,45,45),   "face": (60,60,60),    "outline": (210,205,200),   "marker": (225,220,215),  "numbers": True,  "roman": True,  "dots": False, "rim": 5,  "rim_color": (200,195,190)},
        {"name": "white_arabic_bold",     "bg": (238,238,238), "face": (255,255,255), "outline": (10,10,10),     "marker": (10,10,10),     "numbers": True,  "roman": False, "dots": False, "rim": 7,  "rim_color": (10,10,10)},
        {"name": "pastel_yellow",         "bg": (250,248,225), "face": (255,253,235), "outline": (160,145,60),   "marker": (120,105,30),   "numbers": True,  "roman": False, "dots": False, "rim": 4,  "rim_color": (160,145,60)},
        {"name": "gunmetal_roman",        "bg": (50,52,55),   "face": (65,68,72),    "outline": (170,165,155),   "marker": (185,180,170),  "numbers": True,  "roman": True,  "dots": False, "rim": 5,  "rim_color": (160,155,145)},
        {"name": "white_slim_ticks",      "bg": (245,245,245), "face": (255,255,255), "outline": (40,40,40),     "marker": (25,25,25),     "numbers": False, "roman": False, "dots": False, "rim": 2,  "rim_color": (40,40,40)},
        {"name": "sky_blue_white",        "bg": (200,225,245), "face": (215,238,255), "outline": (50,100,155),   "marker": (35,75,125),    "numbers": True,  "roman": False, "dots": False, "rim": 4,  "rim_color": (50,100,155)},
        {"name": "terracotta",            "bg": (215,185,165), "face": (200,155,120), "outline": (120,65,35),    "marker": (70,30,10),     "numbers": False, "roman": False, "dots": True,  "rim": 5,  "rim_color": (120,65,35)},
        {"name": "mint_green",            "bg": (210,240,235), "face": (195,235,228), "outline": (40,115,100),   "marker": (20,85,70),     "numbers": True,  "roman": False, "dots": False, "rim": 4,  "rim_color": (40,115,100)},
        {"name": "platinum_roman",        "bg": (220,222,225), "face": (235,237,240), "outline": (130,132,138),  "marker": (50,50,55),     "numbers": True,  "roman": True,  "dots": False, "rim": 5,  "rim_color": (130,132,138)},
        {"name": "pop_art",               "bg": (200,230,255), "face": (255,220,0),   "outline": (0,0,220),      "marker": (220,0,0),      "numbers": True,  "roman": False, "dots": True,  "rim": 6,  "rim_color": (0,0,220)},
    ]
    return styles[style_id % len(styles)]


# =========================
# Background texture
# =========================
def add_background_texture(img: Image.Image, bg_color) -> Image.Image:
    """Add subtle texture outside the clock circle to simulate real backgrounds."""
    choice = random.random()
    if choice < 0.4:
        return img   # plain color — keep as is

    draw = ImageDraw.Draw(img)
    w, h = img.size

    if choice < 0.65:
        # subtle gradient
        r, g, b = bg_color
        for y in range(h):
            factor = y / h * 0.12
            cr = max(0, min(255, int(r * (1 - factor))))
            cg = max(0, min(255, int(g * (1 - factor))))
            cb = max(0, min(255, int(b * (1 - factor))))
            draw.line([(0, y), (w, y)], fill=(cr, cg, cb))
    else:
        # slight noise
        pixels = img.load()
        for _ in range((w * h) // 30):
            x, y = random.randint(0, w-1), random.randint(0, h-1)
            r, g, b = pixels[x, y]
            d = random.randint(-15, 15)
            pixels[x, y] = (max(0,min(255,r+d)), max(0,min(255,g+d)), max(0,min(255,b+d)))
    return img


# =========================
# Glass reflection overlay
# =========================
def add_glass_reflection(img: Image.Image, center, radius) -> Image.Image:
    """Add a subtle semi-transparent white glare to simulate glass cover."""
    if random.random() > 0.55:
        return img

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ov_draw = ImageDraw.Draw(overlay)

    # ellipse shifted to upper-left or upper-right
    side   = random.choice([-1, 1])
    offset = int(radius * random.uniform(0.15, 0.35))
    glare_cx = center[0] + side * offset
    glare_cy = center[1] - int(radius * random.uniform(0.1, 0.25))
    rx = int(radius * random.uniform(0.35, 0.55))
    ry = int(radius * random.uniform(0.20, 0.35))
    alpha = random.randint(18, 42)
    ov_draw.ellipse([glare_cx - rx, glare_cy - ry,
                     glare_cx + rx, glare_cy + ry],
                    fill=(255, 255, 255, alpha))

    glare = overlay.filter(ImageFilter.GaussianBlur(radius=int(radius * 0.12)))
    return Image.alpha_composite(img.convert("RGBA"), glare).convert("RGB")


# =========================
# Draw clock face (clean, no hands)
# =========================
def create_analog_clean(style_id: int, image_size: int = IMAGE_SIZE) -> Image.Image:
    cfg = analog_style_config(style_id)

    # Slight random offset so clock isn't always perfectly centered
    offset_x = random.randint(-15, 15)
    offset_y = random.randint(-15, 15)

    img  = Image.new("RGB", (image_size, image_size), color=cfg["bg"])
    img  = add_background_texture(img, cfg["bg"])
    draw = ImageDraw.Draw(img)

    cx = image_size // 2 + offset_x
    cy = image_size // 2 + offset_y
    center = (cx, cy)
    radius = image_size // 2 - random.randint(20, 35)

    # Drop shadow
    shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    sdraw  = ImageDraw.Draw(shadow)
    so     = random.randint(4, 10)
    sdraw.ellipse([cx - radius + so, cy - radius + so,
                   cx + radius + so, cy + radius + so], fill=(0, 0, 0, 40))
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=10))
    img    = Image.alpha_composite(img.convert("RGBA"), shadow).convert("RGB")
    draw   = ImageDraw.Draw(img)

    rim_w = max(3, cfg.get("rim", 5) + random.randint(0, 3))
    rim_color = cfg.get("rim_color", cfg["outline"])

    # Gradient face (30% chance)
    face = cfg["face"]
    if random.random() < 0.30 and not cfg.get("complex_numbers"):
        for r in range(radius, 0, -1):
            f = r / radius
            gc = (
                int(face[0] * (1-f) + min(255, face[0]+40) * f),
                int(face[1] * (1-f) + min(255, face[1]+40) * f),
                int(face[2] * (1-f) + min(255, face[2]+40) * f),
            )
            draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=gc)
        draw.ellipse([cx-radius, cy-radius, cx+radius, cy+radius],
                     outline=rim_color, width=rim_w)
    else:
        draw.ellipse([cx-radius, cy-radius, cx+radius, cy+radius],
                     fill=face, outline=rim_color, width=rim_w)

    # ---- Special toy clock ----
    if cfg.get("complex_numbers"):
        font_min  = get_font(12)
        font_main = get_font(38)
        font_inn  = get_font(13)
        for i in range(60):
            ang = i * 6
            if i % 5 == 0:
                pos = angle_to_point(center, radius - 15, ang)
                draw_centered_text(draw, str(i) if i else "60", font_min, pos, (0,100,220))
            else:
                pt = angle_to_point(center, radius - 6, ang)
                draw.ellipse([pt[0]-2, pt[1]-2, pt[0]+2, pt[1]+2], fill=(0,100,220))
        for i in range(1, 13):
            pos = angle_to_point(center, radius - 45, i * 30)
            draw_centered_text(draw, str(i), font_main, pos, (230,30,30))
        for i in range(13, 25):
            pos = angle_to_point(center, radius - 78, (i-12) * 30)
            draw_centered_text(draw, str(i), font_inn, pos, (230,30,30))
        img = add_glass_reflection(img, center, radius)
        return img

    # ---- Regular layout ----
    layout = random.choice(["numbers_outer", "numbers_inner"])
    marker_color = cfg["marker"]
    multi_colors = [(210,30,30),(20,120,20),(20,20,200),(200,120,0),(100,0,180),(0,150,150)]

    if cfg["numbers"]:
        if layout == "numbers_outer":
            font_size     = random.randint(46, 58)
            num_radius    = radius - 44
            tick_out      = radius - 72
            tick_in_5     = tick_out - 13
            tick_in_1     = tick_out - 5
        else:
            font_size     = random.randint(42, 54)
            tick_out      = radius - 5
            tick_in_5     = tick_out - 16
            tick_in_1     = tick_out - 6
            num_radius    = radius - 50
    else:
        font_size  = 48
        num_radius = 0
        tick_out   = radius - 8
        tick_in_5  = tick_out - 16
        tick_in_1  = tick_out - 6

    # Draw tick marks
    for i in range(60):
        ang      = i * 6
        is_5min  = (i % 5 == 0)
        is_15min = (i % 15 == 0)

        if cfg["numbers"] and is_5min and layout == "numbers_inner":
            continue

        if cfg["dots"] and is_5min:
            pt = angle_to_point(center, tick_out, ang)
            dr = 5 if is_15min else 3
            draw.ellipse([pt[0]-dr, pt[1]-dr, pt[0]+dr, pt[1]+dr], fill=marker_color)
        elif not cfg["dots"]:
            outer = angle_to_point(center, tick_out, ang)
            inner = angle_to_point(center, tick_in_5 if is_5min else tick_in_1, ang)
            w     = 3 if is_5min else 1
            c     = marker_color
            if marker_color == "multi":
                c = multi_colors[i % len(multi_colors)] if is_5min else (80,80,80)
            draw.line([inner, outer], fill=c, width=w)

    # Draw numbers
    if cfg["numbers"]:
        font = get_font(font_size)
        labels = ROMAN if cfg.get("roman") else [str(n) for n in range(1, 13)]
        for idx, num in enumerate(labels):
            number = idx + 1
            ang  = number * 30
            pos  = angle_to_point(center, num_radius, ang)
            col  = marker_color
            if marker_color == "multi":
                col = multi_colors[idx % len(multi_colors)]
            draw_centered_text(draw, num, font, pos, col)

    # Brand text
    if random.random() < 0.18:
        bfont = get_font(10)
        by    = cy + int(radius * 0.42)
        draw_centered_text(draw, random.choice(["QUARTZ","SILENT","CLASSIC","ANALOG"]),
                           bfont, (cx, by), marker_color if marker_color != "multi" else (80,80,80))

    img = add_glass_reflection(img, center, radius)
    return img


# =========================
# Hand colors & styles
# =========================
def get_hand_palette(cfg: Dict):
    """
    Returns (hour_color, minute_color, second_color, hand_style).
    Biased toward thin/elegant styles dominant in real clocks.
    """
    dark_styles = [
        {"h": (10,10,10),     "m": (10,10,10),     "s": (210,30,30)},   # classic black + red second
        {"h": (20,20,20),     "m": (20,20,20),     "s": (20,20,20)},    # all black
        {"h": (50,50,55),     "m": (50,50,55),     "s": (210,30,30)},   # dark grey + red
        {"h": (210,170,70),   "m": (210,170,70),   "s": (210,170,70)},  # gold
        {"h": (190,195,200),  "m": (190,195,200),  "s": (210,30,30)},   # silver + red
        {"h": (180,180,185),  "m": (180,180,185),  "s": (180,180,185)}, # silver
        {"h": (235,235,235),  "m": (235,235,235),  "s": (210,30,30)},   # white + red (dark face)
        {"h": (225,225,225),  "m": (225,225,225),  "s": (225,225,225)}, # all white
        {"h": (10,10,10),     "m": (10,10,10),     "s": (255,165,0)},   # black + orange second
        {"h": (140,100,45),   "m": (140,100,45),   "s": (210,30,30)},   # bronze + red
    ]
    palette = random.choice(dark_styles)

    # weight toward thin/elegant — matches real clocks
    style = random.choices(
        ["thin_elegant", "sword", "tapered", "straight", "chunky", "massive", "diamond"],
        weights      =[30,         20,        15,         15,       8,         6,         6],
        k=1
    )[0]

    return palette["h"], palette["m"], palette["s"], style


def build_hand_polys(center, h_len, m_len, hour_ang, min_ang, style):
    if style == "thin_elegant":
        h = get_hand_polygon(center, h_len, hour_ang, 5, 1, tail=12)
        m = get_hand_polygon(center, m_len, min_ang,  4, 1, tail=15)
    elif style == "sword":
        h = get_sword_polygon(center, h_len, hour_ang, 14)
        m = get_sword_polygon(center, m_len, min_ang,  10)
    elif style == "tapered":
        h = get_hand_polygon(center, h_len, hour_ang, 12, 1, tail=10)
        m = get_hand_polygon(center, m_len, min_ang,   8, 1, tail=12)
    elif style == "chunky":
        h = get_hand_polygon(center, h_len, hour_ang, 16, 8, tail=14)
        m = get_hand_polygon(center, m_len, min_ang,  12, 6, tail=18)
    elif style == "massive":
        h = get_hand_polygon(center, h_len, hour_ang, 24, 18, tail=20)
        m = get_hand_polygon(center, m_len, min_ang,  18, 12, tail=24)
    elif style == "diamond":
        h = get_diamond_polygon(center, h_len, hour_ang, 16)
        m = get_diamond_polygon(center, m_len, min_ang,  12)
    else:  # straight
        h = get_hand_polygon(center, h_len, hour_ang, 8, 8, tail=10)
        m = get_hand_polygon(center, m_len, min_ang,  6, 6, tail=14)
    return h, m


# =========================
# Add hands + generate mask
# =========================
def add_hands_and_get_mask(
    clean_clock: Image.Image, hour: int, minute: int, second: int, style_id: int
) -> Tuple[Image.Image, Image.Image]:

    cfg    = analog_style_config(style_id)
    img    = clean_clock.copy()
    draw   = ImageDraw.Draw(img)

    mask      = Image.new("L", img.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    W, H   = img.size
    cx, cy = W // 2, H // 2
    center = (cx, cy)
    radius = W // 2 - 26

    hour_ang, min_ang, sec_ang = time_to_angles(hour, minute, second)

    # --- Hand parameters ---
    h_len = radius * 0.50
    m_len = radius * 0.75
    s_len = radius * 0.87

    if cfg.get("complex_numbers"):
        h_color = (230, 30, 30)
        m_color = (0, 120, 240)
        s_color = None
        style   = "toy_arrow"
    else:
        h_color, m_color, s_color, style = get_hand_palette(cfg)

    so    = random.choice([3, 4, 5])
    sc    = (0, 0, 0, random.randint(50, 90))
    s_cen = (cx + so, cy + so)

    # ---- Draw shadows on image + mark on mask ----
    if style == "toy_arrow":
        draw_toy_hand(draw, s_cen, h_len, hour_ang, sc)
        draw_toy_hand(mask_draw, s_cen, h_len, hour_ang, 255)
        draw_toy_hand(draw, s_cen, m_len, min_ang, sc)
        draw_toy_hand(mask_draw, s_cen, m_len, min_ang, 255)
    else:
        h_poly, m_poly = build_hand_polys(center, h_len, m_len, hour_ang, min_ang, style)
        sh = [(x+so, y+so) for x, y in h_poly]
        sm = [(x+so, y+so) for x, y in m_poly]
        draw.polygon(sh, fill=sc)
        mask_draw.polygon(sh, fill=255)
        draw.polygon(sm, fill=sc)
        mask_draw.polygon(sm, fill=255)

    # ---- Draw actual hands on image + mask ----
    if style == "toy_arrow":
        draw_toy_hand(draw, center, h_len, hour_ang, h_color)
        draw_toy_hand(mask_draw, center, h_len, hour_ang, 255)
        draw_toy_hand(draw, center, m_len, min_ang, m_color)
        draw_toy_hand(mask_draw, center, m_len, min_ang, 255)
        cap = 12
        draw.ellipse([cx-cap, cy-cap, cx+cap, cy+cap], fill=m_color)
        mask_draw.ellipse([cx-cap, cy-cap, cx+cap, cy+cap], fill=255)
    else:
        h_poly, m_poly = build_hand_polys(center, h_len, m_len, hour_ang, min_ang, style)
        draw.polygon(h_poly, fill=h_color)
        mask_draw.polygon(h_poly, fill=255)
        draw.polygon(m_poly, fill=m_color)
        mask_draw.polygon(m_poly, fill=255)

        # Second hand (thin, usually red)
        if s_color is not None and random.random() > 0.25:
            s_poly = get_hand_polygon(center, s_len, sec_ang, 2, 1, tail=20)
            draw.polygon(s_poly, fill=s_color)
            mask_draw.polygon(s_poly, fill=255)

        # Center pivot cap (visible on real clocks)
        cap = 9 if style in ["chunky", "massive"] else 5
        draw.ellipse([cx-cap, cy-cap, cx+cap, cy+cap], fill=m_color)
        mask_draw.ellipse([cx-cap, cy-cap, cx+cap, cy+cap], fill=255)

    return img, mask


# =========================
# Dataset generation
# =========================
def get_style_split() -> Tuple[List[int], List[int]]:
    ids = list(range(NUM_ANALOG_STYLES))
    random.shuffle(ids)
    split = int(TRAIN_RATIO * len(ids))
    return ids[:split], ids[split:]

def generate_dataset(num_examples: int = DATASET_SIZE) -> None:
    random.seed(SEED)
    reset_output_dir()

    all_rows, train_rows, test_rows = [], [], []
    train_ids, _ = get_style_split()

    print(f"Generating {num_examples} clock samples...")

    for idx in range(num_examples):
        hour, minute, second = random_time()
        style_id = idx % NUM_ANALOG_STYLES
        split    = "train" if style_id in train_ids else "test"

        hour_ang, min_ang, sec_ang = time_to_angles(hour, minute, second)
        sample_id = f"sample_{idx:05d}"

        clean_img = create_analog_clean(style_id)
        clock_img, mask_img = add_hands_and_get_mask(clean_img, hour, minute, second, style_id)
        clock_img = add_sensor_noise(clock_img)

        analog_path = OUTPUT_DIR / split / "analog_clock"         / f"{sample_id}.png"
        mask_path   = OUTPUT_DIR / split / "analog_without_hands" / f"{sample_id}.png"

        clock_img.save(analog_path)
        mask_img.save(mask_path)

        row = {
            "id": sample_id, "split": split,
            "hour": hour, "minute": minute, "second": second,
            "hour_angle":   round(hour_ang, 4),
            "minute_angle": round(min_ang,  4),
            "second_angle": round(sec_ang,  4),
            "analog_style_id": style_id,
            "analog_path": str(analog_path),
            "analog_without_hands_path": str(mask_path),
        }
        all_rows.append(row)
        if split == "train": train_rows.append(row)
        else:                test_rows.append(row)

        if (idx + 1) % 1000 == 0:
            print(f"  {idx + 1} / {num_examples}")

    write_csv(LABELS_PATH,       all_rows)
    write_csv(TRAIN_LABELS_PATH, train_rows)
    write_csv(TEST_LABELS_PATH,  test_rows)

    print(f"Done! Train: {len(train_rows)} | Test: {len(test_rows)}")

def write_csv(path: Path, rows: List[Dict]) -> None:
    with open(path, mode="w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

if __name__ == "__main__":
    generate_dataset()
