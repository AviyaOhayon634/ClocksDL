import os
import random
import math
import csv
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple, List, Set

# ===================== CONSTANTS & CONFIG =====================

ROMAN_NUMERALS = ["XII", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI"]
ARABIC_NUMERALS = ["12", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]

COLOR_PALETTES = {
    'classic_white': {'bg': (245, 245, 245), 'face': (255, 255, 255), 'hands': (30, 30, 30), 'accent': (200, 50, 50), 'markers': (50, 50, 50)},
    'dark_modern':   {'bg': (25, 25, 30), 'face': (45, 45, 50), 'hands': (240, 240, 240), 'accent': (80, 200, 255), 'markers': (200, 200, 200)},
    'vintage_cream': {'bg': (210, 190, 170), 'face': (250, 240, 220), 'hands': (80, 60, 40), 'accent': (150, 80, 50), 'markers': (100, 80, 60)},
    'minimal_gray':  {'bg': (240, 240, 245), 'face': (255, 255, 255), 'hands': (60, 60, 60), 'accent': (220, 60, 60), 'markers': (180, 180, 180)},
    'blue_ocean':    {'bg': (30, 50, 80), 'face': (50, 80, 120), 'hands': (220, 230, 250), 'accent': (100, 200, 255), 'markers': (150, 180, 220)},
    'orange_warm':   {'bg': (255, 200, 150), 'face': (255, 145, 70), 'hands': (100, 50, 10), 'accent': (255, 100, 20), 'markers': (150, 80, 30)},
    'green_nature':  {'bg': (200, 220, 200), 'face': (240, 250, 240), 'hands': (40, 80, 40), 'accent': (80, 150, 80), 'markers': (100, 150, 100)},
    'purple_elegant':{'bg': (230, 220, 240), 'face': (250, 245, 255), 'hands': (80, 60, 100), 'accent': (150, 100, 180), 'markers': (120, 100, 140)},
    'black_white':   {'bg': (0, 0, 0), 'face': (30, 30, 30), 'hands': (255, 255, 255), 'accent': (255, 50, 50), 'markers': (200, 200, 200)},
    'gold_luxury':   {'bg': (50, 40, 30), 'face': (240, 230, 210), 'hands': (150, 120, 60), 'accent': (200, 170, 80), 'markers': (180, 150, 80)}
}

# ===================== HELPER FUNCTIONS =====================

def get_font(size_px, font_name="arial.ttf"):
    import sys
    size_px = int(size_px)
    
    # List of fonts to try in order
    candidates = [
        font_name,
        "Arial.ttf",
        "DejaVuSans.ttf",
        "DejaVuSans-Bold.ttf", 
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/System/Library/Fonts/Helvetica.ttc",      # macOS
        "/System/Library/Fonts/Arial.ttf",          # macOS
        "C:/Windows/Fonts/arial.ttf",               # Windows
        "C:/Windows/Fonts/cour.ttf",                # Windows fallback
    ]
    
    for candidate in candidates:
        try:
            font = ImageFont.truetype(candidate, size=size_px)
            return font
        except (IOError, OSError):
            continue
    
    # If nothing works, print a warning so you know
    print(f"WARNING: No truetype font found! Text will be tiny. Install fonts or add path.")
    return ImageFont.load_default()

def rotate_point(point, center, angle_rad):
    """Rotates a point (x, y) around center (cx, cy) by angle_rad."""
    x, y = point
    cx, cy = center
    new_x = cx + (x - cx) * math.cos(angle_rad) - (y - cy) * math.sin(angle_rad)
    new_y = cy + (x - cx) * math.sin(angle_rad) + (y - cy) * math.cos(angle_rad)
    return new_x, new_y

def draw_hand_fancy(draw, center, angle_rad, length, width, color, style='line'):
    """
    Draws clock hands in various shapes:
    - 'line': Standard rectangle (classic)
    - 'tapered': Triangle getting thinner at the end
    - 'arrow': Classic arrow shape
    - 'diamond': Diamond shape
    """
    cx, cy = center
    
    # Calculate the tip position
    tip_x = cx + length * math.cos(angle_rad)
    tip_y = cy + length * math.sin(angle_rad)

    if style == 'line':
        draw.line((cx, cy, tip_x, tip_y), fill=color, width=int(width))
        
    elif style == 'tapered':
        # Base of the triangle (perpendicular to angle)
        base_w = width * 1.5
        angle_perp = angle_rad + math.pi / 2
        
        # Calculate base points slightly "behind" center so it covers the pivot
        back_offset = width
        base_cx = cx - back_offset * math.cos(angle_rad)
        base_cy = cy - back_offset * math.sin(angle_rad)
        
        p1 = (base_cx + base_w * math.cos(angle_perp), base_cy + base_w * math.sin(angle_perp))
        p2 = (base_cx - base_w * math.cos(angle_perp), base_cy - base_w * math.sin(angle_perp))
        
        draw.polygon([p1, p2, (tip_x, tip_y)], fill=color)

    elif style == 'arrow':
        # Shaft + Head
        shaft_len = length * 0.7
        shaft_w = width
        head_w = width * 2.5
        
        # Shaft end point
        s_end_x = cx + shaft_len * math.cos(angle_rad)
        s_end_y = cy + shaft_len * math.sin(angle_rad)
        
        # Draw Shaft
        draw.line((cx, cy, s_end_x, s_end_y), fill=color, width=int(shaft_w))
        
        # Draw Arrow Head
        angle_perp = angle_rad + math.pi / 2
        p1 = (s_end_x + head_w/2 * math.cos(angle_perp), s_end_y + head_w/2 * math.sin(angle_perp))
        p2 = (s_end_x - head_w/2 * math.cos(angle_perp), s_end_y - head_w/2 * math.sin(angle_perp))
        draw.polygon([p1, p2, (tip_x, tip_y)], fill=color)

    elif style == 'diamond':
        # Kite/Diamond shape
        mid_len = length * 0.3
        max_w = width * 2
        angle_perp = angle_rad + math.pi / 2
        
        # Widest point
        mid_x = cx + mid_len * math.cos(angle_rad)
        mid_y = cy + mid_len * math.sin(angle_rad)
        
        p_left = (mid_x + max_w * math.cos(angle_perp), mid_y + max_w * math.sin(angle_perp))
        p_right = (mid_x - max_w * math.cos(angle_perp), mid_y - max_w * math.sin(angle_perp))
        
        # Back tail
        tail_len = length * 0.15
        tail_x = cx - tail_len * math.cos(angle_rad)
        tail_y = cy - tail_len * math.sin(angle_rad)
        
        draw.polygon([(tail_x, tail_y), p_left, (tip_x, tip_y), p_right], fill=color)

def draw_markers(draw, center, radius, style, color, size, font=None):
    """Draws face markers: lines, dots, arabic numbers, or roman numerals"""
    cx, cy = center
    
    for i in range(12):
        angle = math.radians(i * 30 - 90) # 0 is at 12 o'clock (which is -90 deg)
        
        # Distances from center
        dist_outer = radius - size[0] // 40
        dist_text = radius - size[0] // 10 # Text needs to be further in
        
        if style in ['arabic', 'roman']:
            text = ARABIC_NUMERALS[i] if style == 'arabic' else ROMAN_NUMERALS[i]
            
            # Position for text center
            tx = cx + dist_text * math.cos(angle)
            ty = cy + dist_text * math.sin(angle)
            
            # Calculate text size to center it
            left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
            w, h = right - left, bottom - top
            draw.text((tx - w/2, ty - h/2), text, fill=color, font=font)
            
        elif style == 'line':
            dist_inner = radius - size[0] // 15
            sx = cx + dist_inner * math.cos(angle)
            sy = cy + dist_inner * math.sin(angle)
            ex = cx + dist_outer * math.cos(angle)
            ey = cy + dist_outer * math.sin(angle)
            width = int(size[0] // 50) if i % 3 == 0 else int(size[0] // 100)
            draw.line((sx, sy, ex, ey), fill=color, width=width)
            
        elif style == 'dot':
            dist_dot = radius - size[0] // 20
            dx = cx + dist_dot * math.cos(angle)
            dy = cy + dist_dot * math.sin(angle)
            r = size[0] // 50 if i % 3 == 0 else size[0] // 80
            draw.ellipse((dx-r, dy-r, dx+r, dy+r), fill=color)

# ===================== DIGITAL CLOCK FUNCTIONS =====================

def get_fitted_font(draw, text, max_width, max_height, start_size, font_name="arial.ttf", min_size=10):
    size = int(start_size)

    while size >= min_size:
        font = get_font(size, font_name)
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        w, h = right - left, bottom - top

        if w <= max_width and h <= max_height:
            return font

        size -= 2

    return get_font(min_size, font_name)

def draw_digital_simple(h, m, s, size):
    img = Image.new('RGB', size, color=(15, 15, 20))
    draw = ImageDraw.Draw(img)
    time_str = f"{h:02d}:{m:02d}:{s:02d}"

    max_width = int(size[0] * 0.85)
    max_height = int(size[1] * 0.30)

    font = get_fitted_font(
        draw,
        time_str,
        max_width=max_width,
        max_height=max_height,
        start_size=int(size[0] * 0.25)
    )

    left, top, right, bottom = draw.textbbox((0, 0), time_str, font=font)
    w, h_txt = right - left, bottom - top
    x = (size[0] - w) / 2
    y = (size[1] - h_txt) / 2

    draw.text((x, y), time_str, fill=(220, 60, 60), font=font)
    return img


def draw_digital_segmented(h, m, s, size):
    img = Image.new('RGB', size, color=(15, 15, 20))
    draw = ImageDraw.Draw(img)

    time_str = f"{h:02d}:{m:02d}:{s:02d}"

    mgn = size[0] // 10
    panel = [mgn, size[1] // 3, size[0] - mgn, 2 * size[1] // 3]

    draw.rectangle(
        panel,
        fill=(25, 35, 25),
        outline=(0, 140, 90),
        width=2
    )

    panel_width = panel[2] - panel[0] - 12
    panel_height = panel[3] - panel[1] - 12

    font = get_fitted_font(
        draw,
        time_str,
        max_width=panel_width,
        max_height=panel_height,
        start_size=int(size[0] * 0.22)
    )

    left, top, right, bottom = draw.textbbox((0, 0), time_str, font=font)
    w, h_txt = right - left, bottom - top
    x = (size[0] - w) / 2
    y = (size[1] - h_txt) / 2

    draw.text((x, y), time_str, fill=(0, 230, 150), font=font)
    return img


def draw_digital_lcd(h, m, s, size):
    img = Image.new('RGB', size, color=(180, 200, 180))
    draw = ImageDraw.Draw(img)

    time_str = f"{h:02d}:{m:02d}:{s:02d}"

    mgn = size[0] // 10
    panel = [mgn, size[1] // 3, size[0] - mgn, 2 * size[1] // 3]

    draw.rectangle(
        panel,
        fill=(200, 220, 200),
        outline=(100, 120, 100),
        width=2
    )

    panel_width = panel[2] - panel[0] - 12
    panel_height = panel[3] - panel[1] - 12

    font = get_fitted_font(
        draw,
        time_str,
        max_width=panel_width,
        max_height=panel_height,
        start_size=int(size[0] * 0.22)
    )

    left, top, right, bottom = draw.textbbox((0, 0), time_str, font=font)
    w, h_txt = right - left, bottom - top
    x = (size[0] - w) / 2
    y = (size[1] - h_txt) / 2

    draw.text((x, y), time_str, fill=(40, 60, 40), font=font)
    return img

# ===================== ANALOG CLOCK FUNCTIONS =====================

def draw_analog_dynamic(h, m, s, size, palette):
    """
    Dynamically generates an analog clock with random hands and random marker styles.
    """
    img = Image.new('RGB', size, color=palette['bg'])
    draw = ImageDraw.Draw(img)
    center = (size[0] // 2, size[1] // 2)
    radius = min(size) // 2 - max(5, size[0] // 25)
    
    # 1. Draw Face
    draw.ellipse((center[0]-radius, center[1]-radius, center[0]+radius, center[1]+radius),
                 outline=palette['hands'], width=max(2, size[0]//128), fill=palette['face'])
    
    # 2. Randomize Marker Style
    marker_style = random.choice(['line', 'dot', 'arabic', 'roman'])
    marker_font = get_font(size[0] / 8) if marker_style in ['arabic', 'roman'] else None
    
    draw_markers(draw, center, radius, marker_style, palette['markers'], size, marker_font)

    clean_bg = img.copy()

    # 3. Randomize Hand Style
    # We can mix styles (e.g., hour/min are tapered, second is line) or keep uniform
    hand_style = random.choice(['line', 'tapered', 'arrow', 'diamond'])
    
    # Calculate Angles
    sec_angle = math.radians(s * 6 - 90)
    min_angle = math.radians(m * 6 + s * 0.1 - 90)
    hour_angle = math.radians((h % 12) * 30 + m * 0.5 - 90)
    
    # Draw Hands (Hour, Minute, Second)
    # Hour
    draw_hand_fancy(draw, center, hour_angle, radius * 0.5, size[0] * 0.04, palette['hands'], hand_style)
    # Minute
    draw_hand_fancy(draw, center, min_angle, radius * 0.75, size[0] * 0.03, palette['hands'], hand_style)
    # Second (usually thinner and often red/accent, usually 'line' or 'tapered' looks best)
    sec_style = 'line' if hand_style == 'arrow' else hand_style 
    draw_hand_fancy(draw, center, sec_angle, radius * 0.85, size[0] * 0.01, palette['accent'], sec_style)
    
    # Center Cap
    cap_r = size[0] // 30
    draw.ellipse((center[0]-cap_r, center[1]-cap_r, center[0]+cap_r, center[1]+cap_r), fill=palette['hands'])
    
    return img, clean_bg

def draw_analog_square(h, m, s, size, palette):
    """Square face variant (keeps simple lines for markers to fit corners better)"""
    img = Image.new('RGB', size, color=palette['bg'])
    draw = ImageDraw.Draw(img)
    margin = size[0] // 10
    face_rect = [margin, margin, size[0]-margin, size[1]-margin]
    
    draw.rectangle(face_rect, fill=palette['face'], outline=palette['hands'], width=3)
    
    # Randomize numbers vs lines
    center = (size[0]//2, size[1]//2)
    radius = (size[0]//2) - margin - 10
    marker_style = random.choice(['line', 'arabic'])
    font = get_font(size[0]/9)
    draw_markers(draw, center, radius, marker_style, palette['markers'], size, font)

    clean_bg = img.copy()

    # Hands
    sec_angle = math.radians(s * 6 - 90)
    min_angle = math.radians(m * 6 + s * 0.1 - 90)
    hour_angle = math.radians((h % 12) * 30 + m * 0.5 - 90)
    
    draw_hand_fancy(draw, center, hour_angle, radius*0.5, size[0]*0.04, palette['hands'], 'line')
    draw_hand_fancy(draw, center, min_angle, radius*0.75, size[0]*0.03, palette['hands'], 'line')
    draw_hand_fancy(draw, center, sec_angle, radius*0.85, size[0]*0.01, palette['accent'], 'line')
    
    return img, clean_bg

# ===================== DATASET GENERATION LOGIC =====================

DIGITAL_STYLES = [('simple', draw_digital_simple), ('lcd', draw_digital_lcd), ('segmented', draw_digital_segmented)]
# Note: 'dynamic' covers classic, modern, and fancy combinations
ANALOG_STYLES = [('dynamic', draw_analog_dynamic), ('square', draw_analog_square)]

class DatasetManager:
    def __init__(self, train_max_unique=400):
        self.train_max_unique = train_max_unique
        self.train_times: List[Tuple[int, int, int]] = []
        self.test_times_used: Set[Tuple[int, int, int]] = set()

    def get_train_times(self):
        if self.train_times: return self.train_times
        pool = set()
        while len(pool) < self.train_max_unique:
            pool.add((random.randint(0, 23), random.randint(0, 59), random.randint(0, 59)))
        self.train_times = list(pool)
        return self.train_times

    def get_test_time(self):
        train_pool = set(self.get_train_times())
        while True:
            t = (random.randint(0, 23), random.randint(0, 59), random.randint(0, 59))
            if t not in train_pool: return t

def generate_subset(manager, subset_name, count, root_dir, size):
    print(f"Generating {subset_name} set ({count} images)...")
    base_dir = os.path.join(root_dir, subset_name)
    dig_dir = os.path.join(base_dir, 'digital')
    ana_dir = os.path.join(base_dir, 'analog')
    os.makedirs(dig_dir, exist_ok=True)
    os.makedirs(ana_dir, exist_ok=True)
    
    csv_file = open(os.path.join(base_dir, 'labels.csv'), 'w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(['digital_filename', 'analog_filename', 'analog_clean_filename', 'hour', 'minute', 'second'])
    train_pool = manager.get_train_times()
    
    for i in range(count):
        h, m, s = random.choice(train_pool) if subset_name == 'train' else manager.get_test_time()
        base = f"{h:02d}_{m:02d}_{s:02d}_{i:05d}"

        # Select Styles
        dig_name, dig_func = random.choice(DIGITAL_STYLES)
        ana_name, ana_func = random.choice(ANALOG_STYLES)
        pal_name = random.choice(list(COLOR_PALETTES.keys()))
        
        # Render
        dig_img = dig_func(h, m, s, size)
        ana_img, clean_img = ana_func(h, m, s, size, COLOR_PALETTES[pal_name])
        clean_fn = f"{base}_ana_clean_{ana_name}_{pal_name}.png"
        clean_img.save(os.path.join(ana_dir, clean_fn))
        
        # Save        
        d_fn = f"{base}_dig_{dig_name}.png"
        a_fn = f"{base}_ana_{ana_name}_{pal_name}.png"
        clean_fn = f"{base}_ana_clean_{ana_name}_{pal_name}.png"
        
        dig_img.save(os.path.join(dig_dir, d_fn))
        ana_img.save(os.path.join(ana_dir, a_fn))
        clean_img.save(os.path.join(ana_dir, clean_fn))
        writer.writerow([d_fn, a_fn, clean_fn, h, m, s])
        
        if (i+1) % 100 == 0: print(f"  {subset_name}: {i+1}/{count}")

    csv_file.close()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--train_count", type=int, default=1000)
    parser.add_argument("--test_count", type=int, default=200)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    random.seed(args.seed)
    manager = DatasetManager(train_max_unique=400)
    
    generate_subset(manager, 'train', args.train_count, args.output_dir, (args.image_size, args.image_size))
    generate_subset(manager, 'test', args.test_count, args.output_dir, (args.image_size, args.image_size))

if __name__ == "__main__":
    main()