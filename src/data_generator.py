import csv
import math
import os
import random
from typing import List, Set, Tuple

from PIL import Image, ImageDraw, ImageFont

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================

ROMAN_CHARS = ["XII", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI"]
ARABIC_CHARS = ["12", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]


THEME_CONFIG = {
    'neon_cyberpunk':   {'bg': (10, 10, 20),    'face': (20, 20, 35),    'hands': (0, 255, 255),   'accent': (255, 0, 255),   'markers': (0, 255, 150)},
    'standard_classic': {'bg': (240, 240, 240), 'face': (255, 255, 255), 'hands': (0, 0, 0),       'accent': (255, 0, 0),     'markers': (0, 0, 0)},
    'pastel_dream':     {'bg': (255, 228, 225), 'face': (255, 240, 245), 'hands': (120, 150, 200), 'accent': (250, 128, 114), 'markers': (150, 200, 150)},
    'industrial_alert': {'bg': (40, 40, 40),    'face': (220, 220, 0),   'hands': (10, 10, 10),    'accent': (255, 50, 0),    'markers': (20, 20, 20)},
    'crimson_shadow':   {'bg': (20, 5, 5),      'face': (40, 10, 10),    'hands': (220, 200, 200), 'accent': (255, 0, 0),     'markers': (150, 50, 50)},
    'arctic_frost':     {'bg': (230, 245, 255), 'face': (200, 230, 250), 'hands': (20, 60, 100),   'accent': (0, 150, 255),   'markers': (100, 150, 200)},
    'synthwave_sunset': {'bg': (45, 20, 60),    'face': (70, 30, 80),    'hands': (255, 140, 0),   'accent': (255, 20, 147),  'markers': (200, 100, 150)},
    'desert_terracotta':{'bg': (210, 180, 140), 'face': (230, 200, 160), 'hands': (90, 60, 40),    'accent': (200, 80, 40),   'markers': (130, 90, 60)},
    'deep_forest':      {'bg': (15, 30, 15),    'face': (25, 45, 25),    'hands': (180, 200, 180), 'accent': (150, 200, 50),  'markers': (100, 130, 100)},
    'royal_velvet':     {'bg': (50, 10, 40),    'face': (70, 20, 60),    'hands': (240, 210, 150), 'accent': (255, 215, 0),   'markers': (200, 160, 180)},
    'slate_teal':       {'bg': (30, 45, 50),    'face': (50, 70, 80),    'hands': (240, 250, 250), 'accent': (0, 200, 150),   'markers': (150, 180, 190)}
}

# ==============================================================================
# CORE UTILITIES
# ==============================================================================

def load_truetype_font(pixel_size: int, font_target: str = "arial.ttf") -> ImageFont:
    """Attempts to load a specific font size, falling back to OS defaults."""
    target_size = int(pixel_size)
    
    font_paths = [
        font_target,
        "Arial.ttf",
        "DejaVuSans.ttf",
        "DejaVuSans-Bold.ttf",
    ]
    
    for path in font_paths:
        try:
            return ImageFont.truetype(path, size=target_size)
        except (IOError, OSError):
            continue
            
    print("WARNING: Truetype font missing. Text rendering will use tiny default font.")
    return ImageFont.load_default()

def pivot_coordinates(pt: Tuple[float, float], pivot: Tuple[float, float], rad_angle: float) -> Tuple[float, float]:
    """Applies a rotation matrix to a specific 2D coordinate."""
    px, py = pt
    cx, cy = pivot
    
    sin_a = math.sin(rad_angle)
    cos_a = math.cos(rad_angle)
    
    rot_x = cx + (px - cx) * cos_a - (py - cy) * sin_a
    rot_y = cy + (px - cx) * sin_a + (py - cy) * cos_a
    return rot_x, rot_y

def render_clock_hand(canvas: ImageDraw, pivot_pt: Tuple[float, float], rad_angle: float, 
                      hand_len: float, hand_thickness: float, fill_color: Tuple[int, int, int], 
                      design: str = 'line'):
    """Renders a clock hand based on the designated style string."""
    mid_x, mid_y = pivot_pt
    
    # Pre-calculate the extreme tip of the hand
    end_x = mid_x + hand_len * math.cos(rad_angle)
    end_y = mid_y + hand_len * math.sin(rad_angle)

    if design == 'line':
        canvas.line((mid_x, mid_y, end_x, end_y), fill=fill_color, width=int(hand_thickness))
        
    elif design == 'tapered':
        base_width = hand_thickness * 1.5
        perp_angle = rad_angle + (math.pi / 2)
        
        offset = hand_thickness
        base_x = mid_x - offset * math.cos(rad_angle)
        base_y = mid_y - offset * math.sin(rad_angle)
        
        pt1 = (base_x + base_width * math.cos(perp_angle), base_y + base_width * math.sin(perp_angle))
        pt2 = (base_x - base_width * math.cos(perp_angle), base_y - base_width * math.sin(perp_angle))
        
        canvas.polygon([pt1, pt2, (end_x, end_y)], fill=fill_color)

    elif design == 'arrow':
        stem_length = hand_len * 0.7
        arrow_width = hand_thickness * 2.5
        
        stem_x = mid_x + stem_length * math.cos(rad_angle)
        stem_y = mid_y + stem_length * math.sin(rad_angle)
        
        canvas.line((mid_x, mid_y, stem_x, stem_y), fill=fill_color, width=int(hand_thickness))
        
        perp_angle = rad_angle + (math.pi / 2)
        pt1 = (stem_x + (arrow_width / 2) * math.cos(perp_angle), stem_y + (arrow_width / 2) * math.sin(perp_angle))
        pt2 = (stem_x - (arrow_width / 2) * math.cos(perp_angle), stem_y - (arrow_width / 2) * math.sin(perp_angle))
        
        canvas.polygon([pt1, pt2, (end_x, end_y)], fill=fill_color)

    elif design == 'diamond':
        flare_dist = hand_len * 0.3
        max_span = hand_thickness * 2
        perp_angle = rad_angle + (math.pi / 2)
        
        flare_x = mid_x + flare_dist * math.cos(rad_angle)
        flare_y = mid_y + flare_dist * math.sin(rad_angle)
        
        left_pt = (flare_x + max_span * math.cos(perp_angle), flare_y + max_span * math.sin(perp_angle))
        right_pt = (flare_x - max_span * math.cos(perp_angle), flare_y - max_span * math.sin(perp_angle))
        
        rear_len = hand_len * 0.15
        rear_x = mid_x - rear_len * math.cos(rad_angle)
        rear_y = mid_y - rear_len * math.sin(rad_angle)
        
        canvas.polygon([(rear_x, rear_y), left_pt, (end_x, end_y), right_pt], fill=fill_color)

def render_face_markers(canvas: ImageDraw, pivot: Tuple[int, int], rad: int, 
                        marker_type: str, color_val: Tuple[int, int, int], 
                        dim: Tuple[int, int], custom_font=None):
    """Generates the 12 hour indicators around the perimeter of the clock."""
    cx, cy = pivot
    
    for idx in range(12):
        theta = math.radians(idx * 30 - 90)
        
        outer_bound = rad - dim[0] // 40
        text_bound = rad - dim[0] // 10 
        
        if marker_type in ['arabic', 'roman']:
            lbl = ARABIC_CHARS[idx] if marker_type == 'arabic' else ROMAN_CHARS[idx]
            
            txt_x = cx + text_bound * math.cos(theta)
            txt_y = cy + text_bound * math.sin(theta)
            
            box_l, box_t, box_r, box_b = canvas.textbbox((0, 0), lbl, font=custom_font)
            width = box_r - box_l
            height = box_b - box_t
            
            canvas.text((txt_x - width / 2, txt_y - height / 2), lbl, fill=color_val, font=custom_font)
            
        elif marker_type == 'line':
            inner_bound = rad - dim[0] // 15
            start_x = cx + inner_bound * math.cos(theta)
            start_y = cy + inner_bound * math.sin(theta)
            end_x = cx + outer_bound * math.cos(theta)
            end_y = cy + outer_bound * math.sin(theta)
            
            stroke_weight = int(dim[0] // 50) if idx % 3 == 0 else int(dim[0] // 100)
            canvas.line((start_x, start_y, end_x, end_y), fill=color_val, width=stroke_weight)
            
        elif marker_type == 'dot':
            dot_dist = rad - dim[0] // 20
            dx = cx + dot_dist * math.cos(theta)
            dy = cy + dot_dist * math.sin(theta)
            
            dot_rad = dim[0] // 50 if idx % 3 == 0 else dim[0] // 80
            canvas.ellipse((dx - dot_rad, dy - dot_rad, dx + dot_rad, dy + dot_rad), fill=color_val)

# ==============================================================================
# DIGITAL RENDERERS
# ==============================================================================

def calculate_best_font(canvas: ImageDraw, content: str, max_w: int, max_h: int, 
                        initial_pt: float, typeface="arial.ttf", limit_pt=10) -> ImageFont:
    """Scales down the font size until the text fits within the given bounding box."""
    current_pt = int(initial_pt)

    while current_pt >= limit_pt:
        active_font = load_truetype_font(current_pt, typeface)
        bound_l, bound_t, bound_r, bound_b = canvas.textbbox((0, 0), content, font=active_font)
        
        if (bound_r - bound_l) <= max_w and (bound_b - bound_t) <= max_h:
            return active_font
            
        current_pt -= 2

    return load_truetype_font(limit_pt, typeface)

def create_simple_digital(hr: int, mnt: int, sec: int, dimensions: Tuple[int, int]) -> Image:
    canvas_img = Image.new('RGB', dimensions, color=(15, 15, 20))
    painter = ImageDraw.Draw(canvas_img)
    timestamp = f"{hr:02d}:{mnt:02d}:{sec:02d}"

    w_limit = int(dimensions[0] * 0.85)
    h_limit = int(dimensions[1] * 0.30)

    chosen_font = calculate_best_font(painter, timestamp, w_limit, h_limit, int(dimensions[0] * 0.25))

    b_left, b_top, b_right, b_bot = painter.textbbox((0, 0), timestamp, font=chosen_font)
    txt_w, txt_h = b_right - b_left, b_bot - b_top
    
    pos_x = (dimensions[0] - txt_w) / 2
    pos_y = (dimensions[1] - txt_h) / 2

    painter.text((pos_x, pos_y), timestamp, fill=(220, 60, 60), font=chosen_font)
    return canvas_img

def create_segmented_digital(hr: int, mnt: int, sec: int, dimensions: Tuple[int, int]) -> Image:
    canvas_img = Image.new('RGB', dimensions, color=(15, 15, 20))
    painter = ImageDraw.Draw(canvas_img)
    timestamp = f"{hr:02d}:{mnt:02d}:{sec:02d}"

    margin = dimensions[0] // 10
    display_box = [margin, dimensions[1] // 3, dimensions[0] - margin, 2 * dimensions[1] // 3]

    painter.rectangle(display_box, fill=(25, 35, 25), outline=(0, 140, 90), width=2)

    box_w = display_box[2] - display_box[0] - 12
    box_h = display_box[3] - display_box[1] - 12

    chosen_font = calculate_best_font(painter, timestamp, box_w, box_h, int(dimensions[0] * 0.22))

    b_left, b_top, b_right, b_bot = painter.textbbox((0, 0), timestamp, font=chosen_font)
    txt_w, txt_h = b_right - b_left, b_bot - b_top
    
    pos_x = (dimensions[0] - txt_w) / 2
    pos_y = (dimensions[1] - txt_h) / 2

    painter.text((pos_x, pos_y), timestamp, fill=(0, 230, 150), font=chosen_font)
    return canvas_img

def create_lcd_digital(hr: int, mnt: int, sec: int, dimensions: Tuple[int, int]) -> Image:
    canvas_img = Image.new('RGB', dimensions, color=(180, 200, 180))
    painter = ImageDraw.Draw(canvas_img)
    timestamp = f"{hr:02d}:{mnt:02d}:{sec:02d}"

    margin = dimensions[0] // 10
    display_box = [margin, dimensions[1] // 3, dimensions[0] - margin, 2 * dimensions[1] // 3]

    painter.rectangle(display_box, fill=(200, 220, 200), outline=(100, 120, 100), width=2)

    box_w = display_box[2] - display_box[0] - 12
    box_h = display_box[3] - display_box[1] - 12

    chosen_font = calculate_best_font(painter, timestamp, box_w, box_h, int(dimensions[0] * 0.22))

    b_left, b_top, b_right, b_bot = painter.textbbox((0, 0), timestamp, font=chosen_font)
    txt_w, txt_h = b_right - b_left, b_bot - b_top
    
    pos_x = (dimensions[0] - txt_w) / 2
    pos_y = (dimensions[1] - txt_h) / 2

    painter.text((pos_x, pos_y), timestamp, fill=(40, 60, 40), font=chosen_font)
    return canvas_img

# ==============================================================================
# ANALOG RENDERERS
# ==============================================================================

def generate_dynamic_analog(hr: int, mnt: int, sec: int, dimensions: Tuple[int, int], theme: dict) -> Tuple[Image.Image, Image.Image]:
    """Builds a classic circular analog clock with randomized styles."""
    base_img = Image.new('RGB', dimensions, color=theme['bg'])
    painter = ImageDraw.Draw(base_img)
    
    midpoint = (dimensions[0] // 2, dimensions[1] // 2)
    clock_rad = min(dimensions) // 2 - max(5, dimensions[0] // 25)
    
    # Render background circle
    painter.ellipse((midpoint[0] - clock_rad, midpoint[1] - clock_rad, 
                     midpoint[0] + clock_rad, midpoint[1] + clock_rad),
                    outline=theme['hands'], width=max(2, dimensions[0] // 128), fill=theme['face'])
    
    # Inject markers
    m_style = random.choice(['line', 'dot', 'arabic', 'roman'])
    m_font = load_truetype_font(dimensions[0] / 8) if m_style in ['arabic', 'roman'] else None
    render_face_markers(painter, midpoint, clock_rad, m_style, theme['markers'], dimensions, m_font)

    # Save a clean copy without hands
    handless_copy = base_img.copy()

    # Determine hand architecture
    h_style = random.choice(['line', 'tapered', 'arrow', 'diamond'])
    
    # Compute hand angles (radians)
    theta_sec = math.radians(sec * 6 - 90)
    theta_min = math.radians(mnt * 6 + sec * 0.1 - 90)
    theta_hr = math.radians((hr % 12) * 30 + mnt * 0.5 - 90)
    
    # Paint the hands
    render_clock_hand(painter, midpoint, theta_hr, clock_rad * 0.5, dimensions[0] * 0.04, theme['hands'], h_style)
    render_clock_hand(painter, midpoint, theta_min, clock_rad * 0.75, dimensions[0] * 0.03, theme['hands'], h_style)
    
    s_style = 'line' if h_style == 'arrow' else h_style 
    render_clock_hand(painter, midpoint, theta_sec, clock_rad * 0.85, dimensions[0] * 0.01, theme['accent'], s_style)
    
    # Center pivot cap
    pivot_radius = dimensions[0] // 30
    painter.ellipse((midpoint[0] - pivot_radius, midpoint[1] - pivot_radius, 
                     midpoint[0] + pivot_radius, midpoint[1] + pivot_radius), fill=theme['hands'])
    
    return base_img, handless_copy

def generate_square_analog(hr: int, mnt: int, sec: int, dimensions: Tuple[int, int], theme: dict) -> Tuple[Image.Image, Image.Image]:
    """Builds an alternative square-faced analog clock."""
    base_img = Image.new('RGB', dimensions, color=theme['bg'])
    painter = ImageDraw.Draw(base_img)
    
    padding = dimensions[0] // 10
    bounding_rect = [padding, padding, dimensions[0] - padding, dimensions[1] - padding]
    
    painter.rectangle(bounding_rect, fill=theme['face'], outline=theme['hands'], width=3)
    
    midpoint = (dimensions[0] // 2, dimensions[1] // 2)
    clock_rad = (dimensions[0] // 2) - padding - 10
    
    m_style = random.choice(['line', 'arabic'])
    text_font = load_truetype_font(dimensions[0] / 9)
    render_face_markers(painter, midpoint, clock_rad, m_style, theme['markers'], dimensions, text_font)

    handless_copy = base_img.copy()

    theta_sec = math.radians(sec * 6 - 90)
    theta_min = math.radians(mnt * 6 + sec * 0.1 - 90)
    theta_hr = math.radians((hr % 12) * 30 + mnt * 0.5 - 90)
    
    render_clock_hand(painter, midpoint, theta_hr, clock_rad * 0.5, dimensions[0] * 0.04, theme['hands'], 'line')
    render_clock_hand(painter, midpoint, theta_min, clock_rad * 0.75, dimensions[0] * 0.03, theme['hands'], 'line')
    render_clock_hand(painter, midpoint, theta_sec, clock_rad * 0.85, dimensions[0] * 0.01, theme['accent'], 'line')
    
    return base_img, handless_copy

# ==============================================================================
# PIPELINE ORCHESTRATION
# ==============================================================================

DIGITAL_GENERATORS = [('simple', create_simple_digital), ('lcd', create_lcd_digital), ('segmented', create_segmented_digital)]
ANALOG_GENERATORS = [('dynamic', generate_dynamic_analog), ('square', generate_square_analog)]

class ClockDatasetBuilder:
    def __init__(self, max_unique_train_samples=400):
        self.max_unique_train_samples = max_unique_train_samples
        self.training_times: List[Tuple[int, int, int]] = []

    def fetch_train_times(self) -> List[Tuple[int, int, int]]:
        if self.training_times:
            return self.training_times
            
        unique_times = set()
        while len(unique_times) < self.max_unique_train_samples:
            unique_times.add((random.randint(0, 23), random.randint(0, 59), random.randint(0, 59)))
            
        self.training_times = list(unique_times)
        return self.training_times

    def generate_test_time(self) -> Tuple[int, int, int]:
        train_set = set(self.fetch_train_times())
        while True:
            candidate = (random.randint(0, 23), random.randint(0, 59), random.randint(0, 59))
            if candidate not in train_set:
                return candidate

def compile_dataset(builder: ClockDatasetBuilder, partition: str, total_samples: int, out_path: str, img_dim: Tuple[int, int]):
    print(f"Creating '{partition}' partition with {total_samples} samples...")
    
    # Restructured output paths: dataset -> analog/digital -> train/test
    dir_digital = os.path.join(out_path, 'digital', partition)
    dir_analog = os.path.join(out_path, 'analog', partition)
    
    os.makedirs(dir_digital, exist_ok=True)
    os.makedirs(dir_analog, exist_ok=True)
    
    # Setup separate CSV files for digital and analog
    csv_path_digital = os.path.join(dir_digital, 'labels.csv')
    csv_path_analog = os.path.join(dir_analog, 'labels.csv')
    
    with open(csv_path_digital, 'w', newline='') as file_dig, \
         open(csv_path_analog, 'w', newline='') as file_ana:
        
        writer_dig = csv.writer(file_dig)
        writer_ana = csv.writer(file_ana)
        
        # Write specific headers for each
        writer_dig.writerow(['filename', 'hour', 'minute', 'second'])
        writer_ana.writerow(['filename', 'clean_filename', 'hour', 'minute', 'second'])
        
        training_pool = builder.fetch_train_times()
        
        for idx in range(total_samples):
            hr, mnt, sec = random.choice(training_pool) if partition == 'train' else builder.generate_test_time()
            file_prefix = f"{hr:02d}_{mnt:02d}_{sec:02d}_{idx:05d}"

            dig_label, dig_callback = random.choice(DIGITAL_GENERATORS)
            ana_label, ana_callback = random.choice(ANALOG_GENERATORS)
            theme_key = random.choice(list(THEME_CONFIG.keys()))
            
            digital_out = dig_callback(hr, mnt, sec, img_dim)
            analog_out, analog_clean = ana_callback(hr, mnt, sec, img_dim, THEME_CONFIG[theme_key])
            
            name_dig = f"{file_prefix}_dig_{dig_label}.png"
            name_ana = f"{file_prefix}_ana_{ana_label}_{theme_key}.png"
            name_clean = f"{file_prefix}_ana_clean_{ana_label}_{theme_key}.png"
            
            digital_out.save(os.path.join(dir_digital, name_dig))
            analog_out.save(os.path.join(dir_analog, name_ana))
            analog_clean.save(os.path.join(dir_analog, name_clean))
            
            # Write specific data rows to each CSV
            writer_dig.writerow([name_dig, hr, mnt, sec])
            writer_ana.writerow([name_ana, name_clean, hr, mnt, sec])
            
            if (idx + 1) % 100 == 0:
                print(f"  [{partition.upper()}] Progress: {idx + 1} / {total_samples}")

def main():
    # --- Edit parameters below before executing ---
    EXPORT_PATH = "datasets"
    SAMPLES_TRAIN = 50
    SAMPLES_TEST = 10
    PIXEL_SIZE = 256
    
    # Note: Fixed the init parameter name from your snippet to match the constructor
    dataset_mgr = ClockDatasetBuilder(max_unique_train_samples=400)
    
    compile_dataset(dataset_mgr, 'train', SAMPLES_TRAIN, EXPORT_PATH, (PIXEL_SIZE, PIXEL_SIZE))
    compile_dataset(dataset_mgr, 'test', SAMPLES_TEST, EXPORT_PATH, (PIXEL_SIZE, PIXEL_SIZE))

if __name__ == "__main__":
    main()