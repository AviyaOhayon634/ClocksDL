import os
import math
import random
import string
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# ===================== HELPER FUNCTIONS =====================

def get_bold_font(size):
    """Attempts to load a bold, sans-serif font like the Pyramid clock."""
    candidates = [
        "arialbd.ttf", "Arial Bold.ttf", "Helvetica-Bold.ttf", 
        "tahoma.ttf", "verdana.ttf", "FreeSansBold.ttf"
    ]
    for font_name in candidates:
        try:
            return ImageFont.truetype(font_name, int(size))
        except IOError:
            continue
    return ImageFont.load_default()

def generate_fake_logo():
    """Generates a random 5-8 letter string to simulate the PYRAMID logo."""
    length = random.randint(5, 8)
    return "".join(random.choices(string.ascii_uppercase, k=length))


# ===================== DATASET GENERATOR =====================

class PyramidDatasetGenerator:
    def __init__(self, output_dir="pyramid_clocks", img_size=256):
        self.output_dir = output_dir
        self.img_size = img_size
        self.inputs_dir = os.path.join(output_dir, "inputs")
        self.targets_dir = os.path.join(output_dir, "targets")
        
        os.makedirs(self.inputs_dir, exist_ok=True)
        os.makedirs(self.targets_dir, exist_ok=True)
        
        self.labels_data = []

    def draw_single_clock(self):
        """Generates one dirty input and one clean target."""
        size = self.img_size
        face_color = (random.randint(240, 255), random.randint(240, 255), random.randint(240, 255))
        rim_color = (random.randint(10, 30), random.randint(10, 30), random.randint(10, 30))
        
        img = Image.new("RGBA", (size, size), face_color)
        draw = ImageDraw.Draw(img)
        
        cx, cy = size // 2, size // 2
        outer_radius = int((size // 2) * 0.95)
        rim_thickness = int(size * 0.06)
        radius = outer_radius - rim_thickness
        
        # 1. Bezel
        draw.ellipse([cx - outer_radius, cy - outer_radius, cx + outer_radius, cy + outer_radius],
                     outline=rim_color, width=rim_thickness)

        # 2. Ticks
        for i in range(60):
            angle = math.radians(i * 6 - 90)
            is_hour = (i % 5 == 0)
            tick_len = radius * 0.08 if is_hour else radius * 0.04
            tick_width = max(2, int(radius * 0.03)) if is_hour else max(1, int(radius * 0.01))
            
            start_x = cx + (radius - tick_len) * math.cos(angle)
            start_y = cy + (radius - tick_len) * math.sin(angle)
            end_x = cx + radius * math.cos(angle)
            end_y = cy + radius * math.sin(angle)
            
            draw.line([(start_x, start_y), (end_x, end_y)], fill=(0, 0, 0, 255), width=tick_width)

        # 3. Bold Numbers
        font_size = radius * random.uniform(0.20, 0.28)
        font = get_bold_font(font_size)
        for i in range(1, 13):
            angle = math.radians(i * 30 - 90)
            text_r = radius * 0.76  
            tx = cx + text_r * math.cos(angle)
            ty = cy + text_r * math.sin(angle)
            left, top, right, bottom = draw.textbbox((0, 0), str(i), font=font)
            w, h = right - left, bottom - top
            draw.text((tx - w/2, ty - h/2 - (h*0.1)), str(i), fill=(0, 0, 0, 255), font=font)

        # 4. Fake Logo
        logo_text = generate_fake_logo()
        logo_font = get_bold_font(radius * 0.08)
        logo_y = cy + radius * 0.35
        left, top, right, bottom = draw.textbbox((0, 0), logo_text, font=logo_font)
        draw.text((cx - (right-left)/2, logo_y), logo_text, fill=(0, 0, 0, 255), font=logo_font)

        # --- SAVE CLEAN TARGET ---
        clean_target = img.convert("RGB")

        # 5. Refined Hands & Shadows Setup
        h, m, s = random.randint(0, 11), random.randint(0, 59), random.randint(0, 59)
        h_angle = math.radians((h % 12) * 30 + (m * 0.5) - 90)
        m_angle = math.radians((m * 6) + (s * 0.1) - 90)
        s_angle = math.radians((s * 6) - 90)

        # Lengths (Front, Back Counterweight)
        h_len, h_back = radius * 0.48, radius * 0.12
        m_len, m_back = radius * 0.68, radius * 0.15
        s_len, s_back = radius * 0.72, radius * 0.20

        # Widths (Base, Tip) - The slight difference creates an elegant taper
        h_w_base, h_w_tip = radius * 0.055, radius * 0.045
        m_w_base, m_w_tip = radius * 0.035, radius * 0.025
        s_w_base, s_w_tip = radius * 0.015, radius * 0.010

        def draw_polygon_hand(d, angle, length, back_length, base_w, tip_w, color):
            """Draws a crisp polygon hand with a subtle, realistic taper."""
            angle_perp = angle + math.pi / 2
            
            # Tip coordinates
            t_xl = cx + length * math.cos(angle) + (tip_w/2) * math.cos(angle_perp)
            t_yl = cy + length * math.sin(angle) + (tip_w/2) * math.sin(angle_perp)
            t_xr = cx + length * math.cos(angle) - (tip_w/2) * math.cos(angle_perp)
            t_yr = cy + length * math.sin(angle) - (tip_w/2) * math.sin(angle_perp)
            
            # Back coordinates (Counterweight)
            b_xl = cx + back_length * math.cos(angle + math.pi) + (base_w/2) * math.cos(angle_perp)
            b_yl = cy + back_length * math.sin(angle + math.pi) + (base_w/2) * math.sin(angle_perp)
            b_xr = cx + back_length * math.cos(angle + math.pi) - (base_w/2) * math.cos(angle_perp)
            b_yr = cy + back_length * math.sin(angle + math.pi) - (base_w/2) * math.sin(angle_perp)
            
            d.polygon([(b_xl, b_yl), (t_xl, t_yl), (t_xr, t_yr), (b_xr, b_yr)], fill=color)

        # 6. Shadow Layer (Softer and closer to the dial)
        shadow_layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadow_layer)
        shadow_color = (0, 0, 0, 50) 
        
        draw_polygon_hand(shadow_draw, h_angle, h_len, h_back, h_w_base, h_w_tip, shadow_color)
        draw_polygon_hand(shadow_draw, m_angle, m_len, m_back, m_w_base, m_w_tip, shadow_color)
        draw_polygon_hand(shadow_draw, s_angle, s_len, s_back, s_w_base, s_w_tip, shadow_color)
        
        shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=random.uniform(1.0, 2.0)))
        # Much tighter offset so they don't look like they are floating high
        img.alpha_composite(shadow_layer, dest=(random.randint(1, 3), random.randint(1, 3)))

        # 7. Solid Hands Layer
        hands_layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        hands_draw = ImageDraw.Draw(hands_layer)
        hm_color = (random.randint(15, 30), random.randint(15, 30), random.randint(15, 30), 255)
        sec_color = (random.randint(180, 220), random.randint(20, 40), random.randint(20, 40), 255)
        
        draw_polygon_hand(hands_draw, h_angle, h_len, h_back, h_w_base, h_w_tip, hm_color)
        draw_polygon_hand(hands_draw, m_angle, m_len, m_back, m_w_base, m_w_tip, hm_color)
        draw_polygon_hand(hands_draw, s_angle, s_len, s_back, s_w_base, s_w_tip, sec_color)

        # 8. Refined Center Pivot Cap (3 Layers: Base, Silver Ring, Inner Pin)
        pivot_r = radius * 0.035
        hands_draw.ellipse([cx - pivot_r, cy - pivot_r, cx + pivot_r, cy + pivot_r], fill=hm_color)
        
        inner_r = pivot_r * 0.65
        hands_draw.ellipse([cx - inner_r, cy - inner_r, cx + inner_r, cy + inner_r], fill=(210, 210, 210, 255))
        
        pin_r = pivot_r * 0.25
        hands_draw.ellipse([cx - pin_r, cy - pin_r, cx + pin_r, cy + pin_r], fill=sec_color)

        img.alpha_composite(hands_layer)
        
        # Add Reality Augmentation to dirty input
        dirty_input = img.convert("RGB")
        
        norm_cx, norm_cy, norm_r = cx / size, cy / size, radius / size
        
        return dirty_input, clean_target, norm_cx, norm_cy, norm_r

    def create_dataset(self, total_images=16000):
        print(f"Generating {total_images} Pyramid-style clocks...")
        
        for i in range(total_images):
            dirty, clean, cx, cy, radius = self.draw_single_clock()
            
            filename = f"clock_{i:06d}.jpg"
            input_path = os.path.join(self.inputs_dir, filename)
            target_path = os.path.join(self.targets_dir, filename)
            
            dirty.save(input_path, "JPEG", quality=random.randint(65, 95))
            clean.save(target_path, "JPEG", quality=100)
            
            self.labels_data.append({
                "filename": filename,
                "input_path": input_path,
                "target_path": target_path,
                "cx": cx,
                "cy": cy,
                "radius": radius
            })
            
            if (i + 1) % 500 == 0:
                print(f"Generated {i + 1} / {total_images}...")

        # Save CSV
        df = pd.DataFrame(self.labels_data)
        csv_path = os.path.join(self.output_dir, "labels.csv")
        df.to_csv(csv_path, index=False)
        print(f"Dataset complete! Saved to {self.output_dir} and {csv_path}")

# ===================== RUN =====================
if __name__ == "__main__":
    generator = PyramidDatasetGenerator(output_dir="synthetic_clocks", img_size=256)
    generator.create_dataset(total_images=12000)