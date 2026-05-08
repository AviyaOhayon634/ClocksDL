import math
import cv2
import numpy as np
from PIL import Image, ImageDraw


IMAGE_PATH  = r"download.jpg"
TIME        = "21:17:42"  # פורמט: "HH:MM" או "HH:MM:SS"
OUTPUT_PATH = None  # None = הצג על המסך, או שנה ל: r"result.png"

COLOR_HOUR   = (20,  20,  20)   # שחור
COLOR_MINUTE = (20,  20,  20)   # שחור
COLOR_SECOND = (210, 30,  30)   # אדום
COLOR_CENTER = (20,  20,  20)   # שחור


def predict_hough(image: Image.Image):
    W, H = image.size
    gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min(W, H) * 0.5,
        param1=80,
        param2=40,
        minRadius=int(min(W, H) * 0.25),
        maxRadius=int(min(W, H) * 0.55),
    )

    if circles is None:
        print("Hough לא מצא עיגול — משתמש במרכז התמונה כ-fallback")
        return 0.5, 0.5, 0.45

    cx_px, cy_px, r_px = circles[0][0]
    return float(cx_px) / W, float(cy_px) / H, float(r_px) / W


def parse_time(time_str: str):
    parts = time_str.strip().split(":")
    h = int(parts[0])
    m = int(parts[1])
    s = int(parts[2]) if len(parts) == 3 else None
    return h, m, s


def draw_geometry(image: Image.Image, cx_norm, cy_norm, r_norm, h=None, m=None, s=None) -> Image.Image:
    result = image.convert("RGB").copy()
    W, H = result.size

    cx = cx_norm * W
    cy = cy_norm * H
    radius = r_norm * W

    draw = ImageDraw.Draw(result)

    # היקף השעון
    draw.ellipse(
        [cx - radius, cy - radius, cx + radius, cy + radius],
        outline=(255, 0, 0),
        width=max(2, int(radius * 0.02)),
    )

    # מרכז + קרוס
    dot = max(4, int(radius * 0.04))
    draw.ellipse([cx - dot, cy - dot, cx + dot, cy + dot], fill=(255, 0, 0))
    arm = max(8, int(radius * 0.06))
    draw.line([cx - arm, cy, cx + arm, cy], fill=(255, 0, 0), width=2)
    draw.line([cx, cy - arm, cx, cy + arm], fill=(255, 0, 0), width=2)

    # מחוגים
    if h is not None:
        def tip(angle_deg, length):
            angle = math.radians(angle_deg - 90)
            return (cx + length * math.cos(angle), cy + length * math.sin(angle))

        h_angle = (h % 12) * 30 + m * 0.5 + (s or 0) * (0.5 / 60)
        m_angle = m * 6 + (s or 0) * 0.1
        s_angle = s * 6 if s is not None else None

        face_r = radius * 0.75

        # מחוג שעות — קצר ועבה
        draw.line([cx, cy, *tip(h_angle, face_r * 0.60)], fill=COLOR_HOUR,   width=max(3, int(face_r * 0.030)))
        # מחוג דקות — ארוך ובינוני
        draw.line([cx, cy, *tip(m_angle, face_r * 0.88)], fill=COLOR_MINUTE, width=max(2, int(face_r * 0.020)))
        # מחוג שניות — רק אם סופקו שניות
        if s is not None:
            draw.line([cx, cy, *tip(s_angle, face_r * 0.92)], fill=COLOR_SECOND, width=max(1, int(face_r * 0.010)))

        # כיסוי מרכז
        cap = max(3, int(radius * 0.04))
        draw.ellipse([cx - cap, cy - cap, cx + cap, cy + cap], fill=COLOR_CENTER)

    return result


def main():
    image = Image.open(IMAGE_PATH)
    W, H = image.size

    cx, cy, r = predict_hough(image)
    print(f"Center : ({cx * W:.1f}, {cy * H:.1f}) px")
    print(f"Radius : {r * W:.1f} px")

    h, m, s = parse_time(TIME)
    time_str = f"{h:02d}:{m:02d}" + (f":{s:02d}" if s is not None else "")
    print(f"Time   : {time_str}")

    result = draw_geometry(image, cx, cy, r, h, m, s)

    if OUTPUT_PATH:
        result.save(OUTPUT_PATH)
        print(f"Saved  : {OUTPUT_PATH}")
    else:
        result.show()


if __name__ == "__main__":
    main()
