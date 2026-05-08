import math
import cv2
import numpy as np
from PIL import Image, ImageDraw


COLOR_HOUR   = (20,  20,  20)
COLOR_MINUTE = (20,  20,  20)
COLOR_SECOND = (210, 30,  30)
COLOR_CENTER = (20,  20,  20)


def get_clock_geometry(image: Image.Image) -> tuple[tuple[float, float], float]:
    """
    Returns (center, radius) in pixels using Hough circle detection.
    center = (cx, cy), radius = float
    """
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
        cx, cy, r = W * 0.5, H * 0.5, min(W, H) * 0.45
    else:
        cx, cy, r = circles[0][0]

    center = (float(cx), float(cy))
    radius = float(r)
    return center, radius


def draw_hands(image: Image.Image, center: tuple[float, float], radius: float, time_str: str) -> Image.Image:
    """
    Draws clock hands on the image for the given time.
    time_str format: "HH:MM" or "HH:MM:SS"
    Returns a new image with the hands drawn.
    """
    parts = time_str.strip().split(":")
    h = int(parts[0])
    m = int(parts[1])
    s = int(parts[2]) if len(parts) == 3 else None

    result = image.convert("RGB").copy()
    cx, cy = center
    draw = ImageDraw.Draw(result)

    def tip(angle_deg, length):
        angle = math.radians(angle_deg - 90)
        return (cx + length * math.cos(angle), cy + length * math.sin(angle))

    h_angle = (h % 12) * 30 + m * 0.5 + (s or 0) * (0.5 / 60)
    m_angle = m * 6 + (s or 0) * 0.1
    s_angle = s * 6 if s is not None else None

    face_r = radius * 0.75

    draw.line([cx, cy, *tip(h_angle, face_r * 0.60)], fill=COLOR_HOUR,   width=max(3, int(face_r * 0.030)))
    draw.line([cx, cy, *tip(m_angle, face_r * 0.88)], fill=COLOR_MINUTE, width=max(2, int(face_r * 0.020)))
    if s is not None:
        draw.line([cx, cy, *tip(s_angle, face_r * 0.92)], fill=COLOR_SECOND, width=max(1, int(face_r * 0.010)))

    cap = max(3, int(radius * 0.04))
    draw.ellipse([cx - cap, cy - cap, cx + cap, cy + cap], fill=COLOR_CENTER)

    return result


def main():
    from PIL import Image as _Image

    IMAGE_PATH  = r"download.jpg"
    TIME        = "21:17:42"
    OUTPUT_PATH = None

    image = Image.open(IMAGE_PATH)
    center, radius = get_clock_geometry(image)
    print(f"Center : {center}")
    print(f"Radius : {radius:.1f} px")

    result = draw_hands(image, center, radius, TIME)

    if OUTPUT_PATH:
        result.save(OUTPUT_PATH)
        print(f"Saved  : {OUTPUT_PATH}")
    else:
        result.show()


if __name__ == "__main__":
    main()
