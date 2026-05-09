import math
import cv2
import numpy as np

from PIL import Image, ImageDraw


# =========================================================
# Colors
# =========================================================
COLOR_HOUR   = (20, 20, 20)
COLOR_MINUTE = (20, 20, 20)
COLOR_SECOND = (210, 30, 30)
COLOR_CENTER = (20, 20, 20)


# =========================================================
# Detect clock geometry
# =========================================================
def get_clock_geometry(
    image: Image.Image
) -> tuple[tuple[float, float], float]:
    """
    Detects clock center and radius using Hough circles.

    Returns:
        center -> (cx, cy)
        radius -> float
    """

    W, H = image.size

    gray = cv2.cvtColor(
        np.array(image.convert("RGB")),
        cv2.COLOR_RGB2GRAY
    )

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

    # Fallback
    if circles is None:

        print("Hough failed -> using image center fallback")

        cx = W * 0.5
        cy = H * 0.5
        r  = min(W, H) * 0.45

    else:

        cx, cy, r = circles[0][0]

    center = (float(cx), float(cy))
    radius = float(r)

    return center, radius


# =========================================================
# Draw hands
# =========================================================
def draw_hands(
    image: Image.Image,
    center: tuple[float, float],
    radius: float,
    time_str: str,
) -> Image.Image:
    """
    Draw realistic clock hands on image.
    """

    parts = time_str.strip().split(":")

    h = int(parts[0])
    m = int(parts[1])

    s = int(parts[2]) if len(parts) == 3 else None

    result = image.convert("RGB").copy()

    cx, cy = center

    draw = ImageDraw.Draw(result)

    # =====================================================
    # Helper
    # =====================================================
    def tip(angle_deg, length):

        angle = math.radians(angle_deg - 90)

        return (
            cx + length * math.cos(angle),
            cy + length * math.sin(angle)
        )

    # =====================================================
    # Angles
    # =====================================================
    h_angle = (
        (h % 12) * 30
        + m * 0.5
        + (s or 0) * (0.5 / 60)
    )

    m_angle = (
        m * 6
        + (s or 0) * 0.1
    )

    s_angle = s * 6 if s is not None else None

    # =====================================================
    # Sizes
    # =====================================================
    face_r = radius * 0.75

    # Hand lengths
    hour_len   = face_r * 0.58
    minute_len = face_r * 0.90
    second_len = face_r * 0.96

    # Hand thicknesses
    hour_w   = max(8, int(radius * 0.045))
    minute_w = max(5, int(radius * 0.028))
    second_w = max(2, int(radius * 0.015))

    # =====================================================
    # Draw hour hand
    # =====================================================
    hx, hy = tip(h_angle, hour_len)

    draw.line(
        [cx, cy, hx, hy],
        fill=(25, 25, 25),
        width=hour_w
    )

    # =====================================================
    # Draw minute hand
    # =====================================================
    mx, my = tip(m_angle, minute_len)

    draw.line(
        [cx, cy, mx, my],
        fill=(35, 35, 35),
        width=minute_w
    )

    # =====================================================
    # Draw second hand
    # =====================================================
    if s is not None:

        sx, sy = tip(s_angle, second_len)

        draw.line(
            [cx, cy, sx, sy],
            fill=(220, 60, 60),
            width=second_w
        )

        # Small counterweight behind center
        back_len = radius * 0.15

        bx, by = tip(s_angle + 180, back_len)

        draw.line(
            [cx, cy, bx, by],
            fill=(220, 60, 60),
            width=max(1, second_w)
        )

    # =====================================================
    # Center cap
    # =====================================================
    cap_outer = max(7, int(radius * 0.035))
    cap_inner = int(cap_outer * 0.45)

    # Outer ring
    draw.ellipse(
        [
            cx - cap_outer,
            cy - cap_outer,
            cx + cap_outer,
            cy + cap_outer
        ],
        fill=(30, 30, 30)
    )

    # Inner metallic pin
    draw.ellipse(
        [
            cx - cap_inner,
            cy - cap_inner,
            cx + cap_inner,
            cy + cap_inner
        ],
        fill=(180, 180, 180)
    )

    return result


# =========================================================
# MAIN WRAPPER
# =========================================================
def add_clock_hands(
    image,
    time_str,
    save_path=None,
):
    """
    Adds clock hands automatically.

    image:
        - image path
        - PIL image

    time_str:
        HH:MM
        HH:MM:SS

    returns:
        PIL image
    """

    # -----------------------------------------------------
    # Load image
    # -----------------------------------------------------
    if isinstance(image, str):

        image = Image.open(image)

    # -----------------------------------------------------
    # Detect geometry
    # -----------------------------------------------------
    center, radius = get_clock_geometry(image)

    # -----------------------------------------------------
    # Draw hands
    # -----------------------------------------------------
    result = draw_hands(
        image=image,
        center=center,
        radius=radius,
        time_str=time_str,
    )

    # -----------------------------------------------------
    # Optional save
    # -----------------------------------------------------
    if save_path is not None:

        result.save(save_path)

        # print(f"Saved -> {save_path}")

    return result


# =========================================================
# Example
# =========================================================
if __name__ == "__main__":

    IMAGE_PATH = "analog_clock_new/2.jpg"

    result = add_clock_hands(
        image=IMAGE_PATH,
        time_str="21:17:42",
        save_path="clock_with_hands.png"
    )
    
    result.show()