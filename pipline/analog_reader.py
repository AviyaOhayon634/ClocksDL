from itertools import product

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image


_SEGMENT_ORDER = ("top", "upper_left", "upper_right", "middle", "lower_left", "lower_right", "bottom")
_DIGIT_PATTERNS = {
    (1, 1, 1, 0, 1, 1, 1): 0,
    (0, 0, 1, 0, 0, 1, 0): 1,
    (1, 0, 1, 1, 1, 0, 1): 2,
    (1, 0, 1, 1, 0, 1, 1): 3,
    (0, 1, 1, 1, 0, 1, 0): 4,
    (1, 1, 0, 1, 0, 1, 1): 5,
    (1, 1, 0, 1, 1, 1, 1): 6,
    (1, 0, 1, 0, 0, 1, 0): 7,
    (1, 1, 1, 1, 1, 1, 1): 8,
    (1, 1, 1, 1, 0, 1, 1): 9,
}


def _red_segment_mask(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    red = rgb[..., 0]
    green = rgb[..., 1]
    blue = rgb[..., 2]
    red_mask = (red > 80) & ((red - green) > 45) & ((red - blue) > 45)
    blue_mask = (blue > 80) & ((blue - red) > 35) & ((blue - green) > 20)
    return red_mask | blue_mask


def _find_runs(active_values: np.ndarray) -> list[tuple[int, int]]:
    runs = []
    start = None
    for idx, is_active in enumerate(active_values.tolist()):
        if is_active and start is None:
            start = idx
        elif not is_active and start is not None:
            if idx - start >= 3:
                runs.append((start, idx))
            start = None
    if start is not None and len(active_values) - start >= 3:
        runs.append((start, len(active_values)))
    return runs


def _merge_digit_runs(runs: list[tuple[int, int]], target_count: int = 6) -> list[tuple[int, int]]:
    merged_runs = list(runs)
    while len(merged_runs) > target_count:
        best_index = None
        best_score = None
        for idx in range(len(merged_runs) - 1):
            left_start, left_end = merged_runs[idx]
            right_start, right_end = merged_runs[idx + 1]
            left_width = left_end - left_start
            right_width = right_end - right_start
            gap = right_start - left_end
            combined_width = right_end - left_start

            if gap > 16 or combined_width > 65:
                continue

            # Prefer merges that look like one split digit, not two normal digits.
            score = (combined_width, gap, max(left_width, right_width))
            if best_score is None or score < best_score:
                best_score = score
                best_index = idx

        if best_index is None:
            break

        left_start, _ = merged_runs[best_index]
        _, right_end = merged_runs[best_index + 1]
        merged_runs[best_index:best_index + 2] = [(left_start, right_end)]

    return merged_runs


def _sample_region(mask: np.ndarray, x0: float, x1: float, y0: float, y1: float) -> float:
    height, width = mask.shape
    left = max(0, min(width - 1, int(round(width * x0))))
    right = max(left + 1, min(width, int(round(width * x1))))
    top = max(0, min(height - 1, int(round(height * y0))))
    bottom = max(top + 1, min(height, int(round(height * y1))))
    return float(mask[top:bottom, left:right].mean())


def _decode_digit_mask(mask: np.ndarray) -> int | None:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None

    mask = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    height, width = mask.shape
    min_width = max(width, int(round(height * 0.45)))
    if min_width > width:
        pad_left = (min_width - width) // 2
        pad_right = min_width - width - pad_left
        mask = np.pad(mask, ((0, 0), (pad_left, pad_right)), constant_values=False)

    segments = {
        "top": _sample_region(mask, 0.20, 0.80, 0.03, 0.18),
        "upper_left": _sample_region(mask, 0.03, 0.24, 0.16, 0.46),
        "upper_right": _sample_region(mask, 0.76, 0.97, 0.16, 0.46),
        "middle": _sample_region(mask, 0.20, 0.80, 0.42, 0.58),
        "lower_left": _sample_region(mask, 0.03, 0.24, 0.54, 0.86),
        "lower_right": _sample_region(mask, 0.76, 0.97, 0.54, 0.86),
        "bottom": _sample_region(mask, 0.20, 0.80, 0.82, 0.97),
    }
    height, width = mask.shape
    aspect_ratio = width / max(height, 1)
    if aspect_ratio < 0.45:
        upper_mass = float(mask[: max(1, height // 3), :].mean())
        lower_mass = float(mask[max(0, (2 * height) // 3):, :].mean())
        top_fill = segments["top"]
        if upper_mass > lower_mass * 1.15 or top_fill > 0.18:
            return 7
        return 1

    pattern = tuple(int(segments[name] > 0.25) for name in _SEGMENT_ORDER)
    digit = _DIGIT_PATTERNS.get(pattern)
    if digit is not None:
        return digit

    best_digit = None
    best_distance = len(_SEGMENT_ORDER) + 1
    for known_pattern, known_digit in _DIGIT_PATTERNS.items():
        distance = sum(int(left != right) for left, right in zip(pattern, known_pattern))
        if distance < best_distance:
            best_distance = distance
            best_digit = known_digit
    if best_distance <= 2:
        return best_digit
    return None


def read_time_from_digital_image(image: Image.Image) -> tuple[int, int, int] | None:
    mask = _red_segment_mask(image)
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None

    pad = 3
    top = max(0, ys.min() - pad)
    bottom = min(mask.shape[0], ys.max() + pad + 1)
    left = max(0, xs.min() - pad)
    right = min(mask.shape[1], xs.max() + pad + 1)
    cropped = mask[top:bottom, left:right]

    column_profile = cropped.sum(axis=0)
    active_columns = column_profile > max(2, column_profile.max() * 0.18)
    runs = _find_runs(active_columns)

    merged_runs = []
    idx = 0
    while idx < len(runs):
        run_left, run_right = runs[idx]
        width = run_right - run_left
        if idx + 1 < len(runs):
            next_left, next_right = runs[idx + 1]
            next_width = next_right - next_left
            gap = next_left - run_right
            if width <= 10 and next_width <= 10 and gap <= 10:
                merged_runs.append((run_left, next_right))
                idx += 2
                continue
        merged_runs.append((run_left, run_right))
        idx += 1
    runs = merged_runs

    digit_runs = []
    for run_left, run_right in runs:
        region = cropped[:, run_left:run_right]
        width = run_right - run_left
        area = int(region.sum())
        if width <= 6 and area < 140:
            continue
        digit_runs.append((run_left, run_right))

    if len(digit_runs) > 6:
        digit_runs = _merge_digit_runs(digit_runs, target_count=6)

    if len(digit_runs) != 6:
        return None

    candidate_lists = []
    for run_left, run_right in digit_runs:
        digit_mask = cropped[:, run_left:run_right]
        ys_digit, xs_digit = np.where(digit_mask)
        if len(xs_digit) == 0 or len(ys_digit) == 0:
            return None
        digit_mask = digit_mask[ys_digit.min():ys_digit.max() + 1, xs_digit.min():xs_digit.max() + 1]
        height, width = digit_mask.shape
        aspect_ratio = width / max(height, 1)

        if aspect_ratio < 0.45:
            top_fill = float(digit_mask[: max(1, height // 5), :].mean())
            upper_mass = float(digit_mask[: max(1, height // 3), :].mean())
            lower_mass = float(digit_mask[max(0, (2 * height) // 3):, :].mean())
            score_7 = top_fill + max(0.0, upper_mass - lower_mass)
            score_1 = lower_mass + max(0.0, 0.15 - top_fill)
            if score_7 >= score_1:
                candidate_lists.append([(7, score_7), (1, score_1)])
            else:
                candidate_lists.append([(1, score_1), (7, score_7)])
            continue

        digit = _decode_digit_mask(digit_mask)
        if digit is None:
            return None
        candidate_lists.append([(digit, 1.0)])

    best_time = None
    best_score = float('-inf')
    for candidate_combo in product(*candidate_lists):
        digits = [digit for digit, _ in candidate_combo]
        hours = digits[0] * 10 + digits[1]
        minutes = digits[2] * 10 + digits[3]
        seconds = digits[4] * 10 + digits[5]
        if hours > 23 or minutes > 59 or seconds > 59:
            continue
        score = sum(score for _, score in candidate_combo)
        if score > best_score:
            best_score = score
            best_time = (hours, minutes, seconds)

    return best_time

class DigitalClockClassifier(nn.Module):
    """
    ResNet18 backbone with 3 independent classification heads:
      - hour_head:   24 classes  (0–23)
      - minute_head: 60 classes  (0–59)
      - second_head: 60 classes  (0–59)

    Why classification instead of regression?
    Regression treats 12:59 and 13:00 as "close" in loss space,
    even though they are very different clock positions.
    Classification treats each digit combination as an independent class,
    so the model can learn crisp decision boundaries.
    """
    def __init__(self):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        feat = base.fc.in_features          # 512 for ResNet18
        base.fc = nn.Identity()             # Remove the original head
        self.backbone = base

        # Shared bottleneck (helps all 3 heads)
        self.bottleneck = nn.Sequential(
            nn.Linear(feat, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.hour_head   = nn.Linear(256, 24)
        self.minute_head = nn.Linear(256, 60)
        self.second_head = nn.Linear(256, 60)

    def forward(self, x):
        f = self.backbone(x)
        f = self.bottleneck(f)
        return self.hour_head(f), self.minute_head(f), self.second_head(f)

    def predict_time(self, x):
        """Returns integer (h, m, s) — use this at inference time."""
        h_logits, m_logits, s_logits = self.forward(x)
        h = h_logits.argmax(dim=1)
        m = m_logits.argmax(dim=1)
        s = s_logits.argmax(dim=1)
        return h, m, s