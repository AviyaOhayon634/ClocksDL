import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.io as io
import torchvision.transforms.functional as TF

from pathlib import Path
from PIL import Image

from visualize_geometry import get_clock_geometry


# =========================
# Config
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_PATH = "outputs/checkpoints/best_mask_unet.pth"

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def normalize(t):
    return (t - _MEAN) / _STD


# =========================
# Model
# =========================
class ResDoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

        self.shortcut = (
            nn.Conv2d(in_ch, out_ch, 1, bias=False)
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x):
        return self.block(x) + self.shortcut(x)


class MaskUNet(nn.Module):
    def __init__(self, base=64):
        super().__init__()

        self.pool = nn.MaxPool2d(2)

        self.e1 = ResDoubleConv(3, base)
        self.e2 = ResDoubleConv(base, base * 2)
        self.e3 = ResDoubleConv(base * 2, base * 4)
        self.e4 = ResDoubleConv(base * 4, base * 8)

        self.bn = ResDoubleConv(base * 8, base * 16)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.d4  = ResDoubleConv(base * 16, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.d3  = ResDoubleConv(base * 8, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.d2  = ResDoubleConv(base * 4, base * 2)

        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.d1  = ResDoubleConv(base * 2, base)

        self.head = nn.Sequential(
            nn.Conv2d(base, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))

        b = self.bn(self.pool(e4))

        d = self.d4(torch.cat([self.up4(b), e4], 1))
        d = self.d3(torch.cat([self.up3(d), e3], 1))
        d = self.d2(torch.cat([self.up2(d), e2], 1))
        d = self.d1(torch.cat([self.up1(d), e1], 1))

        return self.head(d)


# =========================
# Clean Mask
# =========================
def clean_mask(
    raw_mask_uint8,
    center,
    big_blob_size=200,
    max_center_dist_ratio=0.35,
    close_kernel=7,
    bridge_thickness=2,
):
    H, W = raw_mask_uint8.shape

    cx, cy = center

    center_pt = (
        int(round(cx)),
        int(round(cy))
    )

    half_diag = np.sqrt(cx**2 + cy**2)

    max_dist = max_center_dist_ratio * half_diag

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        raw_mask_uint8,
        connectivity=8
    )

    if num_labels <= 1:
        return raw_mask_uint8.copy()

    anchor_id = -1
    anchor_dist = float("inf")

    for i in range(1, num_labels):

        bx, by = centroids[i]

        d = np.sqrt((bx - cx) ** 2 + (by - cy) ** 2)

        if d < anchor_dist:
            anchor_dist = d
            anchor_id = i

    result = np.zeros_like(raw_mask_uint8)

    result[labels == anchor_id] = 255

    for i in range(1, num_labels):

        if i == anchor_id:
            continue

        area = stats[i, cv2.CC_STAT_AREA]

        bx, by = centroids[i]

        dist = np.sqrt((bx - cx) ** 2 + (by - cy) ** 2)

        if area >= big_blob_size and dist <= max_dist:

            result[labels == i] = 255

            blob_pt = (
                int(round(bx)),
                int(round(by))
            )

            cv2.line(
                result,
                blob_pt,
                center_pt,
                color=255,
                thickness=bridge_thickness
            )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (close_kernel, close_kernel)
    )

    cleaned = cv2.morphologyEx(
        result,
        cv2.MORPH_CLOSE,
        kernel
    )

    return cleaned


# =========================
# Load model once
# =========================
model = MaskUNet(base=64).to(DEVICE)

model.load_state_dict(
    torch.load(CHECKPOINT_PATH, map_location=DEVICE)
)

model.eval()


# =========================
# MAIN FUNCTION
# =========================
@torch.no_grad()
def get_mask(
    image_path,
    save_path=None,
    threshold=0.5,
):
    """
    Returns:
        mask_np   -> numpy mask image
        mask_pil  -> PIL mask image
    """

    # -------------------------
    # Geometry
    # -------------------------
    pil_img = Image.open(image_path)

    center, radius = get_clock_geometry(pil_img)

    # -------------------------
    # Read image
    # -------------------------
    img_t = io.read_image(str(image_path))

    if img_t.shape[0] == 4:
        img_t = img_t[:3]

    img_float = img_t.float() / 255.0

    _, H, W = img_float.shape

    # -------------------------
    # Resize + normalize
    # -------------------------
    img_256 = TF.resize(
        img_float,
        [256, 256],
        antialias=True
    )

    x = normalize(img_256).unsqueeze(0).to(DEVICE)

    # -------------------------
    # Predict
    # -------------------------
    prob_256 = model(x)

    prob = TF.resize(
        prob_256,
        [H, W],
        interpolation=TF.InterpolationMode.BILINEAR
    )

    raw_mask = (
        (prob.squeeze() > threshold)
        .float()
        .cpu()
        .numpy() * 255
    ).astype(np.uint8)

    # -------------------------
    # Clean
    # -------------------------
    mask_np = clean_mask(
        raw_mask,
        center=center,
        big_blob_size=30,
        max_center_dist_ratio=0.35,
        close_kernel=7,
        bridge_thickness=10,
    )

    # -------------------------
    # Convert to PIL
    # -------------------------
    mask_pil = Image.fromarray(mask_np)

    # -------------------------
    # Optional save
    # -------------------------
    if save_path is not None:
        mask_pil.save(save_path)
        print(f"Mask saved -> {save_path}")

    return mask_np, mask_pil #use mask_np for geometry estimation, mask_pil for visualization