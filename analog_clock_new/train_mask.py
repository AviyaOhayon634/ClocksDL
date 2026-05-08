import csv
import random
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torchvision.io as io
from tqdm import tqdm

# =========================
# Configuration
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

IMAGE_SIZE = 256       # All images resized to 256x256 for training
BATCH_SIZE = 32        # Larger batch is fine at 256x256
NUM_EPOCHS = 60
LEARNING_RATE = 1e-4

DATA_DIR = Path("data")
TRAIN_CSV = DATA_DIR / "train_labels.csv"
TEST_CSV = DATA_DIR / "test_labels.csv"

CHECKPOINT_DIR = Path("outputs/checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# ImageNet normalization — improves generalization to real-world clock photos
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def normalize(t: torch.Tensor) -> torch.Tensor:
    return (t - _MEAN) / _STD

# =========================
# Dataset
# =========================
class MaskDataset(Dataset):
    def __init__(self, csv_path: str | Path, augment: bool = True):
        self.augment = augment
        self.rows = []
        with open(csv_path, mode="r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self.rows.append(row)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]

        # --- Load input (clock with hands) ---
        img = io.read_image(row["analog_path"])
        if img.shape[0] == 4:
            img = img[:3]           # drop alpha if present
        img = img.float() / 255.0

        # --- Load mask (binary: white = hand pixels) ---
        mask = io.read_image(row["analog_without_hands_path"])
        if mask.shape[0] > 1:
            mask = mask[:1]         # keep single channel
        mask = mask.float() / 255.0

        # --- Resize to fixed training size ---
        img  = TF.resize(img,  [IMAGE_SIZE, IMAGE_SIZE], antialias=True)
        mask = TF.resize(mask, [IMAGE_SIZE, IMAGE_SIZE],
                         interpolation=TF.InterpolationMode.NEAREST)

        # --- Augmentations (input + mask together to stay aligned) ---
        if self.augment:
            # Horizontal flip
            if random.random() > 0.5:
                img  = TF.hflip(img)
                mask = TF.hflip(mask)

            # Vertical flip
            if random.random() > 0.5:
                img  = TF.vflip(img)
                mask = TF.vflip(mask)

            # Random rotation — critical: real clocks can appear tilted
            if random.random() > 0.3:
                angle = random.uniform(-25, 25)
                img  = TF.rotate(img,  angle)
                mask = TF.rotate(mask, angle,
                                 interpolation=TF.InterpolationMode.NEAREST)

            # Color jitter on input only (mask must not change)
            if random.random() > 0.4:
                img = T.ColorJitter(
                    brightness=0.35, contrast=0.35, saturation=0.2, hue=0.05
                )(img)

            # Gaussian blur on input only (simulates real-photo softness)
            if random.random() > 0.5:
                k = random.choice([3, 5])
                img = TF.gaussian_blur(img, kernel_size=k)

        # Binarize mask after any interpolation drift
        mask = (mask > 0.5).float()

        return normalize(img), mask


# =========================
# Loss Functions
# =========================
def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    p = pred.view(-1)
    t = target.view(-1)
    return 1.0 - (2.0 * (p * t).sum() + smooth) / (p.sum() + t.sum() + smooth)

def combined_loss(pred: torch.Tensor, target: torch.Tensor,
                  bce_w: float = 0.5) -> torch.Tensor:
    """BCE handles per-pixel accuracy; Dice handles class imbalance (sparse hands)."""
    bce  = F.binary_cross_entropy(pred, target)
    dice = dice_loss(pred, target)
    return bce_w * bce + (1.0 - bce_w) * dice


def iou_score(pred: torch.Tensor, target: torch.Tensor,
              threshold: float = 0.5) -> float:
    pred_bin = (pred > threshold).float()
    inter = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum() - inter
    return ((inter + 1e-6) / (union + 1e-6)).item()


# =========================
# Model — 4-level UNet with residual blocks
# =========================
class ResDoubleConv(nn.Module):
    """Double conv with a 1×1 residual shortcut for better gradient flow."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.shortcut = (nn.Conv2d(in_ch, out_ch, 1, bias=False)
                         if in_ch != out_ch else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.shortcut(x)


class MaskUNet(nn.Module):
    """
    4-level UNet for clock-hand segmentation.
    Input:  (B, 3, H, W)  — any H/W divisible by 16 works at inference time.
    Output: (B, 1, H, W)  — probability mask in [0, 1].
    """
    def __init__(self, base: int = 64):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

        # Encoder
        self.e1 = ResDoubleConv(3,       base)
        self.e2 = ResDoubleConv(base,    base*2)
        self.e3 = ResDoubleConv(base*2,  base*4)
        self.e4 = ResDoubleConv(base*4,  base*8)

        # Bottleneck
        self.bn = ResDoubleConv(base*8, base*16)

        # Decoder
        self.up4 = nn.ConvTranspose2d(base*16, base*8,  2, stride=2)
        self.d4  = ResDoubleConv(base*16, base*8)

        self.up3 = nn.ConvTranspose2d(base*8,  base*4,  2, stride=2)
        self.d3  = ResDoubleConv(base*8,  base*4)

        self.up2 = nn.ConvTranspose2d(base*4,  base*2,  2, stride=2)
        self.d2  = ResDoubleConv(base*4,  base*2)

        self.up1 = nn.ConvTranspose2d(base*2,  base,    2, stride=2)
        self.d1  = ResDoubleConv(base*2,  base)

        self.head = nn.Sequential(
            nn.Conv2d(base, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        b  = self.bn(self.pool(e4))

        d  = self.d4(torch.cat([self.up4(b),  e4], dim=1))
        d  = self.d3(torch.cat([self.up3(d),  e3], dim=1))
        d  = self.d2(torch.cat([self.up2(d),  e2], dim=1))
        d  = self.d1(torch.cat([self.up1(d),  e1], dim=1))
        return self.head(d)


# =========================
# Inference helper
# =========================
@torch.no_grad()
def predict_mask(model: nn.Module,
                 image_tensor: torch.Tensor,
                 threshold: float = 0.5,
                 pad_to_multiple: int = 16) -> torch.Tensor:
    """
    Accepts a raw RGB float tensor (C, H, W) in [0,1] of ANY size.
    Pads to multiple of pad_to_multiple, runs model, unpadds, thresholds.
    Returns a binary (1, H, W) mask at the original resolution.
    """
    model.eval()
    _, H, W = image_tensor.shape

    # Pad so dimensions are multiples of 16 (required by 4-level UNet)
    pad_h = (pad_to_multiple - H % pad_to_multiple) % pad_to_multiple
    pad_w = (pad_to_multiple - W % pad_to_multiple) % pad_to_multiple
    x = F.pad(image_tensor.unsqueeze(0), (0, pad_w, 0, pad_h))
    x = normalize(x.squeeze(0)).unsqueeze(0).to(next(model.parameters()).device)

    prob = model(x)                              # (1, 1, H+pad_h, W+pad_w)
    prob = prob[:, :, :H, :W]                   # remove padding
    return (prob.squeeze(0) > threshold).float()


# =========================
# Training Loop
# =========================
if __name__ == "__main__":
    train_ds = MaskDataset(TRAIN_CSV, augment=True)
    test_ds  = MaskDataset(TEST_CSV,  augment=False)

    # num_workers=0 is safest on Windows; increase on Linux if I/O is the bottleneck
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=8, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    model     = MaskUNet(base=64).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
        eta_min=1e-6
    )

    best_iou = 0.0

    # Early stopping
    patience = 10
    no_improve = 0

    print(f"Starting Mask Training — {NUM_EPOCHS} epochs | device: {DEVICE}")
    print(f"Train: {len(train_ds)} samples | Test: {len(test_ds)} samples")

    for epoch in range(1, NUM_EPOCHS + 1):

        # ---------- Train ----------
        model.train()
        train_loss = 0.0

        train_pbar = tqdm(
            train_loader,
            desc=f"Epoch [{epoch}/{NUM_EPOCHS}] Train",
            leave=False
        )

        for imgs, masks in train_pbar:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

            optimizer.zero_grad()

            preds = model(imgs)
            loss  = combined_loss(preds, masks)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)

            train_pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss /= len(train_ds)

        # ---------- Evaluate ----------
        model.eval()
        test_loss = 0.0
        test_iou  = 0.0

        test_pbar = tqdm(
            test_loader,
            desc=f"Epoch [{epoch}/{NUM_EPOCHS}] Eval",
            leave=False
        )

        with torch.no_grad():
            for imgs, masks in test_pbar:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

                preds = model(imgs)

                loss = combined_loss(preds, masks)

                test_loss += loss.item() * imgs.size(0)
                test_iou  += iou_score(preds, masks) * imgs.size(0)

                test_pbar.set_postfix(loss=f"{loss.item():.4f}")

        test_loss /= len(test_ds)
        test_iou  /= len(test_ds)

        scheduler.step()

        print(
            f"Epoch [{epoch:02d}/{NUM_EPOCHS}] | "
            f"Train Loss: {train_loss:.5f} | "
            f"Test Loss: {test_loss:.5f} | "
            f"IoU: {test_iou:.4f}"
        )

        # ---------- Save best ----------
        if test_iou > best_iou:
            best_iou = test_iou
            no_improve = 0

            torch.save(
                model.state_dict(),
                CHECKPOINT_DIR / "best_mask_unet.pth"
            )

            print(f"  >>> Saved best model (IoU = {best_iou:.4f})")

        else:
            no_improve += 1

            print(f"  >>> No improvement ({no_improve}/{patience})")

            if no_improve >= patience:
                print("Early stopping triggered.")
                break

    print(f"\nTraining complete. Best IoU: {best_iou:.4f}")