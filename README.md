# ClocksDL

A deep learning pipeline for analog clock understanding — detecting clock hands, generating segmentation masks, and removing hands from clock images using inpainting.

---

## What This Project Does

1. **Generates synthetic training data** — renders analog clock images with and without hands across 50+ visual styles
2. **Trains a UNet segmentation model** — learns to detect clock hands (the mask)
3. **Removes clock hands from real images** — uses the trained mask + LaMa inpainting to produce a clean clock face

---

## Project Structure

```
ClocksDL/
├── analog_clock_new/
│   ├── analog_mask_new.py       # Synthetic data generator (27,000 clock pairs)
│   ├── train_mask.py            # UNet training script
│   ├── mask_generator.py        # Inference: generates mask for a given clock image
│   ├── visualize_geometry.py    # Hough circle detection → clock center & radius
│   ├── inpaint_lama.py          # Full pipeline: mask → dilate → LaMa inpainting
│   └── outputs/
│       └── checkpoints/
│           └── best_mask_unet.pth   # Trained model (stored via Git LFS)
├── requirements.txt
└── README.md
```

---

## Model Architecture

**MaskUNet** — a 4-level residual UNet:
- Input: `(B, 3, H, W)` RGB clock image
- Output: `(B, 1, H, W)` probability mask (white = hand pixels)
- Residual double-conv blocks at each level for better gradient flow
- Trained with combined BCE + Dice loss to handle class imbalance (hands are sparse)
- ImageNet normalization for better generalization to real-world photos

---

## Training

### 1. Generate synthetic data
```bash
python analog_clock_new/analog_mask_new.py
```
Generates 27,000 clock image pairs (with hands / without hands) across 50 visual styles.

### 2. Train the model
```bash
python analog_clock_new/train_mask.py
```
- **Epochs:** 60 (with early stopping, patience=10)
- **Batch size:** 32
- **Image size:** 256×256
- **Optimizer:** AdamW (lr=1e-4, weight_decay=1e-4)
- **Scheduler:** CosineAnnealingLR
- **Metric:** IoU score
- Best checkpoint saved to `outputs/checkpoints/best_mask_unet.pth`

---

## Inference — Remove Hands from a Clock Image

```bash
python analog_clock_new/inpaint_lama.py
```

The pipeline:
1. Detects clock geometry (center + radius) via Hough circles
2. Runs the UNet to generate a hand mask
3. Dilates the mask to ensure full coverage
4. Runs LaMa inpainting to fill the masked area
5. Saves the result as `<image_name>_no_hands.png`

To use on your own image, change `IMAGE_PATH` in `inpaint_lama.py`:
```python
IMAGE_PATH = r"path/to/your/clock.jpg"
```

---

## Public API

### `visualize_geometry.py`
```python
from analog_clock_new.visualize_geometry import get_clock_geometry, draw_hands

center, radius = get_clock_geometry(image)          # returns (cx, cy), radius in pixels
result = draw_hands(image, center, radius, "14:30:00")  # draws hands for given time
```

### `mask_generator.py`
```python
from analog_clock_new.mask_generator import get_mask

mask_np, mask_pil = get_mask("clock.jpg")   # returns numpy array + PIL image
```

### `inpaint_lama.py`
```python
from analog_clock_new.inpaint_lama import remove_hands_lama

result = remove_hands_lama("clock.jpg")   # returns PIL image without hands
```

---

## Setup

### Requirements
- Python 3.12.4
- See `requirements.txt` for all dependencies

```bash
pip install -r requirements.txt
```

### Git LFS
Model checkpoints are stored with Git LFS. After cloning, run:
```bash
git lfs install
git lfs pull
```

---

## Data Augmentations (Training)

| Augmentation | Probability |
|---|---|
| Horizontal flip | 50% |
| Vertical flip | 50% |
| Random rotation ±25° | 70% |
| Color jitter (brightness, contrast, saturation, hue) | 60% |
| Gaussian blur | 50% |
