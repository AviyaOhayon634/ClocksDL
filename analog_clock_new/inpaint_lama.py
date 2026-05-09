import sys
import cv2
import numpy as np
from simple_lama_inpainting import SimpleLama
from PIL import Image
from pathlib import Path

import os
sys.path.insert(0, r"C:\Users\amitf\Documents\ClocksDL")
os.chdir(r"C:\Users\amitf\Documents\ClocksDL\analog_clock_new")

IMAGE_PATH = r"C:\Users\amitf\Documents\ClocksDL\analog_clock_new\analog_watch_3.png"


def remove_hands_lama(image_path: str, dilate_px: int = 20, max_size: int = 512) -> Image.Image:
    # Step 1: generate mask using the trained model
    print("Generating mask...")
    from analog_clock_new.mask_generator import get_mask
    mask_np, _ = get_mask(image_path)
    print(f"Mask white pixels: {(mask_np > 127).sum()}")

    # Step 2: dilate mask
    kernel       = np.ones((dilate_px, dilate_px), np.uint8)
    mask_dilated = cv2.dilate(mask_np, kernel, iterations=1)

    # Step 3: resize to max_size for faster LaMa
    image     = Image.open(image_path).convert("RGB")
    orig_size = image.size
    scale     = max_size / max(orig_size)
    if scale < 1.0:
        new_size   = (int(orig_size[0] * scale), int(orig_size[1] * scale))
        image      = image.resize(new_size, Image.LANCZOS)
        mask_small = Image.fromarray(mask_dilated).resize(new_size, Image.NEAREST)
        print(f"Resized {orig_size} → {new_size}")
    else:
        mask_small = Image.fromarray(mask_dilated)

    # Step 4: LaMa inpainting on small image
    print("Running LaMa inpainting...")
    lama         = SimpleLama()
    result_small = lama(image, mask_small)

    # Step 5: resize result back to original size
    if scale < 1.0:
        result_small = result_small.resize(orig_size, Image.LANCZOS)

    return result_small


if __name__ == "__main__":
    import time
    t0     = time.time()
    result = remove_hands_lama(IMAGE_PATH)
    print(f"Done in {time.time() - t0:.2f}s")

    out = Path(IMAGE_PATH).with_stem(Path(IMAGE_PATH).stem + "_no_hands").with_suffix(".png")
    result.save(str(out))
    print(f"Saved → {out}")
    result.show()
