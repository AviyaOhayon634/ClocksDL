import cv2
import numpy as np

from PIL import Image
from simple_lama_inpainting import SimpleLama


# =========================================================
# MAIN FUNCTION
# =========================================================
def inpaint_image(
    image,
    mask_np,
    dilate_px: int = 20,
    max_size: int = 512,
):
    """
    Parameters
    ----------
    image:
        Either:
        - image path (str)
        - PIL image
        - numpy image

    mask_np:
        Binary mask as numpy array (uint8)
        0   = keep
        255 = inpaint

    Returns
    -------
    result_pil:
        PIL.Image.Image
    """

    # =====================================================
    # Convert image to PIL
    # =====================================================
    if isinstance(image, str):

        image_pil = Image.open(image).convert("RGB")

    elif isinstance(image, np.ndarray):

        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        image_pil = Image.fromarray(image)

    elif isinstance(image, Image.Image):

        image_pil = image.convert("RGB")

    else:
        raise TypeError("Unsupported image type")


    # =====================================================
    # Ensure mask is uint8
    # =====================================================
    if mask_np.dtype != np.uint8:
        mask_np = mask_np.astype(np.uint8)


    # =====================================================
    # Dilate mask
    # =====================================================
    kernel = np.ones((dilate_px, dilate_px), np.uint8)

    mask_dilated = cv2.dilate(
        mask_np,
        kernel,
        iterations=1
    )


    # =====================================================
    # Resize for faster LaMa
    # =====================================================
    orig_size = image_pil.size

    scale = max_size / max(orig_size)

    if scale < 1.0:

        new_size = (
            int(orig_size[0] * scale),
            int(orig_size[1] * scale)
        )

        image_small = image_pil.resize(
            new_size,
            Image.LANCZOS
        )

        mask_small = Image.fromarray(mask_dilated).resize(
            new_size,
            Image.NEAREST
        )

    else:

        image_small = image_pil

        mask_small = Image.fromarray(mask_dilated)


    # =====================================================
    # Run LaMa
    # =====================================================
    lama = SimpleLama()

    result_small = lama(
        image_small,
        mask_small
    )


    # =====================================================
    # Resize back
    # =====================================================
    if scale < 1.0:

        result_pil = result_small.resize(
            orig_size,
            Image.LANCZOS
        )

    else:

        result_pil = result_small


    return result_pil