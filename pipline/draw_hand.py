import math 
import numpy as np
import torch
from PIL import Image, ImageDraw

def draw_hands_on_tensor(clean_batch, h_batch, m_batch, s_batch, geom_preds=None):
    """
    Draws pixel-perfect clock hands onto clean clock faces.
    clean_batch : (B, 3, H, W) tensor in [0,1]
    h/m/s_batch : (B,) integer tensors
    geom_preds  : (B, 3) tensor with [cx, cy, radius] normalized (0 to 1). 
    Returns     : (B, 3, H, W) tensor in [0,1]
    """
    B, C, H, W = clean_batch.shape
    outputs = []

    for i in range(B):
        face = clean_batch[i].clone()
        h = int(h_batch[i].item())
        m = int(m_batch[i].item())
        s = int(s_batch[i].item())

        # Convert tensor → PIL, draw, convert back
        np_img = (face.permute(1,2,0).cpu().numpy() * 255).astype('uint8')
        pil    = Image.fromarray(np_img)
        draw   = ImageDraw.Draw(pil)

        if geom_preds is not None:
            p_x, p_y, p_r = geom_preds[i]
            cx = p_x.item() * W
            cy = p_y.item() * H
            radius = p_r.item() * W
        else:
            cx, cy = W / 2, H / 2
            radius = min(W, H) / 2 - max(5, W // 25)

        # Angles: -90 offset so 0 = 12 o'clock, clockwise
        h_angle = math.radians((h % 12) * 30 + m * 0.5 - 90)
        m_angle = math.radians(m * 6   + s * 0.1       - 90)
        s_angle = math.radians(s * 6                   - 90)

        def tip(angle, length):
            return (cx + length * math.cos(angle),
                    cy + length * math.sin(angle))

        # Hour hand — short and thick (scaled to radius)
        hx, hy = tip(h_angle, radius * 0.50)
        draw.line([cx, cy, hx, hy], fill=(30, 30, 30),
                  width=max(3, int(radius * 0.08)))

        # Minute hand — long and medium
        mx, my = tip(m_angle, radius * 0.75)
        draw.line([cx, cy, mx, my], fill=(30, 30, 30),
                  width=max(2, int(radius * 0.05)))

        # Second hand — thin and red
        sx2, sy2 = tip(s_angle, radius * 0.85)
        draw.line([cx, cy, sx2, sy2], fill=(200, 40, 40),
                  width=max(1, int(radius * 0.016)))

        # Center cap
        r_cap = max(3, int(radius * 0.05))
        draw.ellipse([cx-r_cap, cy-r_cap, cx+r_cap, cy+r_cap], fill=(30, 30, 30))

        # Back to tensor
        out_t = torch.from_numpy(
            np.array(pil).astype('float32') / 255.0
        ).permute(2, 0, 1)
        outputs.append(out_t)

    return torch.stack(outputs)