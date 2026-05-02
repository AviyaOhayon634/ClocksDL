import os
import sys
import math
import argparse
import time
from datetime import datetime
 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from torchvision import transforms, models
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataset import ClockDataset
from pipeline.analog_reader import DigitalClockClassifier
from pipeline.draw_hand import draw_hands_on_tensor
from pipeline.hand_eraser import ClockEraserV2

device = torch.device(
    'cuda' if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available()
    else 'cpu'
)

_eraser_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
_reader_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

os.makedirs('results', exist_ok=True)

class ClockPipeline:
    """
    Loads both models once, then converts any image pair.
 
    Example:
        pipe   = ClockPipeline()
        result = pipe.run(digital_pil, analog_pil)
        result.save('output.png')
    """
 
    def __init__(self,
                 reader_ckpt='src/checkpoints/digital_reader_best.pth',
                 eraser_ckpt='src/checkpoints/eraser_v2_best.pth'):
        print(f"Loading pipeline on {device} ...")
        self.reader = DigitalClockClassifier().to(device)
        self.reader.load_state_dict(torch.load(reader_ckpt, map_location=device))
        self.reader.eval()
        print("  [OK] Reader  —", reader_ckpt)
 
        self.eraser = ClockEraserV2().to(device)
        self.eraser.load_state_dict(torch.load(eraser_ckpt, map_location=device))
        self.eraser.eval()
        print("  [OK] Eraser  —", eraser_ckpt)
        print("Pipeline ready.\n")
 
    @torch.no_grad()
    def run(self, digital_pil: Image.Image,
            analog_pil: Image.Image,
            verbose: bool = True) -> Image.Image:
        """
        Convert one image pair.
 
        Args:
            digital_pil : PIL Image of the digital clock
            analog_pil  : PIL Image of the analog clock (any time)
            verbose     : if True, prints the detected time
 
        Returns:
            PIL Image — analog clock showing the digital clock's time
        """
        # Stage 1 — read time from digital
        dig_t = _reader_transform(
            digital_pil.convert('RGB')).unsqueeze(0).to(device)
        h_p, m_p, s_p = self.reader.predict_time(dig_t)
        h, m, s = h_p.item(), m_p.item(), s_p.item()
        if verbose:
            print(f"  Detected time: {h:02d}:{m:02d}:{s:02d}")
 
        # Stage 2 — erase hands from analog
        ana_t   = _eraser_transform(
            analog_pil.convert('RGB')).unsqueeze(0).to(device)
        clean_t = self.eraser(ana_t)
 
        # Stage 3 — draw correct hands using geometry
        out_t = draw_hands_on_tensor(
            clean_t.cpu(),
            h_p.cpu(), m_p.cpu(), s_p.cpu())
 
        # Resize to original analog dimensions
        out_np  = out_t[0].permute(1,2,0).clamp(0,1).numpy()
        out_pil = Image.fromarray((out_np*255).astype('uint8'))
        return out_pil.resize(analog_pil.size, Image.LANCZOS)

def run_single(digital_path, analog_path, output_path=f'results/output_{datetime.now()}.png',
                reader_ckpt='src/checkpoints/digital_reader_best.pth',
                eraser_ckpt='src/checkpoints/eraser_v2_best.pth'):
    digital_pil = Image.open(digital_path).convert('RGB')
    analog_pil  = Image.open(analog_path).convert('RGB')
 
    pipe   = ClockPipeline(reader_ckpt, eraser_ckpt)
    result = pipe.run(digital_pil, analog_pil)
    result.save(output_path)
 
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(digital_pil); axes[0].set_title('Digital input')
    axes[1].imshow(analog_pil);  axes[1].set_title('Analog input')
    axes[2].imshow(result);      axes[2].set_title('Output')
    for ax in axes: ax.axis('off')
    plt.suptitle('Clock Conversion Pipeline', fontsize=13)
    plt.tight_layout()
    cmp = output_path.replace('.png', '_comparison.png')
    plt.savefig(cmp, dpi=150, bbox_inches='tight')
    print(f"Saved output  : {output_path}")
    print(f"Saved comparison: {cmp}")
    plt.show()


def run_batch(data_dir='src/clock_dataset', n=6,
            output_path=f'results/batch_results_{datetime.now()}.png',
            reader_ckpt='src/checkpoints/digital_reader_best.pth',
            eraser_ckpt='src/checkpoints/eraser_v2_best.pth'):
    pipe = ClockPipeline(reader_ckpt, eraser_ckpt)
 
    ds     = ClockDataset(data_dir, subset='test',
                          transform=_eraser_transform)
    loader = DataLoader(ds, batch_size=n, shuffle=True, num_workers=0)
    batch  = next(iter(loader))
 
    dig_for_reader = torch.stack([
        _reader_transform(transforms.ToPILImage()(batch['digital_img'][i]))
        for i in range(n)
    ]).to(device)
 
    analog_imgs = batch['analog_img'].to(device)
    true_times  = batch['original_time']
 
    with torch.no_grad():
        h_pred, m_pred, s_pred = pipe.reader.predict_time(dig_for_reader)
        clean_faces = pipe.eraser(analog_imgs)
        outputs     = draw_hands_on_tensor(
            clean_faces.cpu(),
            h_pred.cpu(), m_pred.cpu(), s_pred.cpu())
 
    fig, axes = plt.subplots(4, n, figsize=(3*n, 13))
    row_labels = ['Digital input', 'Analog input', 'Our output', 'Ground truth']
 
    for i in range(n):
        h_t = int(true_times[i][0]); m_t = int(true_times[i][1]); s_t = int(true_times[i][2])
        h_p = int(h_pred[i].item()); m_p = int(m_pred[i].item()); s_p = int(s_pred[i].item())
        correct = (h_t==h_p and m_t==m_p and s_t==s_p)
 
        axes[0,i].imshow(batch['digital_img'][i].permute(1,2,0))
        axes[0,i].set_title(f"read: {h_p:02d}:{m_p:02d}:{s_p:02d}",
                             color='green' if correct else 'red', fontsize=8)
        axes[1,i].imshow(analog_imgs[i].cpu().permute(1,2,0))
        axes[1,i].set_title(f"true: {h_t:02d}:{m_t:02d}:{s_t:02d}", fontsize=8)
        axes[2,i].imshow(outputs[i].permute(1,2,0).clamp(0,1))
        axes[2,i].set_title('output', fontsize=8)
        axes[3,i].imshow(batch['analog_img'][i].permute(1,2,0))
        axes[3,i].set_title('ground truth', fontsize=8)
 
        for r in range(4):
            if i == 0: axes[r,0].set_ylabel(row_labels[r], fontsize=9)
            axes[r,i].axis('off')
 
    plt.suptitle('Complete Pipeline: Digital → Analog', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.show()

def run_animate(analog_path,
                reader_ckpt='src/checkpoints/digital_reader_best.pth',
                eraser_ckpt='src/checkpoints/eraser_v2_best.pth'):
    """
    BONUS: show the provided analog clock displaying the current real time,
    with the second hand moving every second. Press Ctrl+C to stop.
    """
    print("Starting live animation — press Ctrl+C to stop.")
    pipe = ClockPipeline(reader_ckpt, eraser_ckpt)
 
    analog_pil = Image.open(analog_path).convert('RGB')
    ana_t = _eraser_transform(analog_pil).unsqueeze(0).to(device)
 
    # Erase once — face never changes
    with torch.no_grad():
        clean_t = pipe.eraser(ana_t)
 
    plt.ion()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.axis('off')
    init_np  = clean_t[0].permute(1,2,0).cpu().clamp(0,1).numpy()
    im_handle = ax.imshow(Image.fromarray((init_np*255).astype('uint8')))
    fig.tight_layout()
 
    try:
        while True:
            now = datetime.now()
            h, m, s = now.hour, now.minute, now.second
 
            frame_t = draw_hands_on_tensor(
                clean_t.cpu(), 
                torch.tensor([h]), torch.tensor([m]), torch.tensor([s]))
            frame_np  = frame_t[0].permute(1,2,0).clamp(0,1).numpy()
            frame_pil = Image.fromarray(
                (frame_np*255).astype('uint8')
            ).resize(analog_pil.size, Image.LANCZOS)
 
            im_handle.set_data(frame_pil)
            ax.set_title(f"{h:02d}:{m:02d}:{s:02d}", fontsize=22)
            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(1)
 
    except KeyboardInterrupt:
        plt.ioff()
        print("\nStopped.")

# run_evaluation.py
import sys
import os

# Ensure the pipeline module can be found
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipeline.full_pipeline import run_batch

if __name__ == "__main__":
    print("Starting batch evaluation...")
    
    # Run the batch pipeline on 6 random samples from the test set
    run_batch(
        data_dir='src/clock_dataset',  # Points to the directory containing 'test/'
        n=6,                           # Number of images to process and visualize
        output_path='results/test_batch_results.png',
        reader_ckpt='src/checkpoints/digital_reader_best.pth',
        eraser_ckpt='src/checkpoints/eraser_v2_best.pth'
    )