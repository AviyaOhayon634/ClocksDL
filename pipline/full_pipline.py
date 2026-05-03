import os
import sys
import math
import argparse
import time
import io
from datetime import datetime
from urllib.parse import urlparse
from urllib.request import urlopen
 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from torchvision import transforms, models
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataset import TimepieceDataset
from pipeline.analog_reader import DigitalClockClassifier, read_time_from_digital_image
from pipeline.draw_hand import draw_hands_on_tensor
from pipeline.hand_eraser import BasicHandRemover, ClockEraserV2

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


def build_eraser_model(eraser_ckpt):
    ckpt_name = os.path.basename(eraser_ckpt).lower()
    if 'clock_eraser_basic' in ckpt_name:
        return BasicHandRemover().to(device)
    return ClockEraserV2().to(device)


def _load_image(source):
    parsed = urlparse(str(source))
    if parsed.scheme in {'http', 'https'}:
        with urlopen(source) as response:
            image_bytes = response.read()
        return Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return Image.open(source).convert('RGB')


def _preserve_face_details(original_face, cleaned_face, diff_threshold=0.05):
    batch_size, _, height, width = original_face.shape
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=original_face.device),
        torch.linspace(-1.0, 1.0, width, device=original_face.device),
        indexing='ij',
    )
    radial_distance = torch.sqrt(xx.square() + yy.square()).view(1, 1, height, width)

    original_luma = original_face.mean(dim=1, keepdim=True)
    cleaned_luma = cleaned_face.mean(dim=1, keepdim=True)
    diff_map = (cleaned_face - original_face).abs().mean(dim=1, keepdim=True)

    # Old hands tend to be dark in the source and noticeably brighter after cleaning.
    hand_candidate = (
        (cleaned_luma - original_luma > 0.07)
        & (original_luma < 0.72)
        & (diff_map > diff_threshold)
        & (radial_distance <= 0.80)
    ).float()
    blend_mask = F.max_pool2d(hand_candidate, kernel_size=9, stride=1, padding=4)
    blend_mask = blend_mask.expand(batch_size, 1, height, width)
    return cleaned_face * blend_mask + original_face * (1.0 - blend_mask)


def _restore_outer_details(original_face, rendered_face):
    batch_size, _, height, width = original_face.shape
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=original_face.device),
        torch.linspace(-1.0, 1.0, width, device=original_face.device),
        indexing='ij',
    )
    radial_distance = torch.sqrt(xx.square() + yy.square()).view(1, 1, height, width)
    original_luma = original_face.mean(dim=1, keepdim=True)

    detail_mask = (
        (radial_distance >= 0.55)
        & (radial_distance <= 0.92)
        & (original_luma < 0.72)
    ).float().expand(batch_size, 1, height, width)

    restored = torch.minimum(rendered_face, original_face)
    return restored * detail_mask + rendered_face * (1.0 - detail_mask)

class ClockPipeline:
    """
    Loads both models once, then converts any image pair.
 
    Example:
        pipe   = ClockPipeline()
        result = pipe.run(digital_pil, analog_pil)
        result.save('output.png')
    """
 
    def __init__(self,
                 reader_ckpt='src/checkpoints/digital_classifier_best.pth',
                 eraser_ckpt='src/checkpoints/clock_eraser_basic_best.pth'):
        print(f"Loading pipeline on {device} ...")
        self.reader = None
        try:
            reader = DigitalClockClassifier().to(device)
            reader.load_state_dict(torch.load(reader_ckpt, map_location=device))
            reader.eval()
            self.reader = reader
            print("  [OK] Reader  —", reader_ckpt)
        except RuntimeError:
            print("  [WARN] Reader checkpoint mismatch; using OCR-only digital reader")
 
        self.eraser = build_eraser_model(eraser_ckpt)
        self.eraser.load_state_dict(torch.load(eraser_ckpt, map_location=device))
        self.eraser.eval()
        print("  [OK] Eraser  —", eraser_ckpt)
        print("Pipeline ready.\n")

    def _predict_time_from_pil(self, digital_pil: Image.Image) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        detected_time = read_time_from_digital_image(digital_pil)
        if detected_time is not None:
            hour, minute, second = detected_time
            return (
                torch.tensor([hour], device=device),
                torch.tensor([minute], device=device),
                torch.tensor([second], device=device),
                '7-segment OCR',
            )

        if self.reader is None:
            raise RuntimeError('Could not decode the digital clock image and no compatible reader checkpoint is available.')

        dig_t = _reader_transform(
            digital_pil.convert('RGB')).unsqueeze(0).to(device)
        hour_pred, minute_pred, second_pred = self.reader.predict_time(dig_t)
        return hour_pred, minute_pred, second_pred, 'classifier fallback'
 
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
        h_p, m_p, s_p, detection_source = self._predict_time_from_pil(digital_pil)
        h, m, s = h_p.item(), m_p.item(), s_p.item()
        if verbose:
            print(f"  Detected time: {h:02d}:{m:02d}:{s:02d} ({detection_source})")
 
        # Stage 2 — erase hands from analog
        ana_t   = _eraser_transform(
            analog_pil.convert('RGB')).unsqueeze(0).to(device)
        clean_t = self.eraser(ana_t)
        preserved_face = _preserve_face_details(ana_t, clean_t)
 
        # Stage 3 — draw correct hands using geometry
        out_t = draw_hands_on_tensor(
            preserved_face.cpu(),
            h_p.cpu(), m_p.cpu(), s_p.cpu())
        out_t = _restore_outer_details(ana_t.cpu(), out_t)
 
        # Resize to original analog dimensions
        out_np  = out_t[0].permute(1,2,0).clamp(0,1).numpy()
        out_pil = Image.fromarray((out_np*255).astype('uint8'))
        return out_pil.resize(analog_pil.size, Image.LANCZOS)

def run_single(digital_path, analog_path, output_path=f'results/output_{datetime.now()}.png',
                reader_ckpt='src/checkpoints/digital_classifier_best.pth',
                eraser_ckpt='src/checkpoints/clock_eraser_basic_best.pth'):
    digital_pil = _load_image(digital_path)
    analog_pil  = _load_image(analog_path)
 
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


def run_batch(data_dir='datasets', n=6,
            output_path=f'results/batch_results_{datetime.now()}.png',
            reader_ckpt='src/checkpoints/digital_classifier_best.pth',
            eraser_ckpt='src/checkpoints/clock_eraser_basic_best.pth'):
    pipe = ClockPipeline(reader_ckpt, eraser_ckpt)
 
    ds     = TimepieceDataset(
        base_path=data_dir,
        partition='test',
        transforms_pipeline=_eraser_transform,
    )
    loader = DataLoader(ds, batch_size=n, shuffle=True, num_workers=0)
    batch  = next(iter(loader))
 
    dig_for_reader = torch.stack([
        _reader_transform(transforms.ToPILImage()(batch['digital_img'][i]))
        for i in range(n)
    ]).to(device)
 
    analog_imgs = batch['analog_img'].to(device)
    true_times  = batch['original_time']
 
    predicted_times = [None] * n
    fallback_indices = []
    for i in range(n):
        pil_image = transforms.ToPILImage()(batch['digital_img'][i].cpu())
        predicted_times[i] = read_time_from_digital_image(pil_image)
        if predicted_times[i] is None:
            fallback_indices.append(i)

    if fallback_indices:
        if pipe.reader is None:
            raise RuntimeError('Some batch images could not be decoded with OCR and no compatible reader checkpoint is available.')
        with torch.no_grad():
            fallback_tensor = torch.stack([
                _reader_transform(transforms.ToPILImage()(batch['digital_img'][i].cpu()))
                for i in fallback_indices
            ]).to(device)
            h_fb, m_fb, s_fb = pipe.reader.predict_time(fallback_tensor)
        for local_idx, sample_idx in enumerate(fallback_indices):
            predicted_times[sample_idx] = (
                int(h_fb[local_idx].item()),
                int(m_fb[local_idx].item()),
                int(s_fb[local_idx].item()),
            )

    h_pred = torch.tensor([item[0] for item in predicted_times], device=device)
    m_pred = torch.tensor([item[1] for item in predicted_times], device=device)
    s_pred = torch.tensor([item[2] for item in predicted_times], device=device)

    with torch.no_grad():
        clean_faces = pipe.eraser(analog_imgs)
        preserved_faces = _preserve_face_details(analog_imgs, clean_faces)
        outputs     = draw_hands_on_tensor(
            preserved_faces.cpu(),
            h_pred.cpu(), m_pred.cpu(), s_pred.cpu())
        outputs = _restore_outer_details(analog_imgs.cpu(), outputs)
 
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
                reader_ckpt='src/checkpoints/digital_classifier_best.pth',
                eraser_ckpt='src/checkpoints/clock_eraser_basic_best.pth'):
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
        clean_t = _preserve_face_details(ana_t, clean_t)
 
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
            frame_t = _restore_outer_details(ana_t.cpu(), frame_t)
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

if __name__ == "__main__":
    print("Starting batch evaluation...")
    
    # Run the batch pipeline on 6 random samples from the test set
    run_batch(
        data_dir='datasets',           # Points to the directory containing train/test
        n=6,                           # Number of images to process and visualize
        output_path='results/test_batch_results.png',
        reader_ckpt='src/checkpoints/digital_classifier_best.pth',
        eraser_ckpt='src/checkpoints/clock_eraser_basic_best.pth'
    )