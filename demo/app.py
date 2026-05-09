"""Gradio demo — Digital → Analog Clock Converter (ClocksDL)."""

from __future__ import annotations

import datetime
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

# ── Root setup — must happen before local imports ─────────────────────────────
# Modules use relative paths for checkpoints, so cwd must be the project root.
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
os.chdir(str(_root))

import gradio as gr

from digital_clock_notebooks.time_extractor import TimeExtractor
from analog_clock_new.mask_generator import get_mask          # loads MaskUNet on import
from analog_clock_new.inpaint_lama import inpaint_image
from analog_clock_new.visualize_geometry import (
    add_clock_hands,
    draw_hands,
    get_clock_geometry,
)

# ── Lazy TimeExtractor (YOLO loads on first call) ─────────────────────────────
_extractor: TimeExtractor | None = None


def _get_extractor() -> TimeExtractor:
    global _extractor
    if _extractor is None:
        _extractor = TimeExtractor(verbose=False)
    return _extractor


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(img)


def _to_np(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"))


def _ensure_hms(time_str: str) -> str:
    """Pad 'HH:MM' → 'HH:MM:00' so draw_hands always gets seconds."""
    parts = time_str.strip().split(":")
    return time_str if len(parts) >= 3 else time_str + ":00"


# ── Sample images (for the quick-example strip) ───────────────────────────────

_DIG = [p for p in [
    str(_root / "digital_clock_notebooks" / "1.jpg"),
    str(_root / "digital_clock_notebooks" / "2.jpg"),
    str(_root / "digital_clock_notebooks" / "3.jpg"),
    str(_root / "digital_clock_notebooks" / "5.jpg"),
    str(_root / "digital_clock_notebooks" / "6.jpg"),
] if Path(p).exists()]

_ANA = [p for p in [
    str(_root / "analog_clock_new" / "1.jpg"),
    str(_root / "analog_clock_new" / "10.jpg"),
    str(_root / "analog_clock_new" / "analog_watch.jpg"),
    str(_root / "analog_clock_new" / "analog_clock_2.jpg"),
    str(_root / "analog_clock_new" / "analog_watch_3.png"),
] if Path(p).exists()]

_EXAMPLE_PAIRS = [list(pair) for pair in zip(_DIG, _ANA)][: min(len(_DIG), len(_ANA))]


# ── Convert ───────────────────────────────────────────────────────────────────

def convert(digital_img: np.ndarray | None, analog_img: np.ndarray | None):
    if digital_img is None or analog_img is None:
        return None, "⚠️  Please upload both a digital and an analog clock image."

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as df:
        digital_path = df.name
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as af:
        analog_path = af.name

    try:
        _to_pil(digital_img).save(digital_path)
        _to_pil(analog_img).save(analog_path)

        # Module A — read time from digital clock (YOLO-based)
        extractor = _get_extractor()
        time_str = str(extractor.extract_time(digital_path))

        if not time_str or "No time" in time_str:
            return None, "❌  Could not detect time from the digital clock image."

        time_str = _ensure_hms(time_str)
        h, m, s = (int(x) for x in time_str.split(":"))

        # Module B — segment clock hands (MaskUNet)
        mask_np, _ = get_mask(analog_path)

        # Inpaint — erase hands (LaMa)
        clean_pil = inpaint_image(_to_pil(analog_img), mask_np)

        # Redraw hands at the detected time
        result_pil = add_clock_hands(clean_pil, time_str)

        return _to_np(result_pil), f"🕐  Detected time: {h:02d}:{m:02d}:{s:02d}"

    except Exception as exc:
        return None, f"❌  Error: {exc}"
    finally:
        for p in (digital_path, analog_path):
            try:
                os.unlink(p)
            except OSError:
                pass


# ── Live clock ────────────────────────────────────────────────────────────────

def prepare_clean_clock(analog_img: np.ndarray | None) -> np.ndarray | None:
    """Run inpainting once when a new image is uploaded. Returns clean numpy frame."""
    if analog_img is None:
        return None
    pil_img = _to_pil(analog_img)
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as af:
            analog_path = af.name
        pil_img.save(analog_path)
        mask_np, _ = get_mask(analog_path)
        clean_pil = inpaint_image(pil_img, mask_np)
    except Exception:
        clean_pil = pil_img
    finally:
        try:
            os.unlink(analog_path)
        except OSError:
            pass
    return _to_np(clean_pil)


def live_clock(clean_img: np.ndarray | None):
    """Draw hands on the cached clean image — runs every second, no inpainting."""
    if clean_img is None:
        return None, "Upload an analog clock to start."
    now = datetime.datetime.now()
    time_str = now.strftime("%H:%M:%S")
    result_pil = add_clock_hands(_to_pil(clean_img), time_str)
    return _to_np(result_pil), time_str


# ── CSS ───────────────────────────────────────────────────────────────────────

CSS = """
body, .gradio-container {
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
    background: #111318 !important;
    color: #dde1ec !important;
}

/* Header */
.app-header {
    text-align: center;
    padding: 2.6rem 1rem 1.2rem;
}
.app-header h1 {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(120deg, #c4b5fd 0%, #818cf8 40%, #38bdf8 75%, #34d399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.4rem;
    letter-spacing: -1px;
}
.app-header .subtitle { color: #64748b; font-size: 0.92rem; margin: 0; }
.app-header .badge {
    display: inline-block;
    margin-top: 0.75rem;
    padding: 0.2rem 0.85rem;
    background: rgba(99,102,241,0.1);
    border: 1px solid rgba(99,102,241,0.28);
    border-radius: 100px;
    color: #818cf8;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.07em;
    text-transform: uppercase;
}

/* Panels */
.upload-panel {
    background: #1b1f2e !important;
    border: 1px solid #2a2f45 !important;
    border-radius: 16px !important;
    padding: 1.1rem !important;
}
.output-panel {
    background: #151a2a !important;
    border: 1px solid #1e3050 !important;
    border-radius: 16px !important;
    padding: 1.1rem !important;
}

/* Panel badges */
.panel-label {
    font-size: 0.70rem;
    font-weight: 700;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
    padding: 0.22rem 0.75rem;
    border-radius: 100px;
    display: inline-block;
}
.label-digital  { background: rgba(74,222,128,0.08); color: #4ade80; border: 1px solid rgba(74,222,128,0.18); }
.label-analog   { background: rgba(96,165,250,0.08); color: #60a5fa; border: 1px solid rgba(96,165,250,0.18); }
.label-output   { background: rgba(167,139,250,0.08); color: #c4b5fd; border: 1px solid rgba(167,139,250,0.22); }

/* Convert button */
.convert-btn {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 70%, #a855f7 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    color: white !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    padding: 0.85rem 2rem !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.38) !important;
    transition: all 0.18s ease !important;
}
.convert-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(99,102,241,0.52) !important;
}

/* Time display */
.time-display textarea {
    background: #1b1f2e !important;
    border: 1px solid rgba(99,102,241,0.28) !important;
    border-radius: 10px !important;
    color: #a78bfa !important;
    font-family: 'Courier New', monospace !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    text-align: center !important;
}

/* Divider */
.section-divider { border: none; border-top: 1px solid #1e2235; margin: 1rem 0; }

/* Section heading */
.section-heading {
    color: #475569;
    font-size: 0.70rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin: 0.6rem 0 0.4rem;
}

/* Tabs */
.tab-nav button {
    background: transparent !important;
    border-bottom: 2px solid transparent !important;
    color: #475569 !important;
    font-weight: 600 !important;
    padding: 0.65rem 1.4rem !important;
    font-size: 0.88rem !important;
}
.tab-nav button.selected {
    border-bottom-color: #8b5cf6 !important;
    color: #c4b5fd !important;
}

/* Footer */
.app-footer {
    text-align: center;
    padding: 1.2rem 1rem;
    color: #2d3651;
    font-size: 0.74rem;
    letter-spacing: 0.03em;
    border-top: 1px solid #1a1e2e;
    margin-top: 0.5rem;
}
"""

# ── UI ────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="ClocksDL — Digital → Analog Converter") as demo:

    gr.HTML("""
    <div class="app-header">
        <h1>ClocksDL</h1>
        <p class="subtitle">Digital → Analog Clock Converter &nbsp;·&nbsp; Deep Learning Pipeline</p>
        <span class="badge">Module A: YOLO Digit Detector &nbsp;·&nbsp; Module B: MaskUNet &nbsp;·&nbsp; LaMa Inpainting</span>
    </div>
    """)

    with gr.Tab("🔄  Convert"):

        with gr.Row(equal_height=True):

            # Digital input
            with gr.Column(scale=1, elem_classes="upload-panel"):
                gr.HTML("<span class='panel-label label-digital'>📟 Digital Clock</span>")
                digital_input = gr.Image(
                    label="", type="numpy", height=260,
                    show_label=False, sources=["upload"],
                )

            # Analog input
            with gr.Column(scale=1, elem_classes="upload-panel"):
                gr.HTML("<span class='panel-label label-analog'>🕰 Analog Reference</span>")
                analog_input = gr.Image(
                    label="", type="numpy", height=260,
                    show_label=False, sources=["upload"],
                )

            # Output
            with gr.Column(scale=1, elem_classes="output-panel"):
                gr.HTML("<span class='panel-label label-output'>✨ Result</span>")
                output_image = gr.Image(
                    label="", type="numpy", height=260,
                    show_label=False,
                )
                time_label = gr.Textbox(
                    label="Detected Time",
                    interactive=False,
                    placeholder="Detected time will appear here…",
                    elem_classes="time-display",
                )

        with gr.Row():
            convert_btn = gr.Button(
                "▶  Convert", variant="primary",
                scale=1, elem_classes="convert-btn",
            )

        if _EXAMPLE_PAIRS:
            gr.HTML("<hr class='section-divider'>")
            gr.HTML("<div class='section-heading'>📂 &nbsp;Quick Examples — click to load</div>")
            gr.Examples(
                examples=_EXAMPLE_PAIRS,
                inputs=[digital_input, analog_input],
                label=None,
            )

        gr.HTML("<div class='app-footer'>University Computer Vision Project &nbsp;·&nbsp; 2026</div>")

        convert_btn.click(
            fn=convert,
            inputs=[digital_input, analog_input],
            outputs=[output_image, time_label],
        )

    # ── Live Clock tab ────────────────────────────────────────────────────────
    with gr.Tab("🔴  Live Clock"):
        gr.HTML("""
        <div style='padding:1rem 0 0.4rem; color:#94a3b8; font-size:0.92rem;'>
            Pick an analog clock as a style reference.
            Hands are redrawn every second to show <strong>real time</strong>.
        </div>
        """)

        live_clean_state = gr.State(None)

        with gr.Row(equal_height=True):
            with gr.Column(scale=1, elem_classes="upload-panel"):
                gr.HTML("<span class='panel-label label-analog'>🕰 Style Reference</span>")
                live_analog_input = gr.Image(
                    label="", type="numpy", height=260,
                    show_label=False, sources=["upload"],
                )
                live_btn = gr.Button("▶  Start Live Clock", variant="secondary")

            with gr.Column(scale=1, elem_classes="output-panel"):
                gr.HTML("<span class='panel-label label-output'>🔴 Live Output</span>")
                live_output = gr.Image(
                    label="", type="numpy", height=260, show_label=False,
                )
                live_time = gr.Textbox(
                    label="Current Time", interactive=False,
                    elem_classes="time-display",
                )

        # When a new image is uploaded, run inpainting once and cache the clean frame
        live_analog_input.change(
            fn=prepare_clean_clock,
            inputs=[live_analog_input],
            outputs=[live_clean_state],
        )

        # Start button draws hands immediately on the cached clean frame
        live_btn.click(
            fn=live_clock,
            inputs=[live_clean_state],
            outputs=[live_output, live_time],
        )

        # Timer reads only the cached clean frame — no inpainting, runs fast
        timer = gr.Timer(value=1)
        timer.tick(
            fn=live_clock,
            inputs=[live_clean_state],
            outputs=[live_output, live_time],
        )


if __name__ == "__main__":
    demo.launch(
        share=False,
        server_port=7860,
        css=CSS,
        theme=gr.themes.Base(
            primary_hue="violet",
            secondary_hue="blue",
            neutral_hue="slate",
        ),
    )
