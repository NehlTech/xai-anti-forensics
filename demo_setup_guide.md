# TRACE Local Demo Setup — Antigravity + Claude Code

Complete step-by-step guide to running the TRACE demo locally,
independent of Colab. CPU inference only (no GPU needed for a demo
running a handful of images).

---

## Step 1: Download your trained checkpoints from Google Drive

Open Google Drive and navigate to your `xai-anti-forensics_data/models/`
folder. Download these three files and keep them together:

| File | What it is |
|---|---|
| `joint_v11_checkpoint.pt` | Trained Concealer (epoch 29) |
| `supervisor_v3_checkpoint.pt` | Trained Supervisor (ResNet-34 U-Net) |
| `spatial_concentration_clf.pkl` | Trained Detector classifier |

Save all three somewhere easy to find on your laptop, e.g. your Downloads
folder, you will move them in Step 3.

---

## Step 2: Clone the repo into Antigravity

Open Antigravity. Use File → Open Folder, then open a terminal and run:

```bash
git clone https://github.com/NehlTech/xai-anti-forensics.git
cd xai-anti-forensics
```

---

## Step 3: Set up the folder structure

Inside the cloned repo, create a `models/` folder and move your
three downloaded checkpoint files into it:

```
xai-anti-forensics/
├── models/
│   ├── joint_v11_checkpoint.pt
│   ├── supervisor_v3_checkpoint.pt
│   └── spatial_concentration_clf.pkl
├── results/
│   └── figures/
├── 00_dataset_prep.ipynb
├── ... (other notebooks)
└── demo/               ← Claude Code will create this
```

---

## Step 4: Download a small set of test images from Drive

You need a handful of real test images with their ground-truth masks for
the demo. Open your Colab notebook and run this cell once to export them:

```python
# ─── Run this once in Colab to export demo images ─────────────────────────
import os, shutil, json
from PIL import Image
import torch

OUTPUT_DIR = f"{REPO_DIR}/demo_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Pick 5 representative indices — run beforehand to confirm these are good
DEMO_INDICES = [4, 17, 32, 51, 88]

manifest_out = []
for i, idx in enumerate(DEMO_INDICES):
    img, mask = test_ds[idx]

    img_path = f"{OUTPUT_DIR}/img_{i:02d}.png"
    mask_path = f"{OUTPUT_DIR}/mask_{i:02d}.png"

    from torchvision.utils import save_image
    save_image(img, img_path)
    save_image(mask, mask_path)

    manifest_out.append({
        "id": i,
        "image": f"img_{i:02d}.png",
        "mask": f"mask_{i:02d}.png",
        "original_index": idx
    })

with open(f"{OUTPUT_DIR}/manifest.json", "w") as f:
    json.dump(manifest_out, f, indent=2)

print(f"Exported {len(DEMO_INDICES)} images to {OUTPUT_DIR}")
print("Download this folder from Drive to your local repo's demo_images/ folder")
```

Then download the resulting `demo_images/` folder from Drive into your
local repo root so the structure looks like:

```
xai-anti-forensics/
├── models/
├── demo_images/
│   ├── img_00.png
│   ├── mask_00.png
│   ├── img_01.png
│   ├── mask_01.png
│   ... (5 image + 5 mask pairs)
│   └── manifest.json
└── demo/
```

---

## Step 5: Install dependencies locally

In the Antigravity terminal:

```bash
pip install torch torchvision segmentation-models-pytorch \
            gradio scikit-learn joblib pillow numpy opencv-python
```

If you are on a Mac with Apple Silicon, PyTorch will run on MPS (Metal)
automatically, which is meaningfully faster than CPU. No GPU is required.

---

## Step 6: Give Claude Code this exact prompt

Open Claude Code in Antigravity (Ctrl+Shift+P → Claude Code, or via the
Spark panel), then paste this prompt:

---

```
I have a trained PyTorch image forensics system called TRACE.
The repo is already cloned and open. Here is exactly what exists:

FOLDER STRUCTURE:
- models/joint_v11_checkpoint.pt         — trained Concealer weights
- models/supervisor_v3_checkpoint.pt     — trained Supervisor weights
- models/spatial_concentration_clf.pkl   — trained Detector classifier
- demo_images/                           — 5 PNG images + 5 PNG masks + manifest.json

ARCHITECTURE SUMMARY:

1. Concealer — U-Net encoder-decoder with:
   - 3 downsampling VGGBlock + MaxPool stages (32, 64, 128 channels)
   - Dilated conv bridge (rates 2, 4, 8, 16)
   - 3 upsampling stages with skip connections
   - Output: tanh() * 0.1 (bounded perturbation)
   - Load: checkpoint["concealer"] key from joint_v11_checkpoint.pt

2. Supervisor — segmentation_models_pytorch Unet:
   - encoder_name="resnet34", encoder_weights=None (load from checkpoint)
   - in_channels=3, classes=1, activation=None
   - Apply sigmoid externally after forward pass
   - Load: checkpoint["supervisor"] key from supervisor_v3_checkpoint.pt

3. Detector — sklearn LogisticRegression:
   - Input: [entropy, mean_magnitude] of Grad-CAM attribution map
   - Classes: 0=clean, 1=local_attack, 2=global_attack
   - Load: joblib.load("models/spatial_concentration_clf.pkl")

4. GradCAM — hooks on supervisor.model.decoder.blocks[-1]:
   - Computes gradients of adversarial loss: -BCE(supervisor(x'), mask)
   - Returns (H, W) attribution map normalised to [0,1]

5. Suppression — avg_pool2d applied within thresholded attribution region:
   - Threshold: top 20% of attribution map by magnitude
   - kernel_size=9, stride=1, padding=4
   - Recombine: x_suppressed = x_attacked*(1-R) + pooled*R

Please build a clean Gradio demo app in a new file called demo/app.py that:

1. Loads all three models at startup (CPU is fine, use map_location="cpu")
2. Reads demo_images/manifest.json to build a dropdown of 5 test images
   labelled "Image 1", "Image 2", etc.
3. On clicking "Run TRACE Pipeline", runs the full pipeline:
   - Stage 1: clean prediction — show original image, ground truth mask,
     Supervisor's predicted mask, and F1 score
   - Stage 2: Concealer attack — show anti-forensic image (visually
     near-identical), Supervisor's attacked prediction, F1 drop %
   - Stage 3: Attribution — show Grad-CAM heatmap as a jet colormap image,
     show Detector's classification as a large label
   - Stage 4: Suppression — if Detector says attack, apply suppression
     and show the recovered prediction; if clean, show "Not applied"
4. Layout all outputs clearly with gr.Row() and gr.Column() groupings,
   with a bold section heading before each stage
5. Use share=False (local only) and launch on port 7860

For image display, convert tensors to (H,W,3) uint8 numpy arrays.
For masks, display as greyscale by stacking to 3 channels.
For the heatmap, use cv2.applyColorMap with COLORMAP_JET then convert
BGR to RGB.

Use this F1 function:
def compute_f1(pred, target, threshold=0.5):
    pred_bin = (pred > threshold).float()
    tp = (pred_bin * target).sum()
    fp = (pred_bin * (1-target)).sum()
    fn = ((1-pred_bin) * target).sum()
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    return (2 * prec * rec / (prec + rec + 1e-8)).item()

Also create demo/requirements.txt listing all required packages.
Also create a one-line demo/run.sh: `python demo/app.py`

Make the app look clean and professional — this is going in front of a
university supervisor to demonstrate a completed MPhil research system.
```

---

## Step 7: Run the demo

Once Claude Code has built `demo/app.py`, in the Antigravity terminal:

```bash
cd xai-anti-forensics
bash demo/run.sh
```

Then open `http://localhost:7860` in your browser. That is what your
supervisor will see.

---

## What to check before the meeting

Run through the demo once with all five images beforehand and confirm:

- [ ] All five load without errors
- [ ] At least two show a meaningful F1 drop after the attack (>30%)
- [ ] At least one triggers the Detector's attack classification
- [ ] The heatmap is visible and non-trivial (not all black or all red)
- [ ] Suppression runs without crashing on the flagged images

If any index gives a trivially bad result (F1 already near zero on the
clean image), swap it out by re-running the Colab export cell with a
different index.

---

## Fallback if something breaks on the day

If the local setup has a last-minute problem, your Colab Gradio cell
from the earlier session is the direct fallback. The `share=True` link
it generates works from any browser without any local install. Keep
that Colab notebook open and ready as backup.
