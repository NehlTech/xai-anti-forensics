import json
import os

import cv2
import gradio as gr
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from PIL import Image
import torchvision.transforms as T

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO_IMAGES = os.path.join(REPO_ROOT, "demo_images")
MODELS_DIR = os.path.join(REPO_ROOT, "models")


# ---------------------------------------------------------------------------
# Concealer
# ---------------------------------------------------------------------------
class VGGBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.conv3 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.relu  = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.conv3(self.relu(self.conv2(self.relu(self.conv1(x))))))

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.vgg  = VGGBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)
    def forward(self, x):
        feat = self.vgg(x)
        return self.pool(feat), feat

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.relu(self.conv(x))

class DilatedConvGroup(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(channels, channels, 3, padding=r, dilation=r)
            for r in [2, 4, 8, 16]
        ])
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        for conv in self.convs:
            x = self.relu(conv(x))
        return x

class Concealer(nn.Module):
    def __init__(self, base_ch=32):
        super().__init__()
        self.down1    = DownBlock(3, base_ch)
        self.down2    = DownBlock(base_ch, base_ch * 2)
        self.down3    = DownBlock(base_ch * 2, base_ch * 4)
        self.bridge   = DilatedConvGroup(base_ch * 4)
        self.up3      = UpBlock(base_ch * 4, base_ch * 4)
        self.up2      = UpBlock(base_ch * 8, base_ch * 2)
        self.up1      = UpBlock(base_ch * 4, base_ch)
        self.out_conv = nn.Conv2d(base_ch * 2, 3, 3, padding=1)
    def forward(self, x):
        d1, s1 = self.down1(x)
        d2, s2 = self.down2(d1)
        d3, s3 = self.down3(d2)
        b  = self.bridge(d3)
        u3 = torch.cat([self.up3(b),  s3], dim=1)
        u2 = torch.cat([self.up2(u3), s2], dim=1)
        u1 = torch.cat([self.up1(u2), s1], dim=1)
        return torch.tanh(self.out_conv(u1)) * 0.1


class Supervisor(nn.Module):
    """
    Wrapper matching supervisor_v3_checkpoint.pt["supervisor"].
    Keys are prefixed with 'model.*' — smp.Unet is stored under .model.
    """
    def __init__(self):
        super().__init__()
        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None,
        )

    def forward(self, x):
        return self.model(x)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def compute_f1(pred, target, threshold=0.5):
    pred_bin = (pred > threshold).float()
    tp = (pred_bin * target).sum()
    fp = (pred_bin * (1 - target)).sum()
    fn = ((1 - pred_bin) * target).sum()
    prec = tp / (tp + fp + 1e-8)
    rec  = tp / (tp + fn + 1e-8)
    return (2 * prec * rec / (prec + rec + 1e-8)).item()


def to_display(tensor_3hw):
    arr = tensor_3hw.detach().cpu().numpy().transpose(1, 2, 0)
    return (np.clip(arr, 0, 1) * 255).astype(np.uint8)


def mask_to_display(tensor):
    arr = tensor.detach().cpu()
    if arr.ndim == 3:
        arr = arr[0]
    arr8 = (np.clip(arr.numpy(), 0, 1) * 255).astype(np.uint8)
    return np.stack([arr8, arr8, arr8], axis=-1)


def heatmap_to_display(heatmap_hw):
    arr8 = (np.clip(heatmap_hw, 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(arr8, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
print("Loading models…")

concealer = Concealer()
joint_ckpt = torch.load(
    os.path.join(MODELS_DIR, "joint_v11_checkpoint.pt"), map_location="cpu"
)
concealer.load_state_dict(joint_ckpt["concealer"])
concealer.eval()

supervisor = Supervisor()
sup_ckpt = torch.load(
    os.path.join(MODELS_DIR, "supervisor_v3_checkpoint.pt"), map_location="cpu"
)
supervisor.load_state_dict(sup_ckpt["supervisor"])
supervisor.eval()

detector = joblib.load(os.path.join(MODELS_DIR, "spatial_concentration_clf.pkl"))

print("All models loaded.")

# ---------------------------------------------------------------------------
# Manifest → dropdown choices
# ---------------------------------------------------------------------------
with open(os.path.join(DEMO_IMAGES, "manifest.json")) as f:
    MANIFEST = json.load(f)

CHOICES = [f"Image {entry['id'] + 1}" for entry in MANIFEST]


# ---------------------------------------------------------------------------
# Grad-CAM hook + two-pass Attention-Shift attribution
# hooks on supervisor.model.decoder.blocks[-1]
# ---------------------------------------------------------------------------
class GradCAMHook:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        target = model.model.decoder.blocks[-1]
        target.register_forward_hook(self._save_acts)
        target.register_full_backward_hook(self._save_grads)

    def _save_acts(self, module, inp, out):
        self.activations = out

    def _save_grads(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]


gradcam_hook = GradCAMHook(supervisor)


def compute_heatmap(supervisor, gradcam_hook, image, roi_threshold=0.5):
    """Computes a single Grad-CAM heatmap using an ROI-summed target."""
    pred = supervisor(image)
    roi_mask = (pred > roi_threshold).float()
    y_R = (pred * roi_mask).sum()

    supervisor.zero_grad()
    y_R.backward(retain_graph=True)

    weights = gradcam_hook.gradients.mean(dim=[2, 3], keepdim=True)
    cam = (weights * gradcam_hook.activations).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=image.shape[2:], mode='bilinear', align_corners=False)
    cam = cam[0, 0].detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam, pred


# ---------------------------------------------------------------------------
# Suppression — avg_pool2d within top-20% attribution region
# ---------------------------------------------------------------------------
def suppress(x_attacked, attribution):
    threshold = np.percentile(attribution, 80)
    R = torch.from_numpy((attribution >= threshold).astype(np.float32)).unsqueeze(0).unsqueeze(0)
    pooled = F.avg_pool2d(x_attacked, kernel_size=9, stride=1, padding=4)
    return x_attacked * (1 - R) + pooled * R


# ---------------------------------------------------------------------------
# Pipeline — split into Step 1 (baseline) / Step 2 (attack + explanation)
# ---------------------------------------------------------------------------
CLASS_LABELS = {0: "Clean", 1: "Local Attack", 2: "Global Attack"}
_to_tensor = T.ToTensor()


def _step1_tensors(x, m):
    """Step 1: clean detection + baseline heatmap. Returns (section_a_outputs, state)."""
    with torch.no_grad():
        clean_pred = torch.sigmoid(supervisor(x))
    f1_clean = compute_f1(clean_pred, m)

    H_A, _ = compute_heatmap(supervisor, gradcam_hook, x)

    state = {"x": x, "m": m, "H_A": H_A, "clean_pred": clean_pred}

    section_a = (
        to_display(x[0]),
        mask_to_display(m[0]),
        mask_to_display(clean_pred[0]),
        f"Clean F1: **{f1_clean:.4f}**",
        heatmap_to_display(H_A),
    )
    return section_a, state


def step1_demo(choice):
    idx = int(choice.split()[-1]) - 1
    entry = MANIFEST[idx]

    img_np = np.array(Image.open(os.path.join(DEMO_IMAGES, entry["image"])).convert("RGB"))
    mask_t = _to_tensor(Image.open(os.path.join(DEMO_IMAGES, entry["mask"])).convert("L"))

    x = torch.from_numpy(img_np).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    m = (mask_t > 0.5).float().unsqueeze(0)

    section_a, state = _step1_tensors(x, m)
    return section_a + (None,) * 7 + (state, gr.update(interactive=True))


def step1_upload(image, mask):
    resize = T.Resize((512, 512))
    x = _to_tensor(resize(image)).unsqueeze(0)

    if mask is None:
        m = torch.ones(1, 1, 512, 512)
    else:
        mask_t = _to_tensor(resize(mask.convert("L")))
        m = (mask_t > 0.5).float().unsqueeze(0)

    section_a, state = _step1_tensors(x, m)
    return section_a + (None,) * 7 + (state, gr.update(interactive=True))


def _step2_tensors(state):
    """Step 2: Concealer attack + attention-shift + detection + suppression."""
    x, m, H_A, clean_pred = state["x"], state["m"], state["H_A"], state["clean_pred"]
    f1_clean = compute_f1(clean_pred, m)

    with torch.no_grad():
        x_att = torch.clamp(x + concealer(x), 0, 1)
        att_pred = torch.sigmoid(supervisor(x_att))
    f1_att  = compute_f1(att_pred, m)
    f1_drop = (f1_clean - f1_att) / (f1_clean + 1e-8) * 100

    H_B, _ = compute_heatmap(supervisor, gradcam_hook, x_att)
    attr_map = np.abs(H_A - H_B)
    attr_map = (attr_map - attr_map.min()) / (attr_map.max() - attr_map.min() + 1e-8)

    entropy  = float(-np.sum(attr_map * np.log(attr_map + 1e-8)))
    mean_mag = float(attr_map.mean())
    det_class = int(detector.predict(np.array([[entropy, mean_mag]]))[0])
    det_label = f"**Detector: {CLASS_LABELS[det_class]}**"

    if det_class == 0:
        s4_img  = None
        s4_info = "Not applied — classified as clean"
    else:
        with torch.no_grad():
            sup_pred = torch.sigmoid(supervisor(suppress(x_att, attr_map)))
        f1_rec  = compute_f1(sup_pred, m)
        s4_img  = mask_to_display(sup_pred[0])
        s4_info = f"Recovered F1: **{f1_rec:.4f}**"

    return (
        to_display(x_att[0]),
        mask_to_display(att_pred[0]),
        f"Attacked F1: **{f1_att:.4f}** | Drop: **{f1_drop:.1f}%**",
        heatmap_to_display(attr_map),
        det_label,
        s4_img,
        s4_info,
    )


def step2_run(state):
    if state is None:
        raise gr.Error("Run Step 1 first.")
    return _step2_tensors(state)


def reset_section():
    """Clears Section A + Section B outputs, the State, and disables Button 2."""
    return (None,) * 12 + (None, gr.update(interactive=False))


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
def _section_a_b(prefix):
    """Builds a Step 1 / Step 2 block. Must be called inside a `with gr.Tab(...)` context."""
    gr.Markdown("---\n## Step 1: Baseline Detection")
    step1_btn = gr.Button("Step 1 — Show Clean Detection", variant="primary")
    with gr.Row():
        a_orig = gr.Image(label="Original Image", type="numpy")
        a_gt   = gr.Image(label="Ground Truth Mask", type="numpy")
        a_pred = gr.Image(label="Supervisor Prediction (Clean)", type="numpy")
    a_f1   = gr.Markdown()
    a_heat = gr.Image(label="Baseline Heatmap (H_A)", type="numpy")

    gr.Markdown("---\n## Step 2: Attack and Explanation")
    step2_btn = gr.Button(
        "Step 2 — Run Attack and Reveal Attention Shift",
        variant="primary",
        interactive=False,
    )
    with gr.Row():
        b_att  = gr.Image(label="Anti-Forensic Image", type="numpy")
        b_pred = gr.Image(label="Attacked Prediction", type="numpy")
    b_info = gr.Markdown()
    with gr.Row():
        with gr.Column():
            b_heat = gr.Image(label="Attention-Shift Map (|H_A - H_B|)", type="numpy")
        with gr.Column():
            b_det  = gr.Markdown()
    with gr.Row():
        with gr.Column():
            b_supp_img  = gr.Image(label="Recovered Prediction (Suppression)", type="numpy")
        with gr.Column():
            b_supp_info = gr.Markdown()

    state = gr.State(value=None)
    section_a = [a_orig, a_gt, a_pred, a_f1, a_heat]
    section_b = [b_att, b_pred, b_info, b_heat, b_det, b_supp_img, b_supp_info]
    return step1_btn, step2_btn, section_a, section_b, state


with gr.Blocks(title="TRACE — XAI Anti-Forensics Demo") as demo:
    gr.Markdown("# TRACE — XAI Anti-Forensics Demo")
    gr.Markdown(
        "Run TRACE in two steps: first reveal the clean baseline detection, then trigger the "
        "Concealer attack and watch the attention shift, detection, and suppression unfold."
    )

    with gr.Tabs():
        with gr.Tab("Pre-loaded Examples"):
            dropdown = gr.Dropdown(choices=CHOICES, value=CHOICES[0], label="Select Image")
            step1_demo_btn, step2_demo_btn, section_a_demo, section_b_demo, demo_state = _section_a_b("demo")

        with gr.Tab("Upload Your Own"):
            with gr.Row():
                upload_img  = gr.Image(type="pil", label="Upload a forged image")
                upload_mask = gr.Image(type="pil", label="Upload its mask (optional)")
            gr.Markdown(
                "If no mask is uploaded, a placeholder mask will be used and F1 scores will "
                "not be meaningful — focus on the heatmap and detector output."
            )
            step1_up_btn, step2_up_btn, section_a_up, section_b_up, upload_state = _section_a_b("upload")

    step1_demo_btn.click(
        fn=step1_demo,
        inputs=[dropdown],
        outputs=section_a_demo + section_b_demo + [demo_state, step2_demo_btn],
    )
    step2_demo_btn.click(fn=step2_run, inputs=[demo_state], outputs=section_b_demo)
    dropdown.change(
        fn=reset_section,
        inputs=[],
        outputs=section_a_demo + section_b_demo + [demo_state, step2_demo_btn],
    )

    step1_up_btn.click(
        fn=step1_upload,
        inputs=[upload_img, upload_mask],
        outputs=section_a_up + section_b_up + [upload_state, step2_up_btn],
    )
    step2_up_btn.click(fn=step2_run, inputs=[upload_state], outputs=section_b_up)
    upload_img.change(
        fn=reset_section,
        inputs=[],
        outputs=section_a_up + section_b_up + [upload_state, step2_up_btn],
    )
    upload_mask.change(
        fn=reset_section,
        inputs=[],
        outputs=section_a_up + section_b_up + [upload_state, step2_up_btn],
    )

if __name__ == "__main__":
    demo.launch(server_port=7860, share=True)
