"""Model architectures, checkpoint loaders, inference, and Grad-CAM helpers."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

try:
    import cv2
except ImportError:
    cv2 = None

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    train_test_split = None  # type: ignore[assignment]

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import models, transforms

    TORCH_IMPORT_ERROR: str | None = None
except ImportError as exc:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    models = None  # type: ignore[assignment]
    transforms = None  # type: ignore[assignment]
    TORCH_IMPORT_ERROR = str(exc)

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
SEED = 42

MODEL_CHECKPOINT_FILES = {
    "VGG16": "checkpoint_vgg16.pth",
    "EfficientNetB0": "checkpoint_efficientnetb0.pth",
    "MobileNetV2": "checkpoint_mobilenetv2.pth",
    "ResNet50": "checkpoint_resnet50.pth",
    "Custom_CNN": "checkpoint_custom_cnn.pth",
}


if torch is not None and nn is not None and models is not None:

    class CustomCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.model_name = "Custom_CNN"
            self.block1 = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout2d(0.20),
            )
            self.block2 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout2d(0.25),
            )
            self.block3 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout2d(0.30),
            )
            self.block4 = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout2d(0.35),
            )
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256, 128, bias=False),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.50),
                nn.Linear(128, 1),
            )

        def get_gradcam_target_layer(self):
            return self.block4[2]

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            x = self.block4(x)
            x = self.gap(x)
            return self.head(x).squeeze(1)

    class ImageNetBackboneWrapper(nn.Module):
        """Wraps a torchvision ImageNet backbone with standardized ImageNet normalization."""

        def __init__(self, model_name: str, backbone: nn.Module, gradcam_layer: nn.Module):
            super().__init__()
            self.model_name = model_name
            self.backbone = backbone
            self._gradcam_layer_ref = [gradcam_layer]
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        def get_gradcam_target_layer(self):
            return self._gradcam_layer_ref[0]

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = (x - self.mean) / self.std
            return self.backbone(x).squeeze(1)

    def build_vgg16() -> ImageNetBackboneWrapper:
        base_vgg = models.vgg16(weights=None)
        base_vgg.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        base_vgg.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.50),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),
            nn.Linear(64, 1),
        )
        return ImageNetBackboneWrapper("VGG16", base_vgg, base_vgg.features[29])

    def build_resnet50() -> ImageNetBackboneWrapper:
        base_resnet = models.resnet50(weights=None)
        in_feat = base_resnet.fc.in_features
        base_resnet.fc = nn.Sequential(
            nn.Linear(in_feat, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.50),
            nn.Linear(256, 1),
        )
        return ImageNetBackboneWrapper("ResNet50", base_resnet, base_resnet.layer4[-1])

    def build_mobilenet_v2() -> ImageNetBackboneWrapper:
        base_mobilenet = models.mobilenet_v2(weights=None)
        in_feat = base_mobilenet.last_channel
        base_mobilenet.classifier = nn.Sequential(
            nn.Linear(in_feat, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(0.40),
            nn.Linear(128, 1),
        )
        return ImageNetBackboneWrapper("MobileNetV2", base_mobilenet, base_mobilenet.features[-1])

    def build_efficientnet_b0() -> ImageNetBackboneWrapper:
        base_efficientnet = models.efficientnet_b0(weights=None)
        in_feat = base_efficientnet.classifier[1].in_features
        base_efficientnet.classifier = nn.Sequential(
            nn.BatchNorm1d(in_feat),
            nn.Dropout(0.40),
            nn.Linear(in_feat, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),
            nn.Linear(128, 1),
        )
        return ImageNetBackboneWrapper("EfficientNetB0", base_efficientnet, base_efficientnet.features[-1])

    MODEL_BUILDERS: dict[str, Any] = {
        "Custom_CNN": CustomCNN,
        "VGG16": build_vgg16,
        "ResNet50": build_resnet50,
        "MobileNetV2": build_mobilenet_v2,
        "EfficientNetB0": build_efficientnet_b0,
    }
else:
    MODEL_BUILDERS = {}


def get_device() -> torch.device:
    if torch is None:
        raise RuntimeError(f"PyTorch is not installed in the active Python environment ({TORCH_IMPORT_ERROR}).")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_from_disk(
    project_dir: Path,
    model_name: str | None = None,
    device: torch.device | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Load either the serialized best model bundle or an individual architecture checkpoint."""
    if torch is None:
        raise RuntimeError(f"PyTorch is not installed in the active Python environment ({TORCH_IMPORT_ERROR}).")
    device = device or get_device()
    models_dir = project_dir / "models"
    best_path = models_dir / "best_pneumonia_model.pth"

    metadata: dict[str, Any] = {
        "decision_thresholds": {
            "default": 0.50,
            "optimal_f1": 0.50,
            "clinical_screening": 0.35,
        }
    }

    if best_path.exists():
        best_bundle = torch.load(best_path, map_location=device, weights_only=False)
        best_name = best_bundle.get("model_name", "VGG16")
        if "decision_thresholds" in best_bundle:
            metadata["decision_thresholds"] = {
                k: float(v) for k, v in best_bundle["decision_thresholds"].items()
            }
        if model_name is None or model_name == best_name:
            model = MODEL_BUILDERS[best_name]().to(device)
            model.load_state_dict(best_bundle["model_state_dict"])
            model.eval()
            metadata["model_name"] = best_name
            metadata["checkpoint_path"] = str(best_path)
            return model, metadata

    target_name = model_name or "VGG16"
    ckpt_file = models_dir / MODEL_CHECKPOINT_FILES[target_name]
    raw = torch.load(ckpt_file, map_location=device, weights_only=False)
    state_dict = raw["model_state_dict"] if isinstance(raw, dict) and "model_state_dict" in raw else raw
    model = MODEL_BUILDERS[target_name]().to(device)
    model.load_state_dict(state_dict)
    model.eval()
    metadata["model_name"] = target_name
    metadata["checkpoint_path"] = str(ckpt_file)
    return model, metadata


def preprocess_uploaded_image(pil_image: Image.Image) -> tuple[np.ndarray, Any]:
    """Convert any PIL image into resized 224x224 RGB uint8 display array and (1, 3, 224, 224) float tensor."""
    pil_rgb = pil_image.convert("RGB")
    if cv2 is not None:
        rgb_resized = cv2.resize(np.array(pil_rgb), (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_LINEAR)
    else:
        rgb_resized = np.array(pil_rgb.resize((IMG_WIDTH, IMG_HEIGHT), Image.Resampling.BILINEAR), dtype=np.uint8)
    tensor_1chw = transforms.ToTensor()(Image.fromarray(rgb_resized)).unsqueeze(0) if transforms is not None else None
    return rgb_resized, tensor_1chw


def predict_with_gradcam(
    model: nn.Module,
    img_tensor_1chw: torch.Tensor,
    device: torch.device | None = None,
) -> tuple[float, np.ndarray, float]:
    """Run forward inference + Grad-CAM saliency computation safely for all 5 PyTorch models."""
    if torch is None:
        raise RuntimeError(f"PyTorch is not installed in the active Python environment ({TORCH_IMPORT_ERROR}).")
    device = device or next(model.parameters()).device
    model.eval()
    target_layer = model.get_gradcam_target_layer()
    activations: dict[str, torch.Tensor] = {}
    gradients: dict[str, torch.Tensor] = {}
    tensor_hooks = []

    toggled_inplace = []
    for m in model.modules():
        if getattr(m, "inplace", False):
            m.inplace = False
            toggled_inplace.append(m)

    def forward_hook(_module, _inp, out):
        activations["value"] = out.detach()
        if out.requires_grad:
            tensor_hooks.append(out.register_hook(lambda grad: gradients.__setitem__("value", grad.detach())))

    h_fwd = target_layer.register_forward_hook(forward_hook)

    try:
        x = img_tensor_1chw.to(device).float().detach().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.amp.autocast(device_type=device.type, enabled=False):
            with torch.enable_grad():
                logit = model(x).view(-1)[0]
                logit.backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        acts = activations["value"]
        grads = gradients["value"]
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(IMG_HEIGHT, IMG_WIDTH), mode="bilinear", align_corners=False)
        cam_np = cam[0, 0].detach().cpu().numpy()
        cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)
        prob = float(torch.sigmoid(logit.detach()).cpu().item())
        return prob, cam_np, latency_ms
    finally:
        h_fwd.remove()
        for h in tensor_hooks:
            h.remove()
        for m in toggled_inplace:
            m.inplace = True


def predict_single_probability(
    model: nn.Module,
    img_tensor_1chw: torch.Tensor,
    device: torch.device | None = None,
) -> tuple[float, float]:
    """Fast forward-only inference returning (probability, latency_ms)."""
    if torch is None:
        raise RuntimeError(f"PyTorch is not installed in the active Python environment ({TORCH_IMPORT_ERROR}).")
    device = device or next(model.parameters()).device
    model.eval()
    with torch.inference_mode():
        x = img_tensor_1chw.to(device).float()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        logit = model(x).view(-1)[0]
        if device.type == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000.0
        prob = float(torch.sigmoid(logit).cpu().item())
    return prob, latency_ms


def overlay_gradcam(original_img_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.42) -> tuple[np.ndarray, np.ndarray]:
    """Blend normalized Grad-CAM heatmap onto resized radiograph using JET colormap."""
    heatmap_norm = np.clip(heatmap, 0.0, 1.0)
    if cv2 is not None:
        heatmap_uint8 = np.uint8(255 * heatmap_norm)
        heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    else:
        r = np.clip(1.5 - np.abs(4.0 * heatmap_norm - 3.0), 0.0, 1.0)
        g = np.clip(1.5 - np.abs(4.0 * heatmap_norm - 2.0), 0.0, 1.0)
        b = np.clip(1.5 - np.abs(4.0 * heatmap_norm - 1.0), 0.0, 1.0)
        heatmap_rgb = np.uint8(255.0 * np.stack([r, g, b], axis=-1))
    overlay_rgb = np.uint8(np.clip(original_img_rgb * (1.0 - alpha) + heatmap_rgb * alpha, 0, 255))
    return heatmap_rgb, overlay_rgb


def load_benchmark_artifacts(project_dir: Path) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Load comparison CSV, evaluation JSON, and training histories JSON."""
    results_dir = project_dir / "results"
    df_comp = pd.read_csv(results_dir / "model_comparison.csv")
    with open(results_dir / "evaluation_report.json", "r", encoding="utf-8") as f:
        eval_report = json.load(f)
    with open(results_dir / "training_history.json", "r", encoding="utf-8") as f:
        train_hist = json.load(f)
    return df_comp, eval_report, train_hist


def discover_holdout_sample_cases(project_dir: Path, max_per_group: int = 3) -> list[dict[str, str]]:
    """Reproduce exact SEED=42 untouched holdout split so built-in demo cases never overlap with training data."""
    samples: list[dict[str, str]] = []
    train_csv = project_dir / "dataset" / "stage2_train_metadata.csv"
    train_img_dir = project_dir / "dataset" / "Training" / "Images"

    if train_csv.exists() and train_img_dir.exists() and train_test_split is not None:
        df_raw = pd.read_csv(train_csv)
        agg = {"Target": "max"}
        if "class" in df_raw.columns:
            agg["class"] = "first"
        df_patients = df_raw.groupby("patientId", as_index=False).agg(agg)
        df_patients["label"] = df_patients["Target"].astype(int).clip(0, 1)
        df_cohort, _ = train_test_split(
            df_patients, train_size=4000, stratify=df_patients["label"], random_state=SEED
        )
        _, test_df = train_test_split(
            df_cohort.reset_index(drop=True),
            test_size=0.15,
            random_state=SEED,
            stratify=df_cohort["label"],
        )
        test_df = test_df.reset_index(drop=True)

        pneu_rows = test_df[test_df["label"] == 1].head(max_per_group)
        norm_rows = test_df[test_df["label"] == 0].head(max_per_group)

        for idx, (_, r) in enumerate(pneu_rows.iterrows(), 1):
            p = train_img_dir / f"{r['patientId']}.png"
            if p.exists():
                samples.append({
                    "title": f"Holdout Pneumonia Case #{idx} ({r['patientId'][:8]}...)",
                    "ground_truth": "Pneumonia (Lung Opacity)",
                    "path": str(p),
                })

        for idx, (_, r) in enumerate(norm_rows.iterrows(), 1):
            p = train_img_dir / f"{r['patientId']}.png"
            if p.exists():
                cls_label = r.get("class", "Normal")
                samples.append({
                    "title": f"Holdout Non-Pneumonia Case #{idx} [{cls_label}] ({r['patientId'][:8]}...)",
                    "ground_truth": f"Negative ({cls_label})",
                    "path": str(p),
                })

    test_unlabeled_dir = project_dir / "dataset" / "Test"
    if test_unlabeled_dir.exists():
        unlabeled = sorted(test_unlabeled_dir.glob("*.png"))[:2]
        for idx, p in enumerate(unlabeled, 1):
            samples.append({
                "title": f"RSNA Challenge Unlabeled Test #{idx} ({p.stem[:8]}...)",
                "ground_truth": "Unlabeled Test Radiograph",
                "path": str(p),
            })

    return samples

