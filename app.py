"""PulmoScan AI — Streamlit Clinical Research UI for Pneumonia CXR Screening & 5-Model Comparison."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.inference import (
    MODEL_BUILDERS,
    MODEL_CHECKPOINT_FILES,
    TORCH_IMPORT_ERROR,
    discover_holdout_sample_cases,
    get_device,
    load_benchmark_artifacts,
    load_model_from_disk,
    overlay_gradcam,
    predict_single_probability,
    predict_with_gradcam,
    preprocess_uploaded_image,
)

PROJECT_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="PulmoScan AI | Pneumonia CXR Screening & Model Benchmark",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2.5rem;
        max-width: 1320px;
    }
    .hero-banner {
        background: linear-gradient(120deg, #0F4C81 0%, #176B9E 58%, #0E7490 100%);
        color: #FFFFFF;
        padding: 1.35rem 1.75rem;
        border-radius: 14px;
        margin-bottom: 1.0rem;
        box-shadow: 0 8px 24px rgba(15, 76, 129, 0.14);
    }
    .hero-title {
        font-size: 1.65rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin: 0 0 0.25rem 0;
    }
    .hero-sub {
        font-size: 0.96rem;
        color: #E0F2FE;
        margin: 0;
        line-height: 1.45;
    }
    .pill-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 0.85rem;
    }
    .pill {
        background: rgba(255, 255, 255, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.28);
        color: #FFFFFF;
        padding: 0.25rem 0.7rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .disclaimer-strip {
        background: #FFFBEB;
        border: 1px solid #FDE68A;
        border-left: 5px solid #D97706;
        color: #78350F;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        font-size: 0.86rem;
        margin-bottom: 1.1rem;
    }
    .diag-card-pos {
        background: #FEF2F2;
        border: 1px solid #FECACA;
        border-left: 6px solid #DC2626;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        margin: 0.75rem 0 1.0rem 0;
    }
    .diag-card-neg {
        background: #ECFDF5;
        border: 1px solid #A7F3D0;
        border-left: 6px solid #059669;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        margin: 0.75rem 0 1.0rem 0;
    }
    .diag-card-borderline {
        background: #FFFBEB;
        border: 1px solid #FDE68A;
        border-left: 6px solid #D97706;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        margin: 0.75rem 0 1.0rem 0;
    }
    .diag-heading {
        font-size: 1.28rem;
        font-weight: 700;
        margin: 0 0 0.3rem 0;
    }
    .action-panel {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-top: 4px solid #DC2626;
        border-radius: 12px;
        padding: 1.15rem 1.3rem;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
        height: 100%;
    }
    .precaution-panel {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-top: 4px solid #0F4C81;
        border-radius: 12px;
        padding: 1.15rem 1.3rem;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
        height: 100%;
    }
    .guidance-neg-panel {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-top: 4px solid #059669;
        border-radius: 12px;
        padding: 1.15rem 1.3rem;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
    }
    .panel-title {
        font-size: 1.06rem;
        font-weight: 700;
        color: #0F172A;
        margin: 0 0 0.65rem 0;
    }
    .step-item {
        margin-bottom: 0.65rem;
        line-height: 1.48;
        font-size: 0.92rem;
        color: #1E293B;
    }
    .step-item strong {
        color: #0F172A;
    }
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 0.95rem 1.05rem;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.03);
    }
    .kpi-label {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #64748B;
        margin-bottom: 0.25rem;
    }
    .kpi-val {
        font-size: 1.45rem;
        font-weight: 700;
        color: #0F172A;
        line-height: 1.2;
    }
    .kpi-sub {
        font-size: 0.8rem;
        color: #475569;
        margin-top: 0.2rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading PyTorch model weights...")
def get_cached_model(model_name: str):
    device = get_device()
    return load_model_from_disk(PROJECT_DIR, model_name=model_name, device=device)


def checkpoint_exists_for_model(model_name: str, champion_name: str = "VGG16") -> bool:
    if TORCH_IMPORT_ERROR is not None:
        return False
    models_dir = PROJECT_DIR / "models"
    if model_name == champion_name and (models_dir / "best_pneumonia_model.pth").exists():
        return True
    ckpt_name = MODEL_CHECKPOINT_FILES.get(model_name)
    return bool(ckpt_name and (models_dir / ckpt_name).exists())


def load_model_safe(model_name: str, champion_name: str = "VGG16"):
    if TORCH_IMPORT_ERROR is not None:
        return (
            None,
            None,
            f"PyTorch is not installed in the active interpreter (`{sys.executable}`). Run `conda activate pneumonia_gpu` then `streamlit run app.py`.",
        )
    if not checkpoint_exists_for_model(model_name, champion_name=champion_name):
        expected = MODEL_CHECKPOINT_FILES.get(model_name, "best_pneumonia_model.pth")
        return None, None, f"Missing model weights (`models/{expected}`)."
    try:
        m_obj, m_meta = get_cached_model(model_name)
        return m_obj, m_meta, None
    except Exception as exc:
        return None, None, f"Failed to load `{model_name}` checkpoint: {exc}"


@st.cache_data(show_spinner=False)
def get_cached_benchmarks():
    return load_benchmark_artifacts(PROJECT_DIR)


@st.cache_data(show_spinner=False)
def get_cached_samples():
    return discover_holdout_sample_cases(PROJECT_DIR)


df_comparison, eval_report, training_histories = get_cached_benchmarks()
champion_name = eval_report.get("selected_best_model", "VGG16")
saved_thresholds = eval_report.get(
    "decision_thresholds", {"default": 0.50, "optimal_f1": 0.50, "clinical_screening": 0.35}
)
device = get_device() if TORCH_IMPORT_ERROR is None else None
device_name = device.type.upper() if device is not None else "CPU"
gpu_label = eval_report.get("device", "CPU") if (device is not None and device.type == "cuda") else "CPU"
model_order = df_comparison.sort_values("Clinical_Composite_Score", ascending=False)["Model Name"].tolist()
available_models = [m for m in model_order if checkpoint_exists_for_model(m, champion_name=champion_name)]

# ---------------------------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Inference Configuration")
    model_labels = {}
    for m in model_order:
        base_lbl = f"{m} (Best Champion)" if m == champion_name else m
        if TORCH_IMPORT_ERROR is not None:
            model_labels[m] = f"{base_lbl} — PyTorch missing"
        else:
            model_labels[m] = base_lbl if m in available_models else f"{base_lbl} — weights missing"

    default_idx = model_order.index(champion_name) if champion_name in model_order else 0
    if available_models and champion_name not in available_models:
        default_idx = model_order.index(available_models[0])

    selected_model_name = st.selectbox(
        "Active Model Architecture",
        options=model_order,
        index=default_idx,
        format_func=lambda x: model_labels.get(x, x),
        help="Defaults to VGG16, selected as the top overall architecture by Clinical Composite Score (0.6709), ROC-AUC (0.8328), and PR-AUC (0.6398).",
    )

    op_mode = st.radio(
        "Clinical Decision Operating Point",
        options=[
            f"Balanced / Optimal F1 (tau = {saved_thresholds['optimal_f1']:.2f})",
            f"High-Sensitivity Triage (tau = {saved_thresholds['clinical_screening']:.2f})",
            "Custom Decision Threshold",
        ],
        index=0,
        help="Balanced maximizes F1; High-Sensitivity lowers missed pneumonia cases (increases test Recall to 82.96% on VGG16).",
    )
    if "Optimal F1" in op_mode:
        active_threshold = float(round(saved_thresholds["optimal_f1"], 2))
    elif "High-Sensitivity" in op_mode:
        active_threshold = float(round(saved_thresholds["clinical_screening"], 2))
    else:
        active_threshold = float(
            st.slider("Custom Threshold (tau)", min_value=0.05, max_value=0.95, value=0.50, step=0.05)
        )

    cam_alpha = st.slider(
        "Grad-CAM Saliency Overlay Opacity",
        min_value=0.15,
        max_value=0.75,
        value=0.42,
        step=0.03,
    )

    model, model_meta, model_load_error = load_model_safe(selected_model_name, champion_name=champion_name)

    st.divider()
    st.markdown("### Selected Model Summary")
    sel_row = df_comparison[df_comparison["Model Name"] == selected_model_name].iloc[0]
    st.metric("Clinical Composite", f"{sel_row['Clinical_Composite_Score']:.4f}")
    col_sb1, col_sb2 = st.columns(2)
    col_sb1.metric("ROC-AUC", f"{sel_row['ROC-AUC']:.3f}")
    col_sb2.metric("PR-AUC", f"{sel_row['PR-AUC']:.3f}")
    col_sb1.metric("Sensitivity", f"{sel_row['Recall (Sensitivity)']:.1%}")
    col_sb2.metric("Specificity", f"{sel_row['Specificity']:.1%}")
    st.caption(
        f"Hardware: `{device_name}` ({gpu_label})  \n"
        f"Params: `{sel_row['Total Parameters'] / 1e6:.2f}M` | Latency: `{sel_row['Inference Latency (ms)']:.2f} ms`"
    )
    if model_load_error:
        st.warning(f"{model_load_error} Tab 2 & Tab 3 benchmarks remain available.")

# ---------------------------------------------------------------------------
# Hero Banner & Disclaimer
# ---------------------------------------------------------------------------
active_pill_status = (
    f"Active Model: {selected_model_name} (tau = {active_threshold:.2f})"
    if model is not None
    else f"Active Model: {selected_model_name} (Benchmark Mode)"
)
st.markdown(
    f"""
    <div class="hero-banner">
        <div class="hero-title">PulmoScan AI — Pneumonia Chest Radiograph Screening & Model Comparison Studio</div>
        <p class="hero-sub">
            Automated binary chest X-ray screening powered by your trained PyTorch CUDA architectures, live Grad-CAM visual attribution, clinical triage action protocols, and 5-model holdout test benchmarking.
        </p>
        <div class="pill-row">
            <span class="pill">Champion Model: {champion_name} (Composite 0.6709 | AUC 0.8328)</span>
            <span class="pill">{active_pill_status}</span>
            <span class="pill">Cohort: {eval_report['cohort_size']:,} Stratified RSNA Patients</span>
            <span class="pill">Compute: {device_name} — {gpu_label}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Primary Navigation Tabs
# ---------------------------------------------------------------------------
tab_screen, tab_compare, tab_audit = st.tabs(
    [
        "🩺 Radiograph Screening & Clinical Actions",
        "📊 5-Model Performance Comparison",
        "🔬 Champion Model Deep Dive (Thresholds, Calibration & Stress Test)",
    ]
)


def render_probability_gauge(prob: float, threshold: float) -> go.Figure:
    bar_color = "#DC2626" if prob >= threshold else ("#D97706" if prob >= max(0.0, threshold - 0.10) else "#059669")
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob * 100.0,
            number={"suffix": "%", "font": {"size": 30, "color": "#0F172A"}},
            title={"text": "Pneumonia Risk Probability", "font": {"size": 14, "color": "#475569"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#64748B"},
                "bar": {"color": bar_color, "thickness": 0.62},
                "bgcolor": "#F8FAFC",
                "borderwidth": 1,
                "bordercolor": "#CBD5E1",
                "steps": [
                    {"range": [0, threshold * 100.0], "color": "#ECFDF5"},
                    {"range": [threshold * 100.0, 75.0 if threshold < 0.75 else 100.0], "color": "#FEF3C7"},
                    {"range": [min(75.0, threshold * 100.0), 100.0], "color": "#FEE2E2"},
                ],
                "threshold": {
                    "line": {"color": "#0F172A", "width": 3},
                    "thickness": 0.85,
                    "value": threshold * 100.0,
                },
            },
        )
    )
    fig.update_layout(height=230, margin=dict(l=20, r=20, t=42, b=10), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def render_positive_guidance(prob: float, threshold: float):
    severity = "High Radiological Suspicion" if prob >= 0.75 else "Moderate / Positive Screening Alert"
    st.markdown(
        f"""
        <div class="diag-card-pos">
            <div class="diag-heading" style="color:#991B1B;">PNEUMONIA DETECTED — POSITIVE SCREEN ({severity})</div>
            <div style="color:#7F1D1D; font-size:0.95rem;">
                Predicted probability <strong>{prob:.1%}</strong> meets or exceeds the active decision threshold (<strong>&tau; = {threshold:.2f}</strong>). Prompt medical evaluation and confirmatory clinical correlation are advised.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_act, col_prec = st.columns(2, gap="medium")
    with col_act:
        st.markdown(
            """
            <div class="action-panel">
                <div class="panel-title">State of Actions to Be Taken Immediately</div>
                <div class="step-item"><strong>1. Urgent Physician / Emergency Review:</strong> Share this chest radiograph and current symptoms with a physician, pulmonologist, or urgent care clinic <strong>today</strong> for clinical examination (lung auscultation) and formal radiological interpretation.</div>
                <div class="step-item"><strong>2. Immediate Vital Signs &amp; Oxygen Check:</strong> Measure pulse oximetry (<strong>SpO2</strong>), respiratory rate, body temperature, heart rate, and blood pressure. <strong>Seek emergency ER care immediately</strong> if SpO2 &lt; 92% (or &lt; 94% with respiratory distress), breathing rate exceeds 24–30 breaths/min, or confusion occurs.</div>
                <div class="step-item"><strong>3. Confirmatory Diagnostic Workup:</strong> Complete clinician-recommended tests such as Complete Blood Count (CBC), CRP / procalcitonin, sputum or blood cultures, viral respiratory panel, and a lateral view chest X-ray or thoracic CT if warranted.</div>
                <div class="step-item"><strong>4. Avoid Unsupervised Self-Medication:</strong> Do <strong>not</strong> initiate leftover antibiotics or strong cough suppressants before medical evaluation—appropriate therapy depends on identifying bacterial vs. viral vs. atypical etiology.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_prec:
        st.markdown(
            """
            <div class="precaution-panel">
                <div class="panel-title">Possible Precautions &amp; Protective Care</div>
                <div class="step-item"><strong>1. Respiratory Isolation &amp; Masking:</strong> Wear a well-fitting medical mask around vulnerable individuals (infants, adults &ge; 65, pregnant people, or immunocompromised patients), sanitize hands frequently, and follow strict cough/sneeze hygiene.</div>
                <div class="step-item"><strong>2. Zero Airway Irritants:</strong> Strictly avoid cigarette smoke, vaping aerosols, wood/biomass smoke, dust, and chemical vapors to prevent worsening alveolar inflammation.</div>
                <div class="step-item"><strong>3. Hydration, Positioning &amp; Rest:</strong> Maintain steady oral fluid intake to thin airway secretions, rest in an elevated semi-upright posture to reduce work of breathing, and log temperature trends every 4–6 hours.</div>
                <div class="step-item"><strong>4. Red-Flag Emergency Escalation Triggers:</strong> Go to the Emergency Department immediately for worsening shortness of breath at rest, pleuritic chest pain, bluish lips or fingernails (cyanosis), persistent fever &gt; 39°C (102.2°F), inability to keep fluids down, or sudden lethargy/dizziness.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_negative_guidance(prob: float, threshold: float):
    is_borderline = prob >= max(0.0, threshold - 0.10)
    if is_borderline:
        st.markdown(
            f"""
            <div class="diag-card-borderline">
                <div class="diag-heading" style="color:#92400E;">BORDERLINE NEGATIVE — BELOW ACTIVE THRESHOLD (&tau; = {threshold:.2f})</div>
                <div style="color:#78350F; font-size:0.95rem;">
                    Predicted probability <strong>{prob:.1%}</strong> is below <strong>&tau; = {threshold:.2f}</strong>, but falls in the borderline band (&ge; {max(0.0, threshold - 0.10):.0%}). Consider switching to the <strong>High-Sensitivity Screening threshold (&tau; = 0.35)</strong> or obtaining physician review if clinical symptoms are present.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="diag-card-neg">
                <div class="diag-heading" style="color:#065F46;">NO PNEUMONIA DETECTED — NEGATIVE SCREEN</div>
                <div style="color:#064E3B; font-size:0.95rem;">
                    Predicted pneumonia probability <strong>{prob:.1%}</strong> is below the active screening threshold (<strong>&tau; = {threshold:.2f}</strong>). Model found low likelihood of airspace consolidation.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="guidance-neg-panel">
            <div class="panel-title">Clinical Correlation &amp; Preventive Precautions</div>
            <div class="step-item"><strong>• Symptoms Always Override AI Output:</strong> Subtle early interstitial infiltrates or retrocardiac opacities behind the cardiac silhouette can still yield a low screening score. If fever, persistent productive cough, chest tightness, or dyspnea continues, seek physician evaluation regardless of this score.</div>
            <div class="step-item"><strong>• Routine Respiratory Hygiene:</strong> Rest adequately, stay hydrated, avoid tobacco/air pollutants, and monitor symptoms over the next 24–48 hours.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# TAB 1: Screening & Clinical Actions
# ---------------------------------------------------------------------------
with tab_screen:
    if TORCH_IMPORT_ERROR is not None:
        st.error(
            f"**Wrong Python Environment Active (`{sys.executable}`):** PyTorch is not installed here (`{TORCH_IMPORT_ERROR}`), so live radiograph inference cannot run.  \n"
            "Stop the current server (`Ctrl+C`) and restart using your **`pneumonia_gpu`** conda environment:  \n"
            "`conda activate pneumonia_gpu` then `streamlit run app.py` (or `C:\\Users\\Yashwant\\anaconda3\\envs\\pneumonia_gpu\\python.exe -m streamlit run app.py`).  \n"
            "Meanwhile, **Tab 2 (`5-Model Performance Comparison`)** and **Tab 3 (`Champion Model Deep Dive`)** remain fully functional."
        )
    elif model is None:
        available_hint = (
            f"Available local checkpoints in sidebar: **{', '.join(available_models)}**."
            if available_models
            else "No `.pth` checkpoint files were found in `models/` (large binary weights are excluded from Git by `.gitignore`)."
        )
        st.warning(
            f"**Live Inference Disabled for `{selected_model_name}`:** {model_load_error}  \n"
            f"{available_hint}  \n"
            "All evaluation benchmarks, learning curves, confusion matrices, and stress tests remain fully interactive in "
            "**Tab 2 (`5-Model Performance Comparison`)** and **Tab 3 (`Champion Model Deep Dive`)**. "
            "To enable live CXR screening and Grad-CAM, train/export checkpoints via `Pneumonia_Detection_Deep_Learning.ipynb` or host `models/*.pth` via Git LFS / cloud storage."
        )

    sample_cases = get_cached_samples()
    source_options = ["Upload Chest X-Ray Image"]
    if sample_cases:
        source_options.append("Select Sample Holdout / Test Radiograph")

    col_in1, col_in2 = st.columns([1.2, 2.0], gap="medium")
    with col_in1:
        input_mode = st.radio("Radiograph Input Source", options=source_options, horizontal=False)

    pil_input: Image.Image | None = None
    image_label = ""
    ground_truth_note = None

    with col_in2:
        if input_mode == "Upload Chest X-Ray Image":
            uploaded = st.file_uploader(
                "Upload Frontal Chest X-Ray (AP/PA — PNG, JPG, JPEG, BMP, TIFF)",
                type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
            )
            if uploaded is not None:
                pil_input = Image.open(uploaded)
                image_label = uploaded.name
        else:
            choice_idx = st.selectbox(
                "Choose an Untouched Holdout or RSNA Test Radiograph",
                options=list(range(len(sample_cases))),
                format_func=lambda idx: sample_cases[idx]["title"],
            )
            chosen = sample_cases[choice_idx]
            pil_input = Image.open(chosen["path"])
            image_label = chosen["title"]
            ground_truth_note = chosen["ground_truth"]

    if pil_input is None:
        if model is not None:
            st.info(
                "Upload a frontal chest radiograph above or switch to **Select Sample Holdout / Test Radiograph** to run live detection, Grad-CAM localization, and clinical triage guidance."
            )
    elif model is None:
        rgb_224, _ = preprocess_uploaded_image(pil_input)
        st.image(rgb_224, caption=f"Uploaded Preview: {image_label} (224x224)", width=340)
        st.info(
            "Radiograph preprocessed cleanly, but live pneumonia prediction and Grad-CAM cannot run without model weights in `models/`. Switch to **Tab 2** or **Tab 3** to inspect saved benchmark results."
        )
    else:
        rgb_224, tensor_1chw = preprocess_uploaded_image(pil_input)
        prob, heatmap, latency_ms = predict_with_gradcam(model, tensor_1chw, device=device)
        heatmap_rgb, overlay_rgb = overlay_gradcam(rgb_224, heatmap, alpha=cam_alpha)
        is_positive = prob >= active_threshold

        # Summary KPI row + Risk Gauge
        c_kpi, c_gauge = st.columns([1.55, 1.0], gap="medium")
        with c_kpi:
            k1, k2 = st.columns(2)
            k1.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Screening Verdict</div>
                    <div class="kpi-val" style="color:{'#DC2626' if is_positive else '#059669'};">
                        {'PNEUMONIA' if is_positive else 'NORMAL / NEG'}
                    </div>
                    <div class="kpi-sub">Active Model: {selected_model_name}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            k2.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">P(Pneumonia) Score</div>
                    <div class="kpi-val">{prob:.2%}</div>
                    <div class="kpi-sub">Threshold &tau; = {active_threshold:.2f} (Margin: {prob - active_threshold:+.2%})</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("<div style='height:0.65rem;'></div>", unsafe_allow_html=True)
            k3, k4 = st.columns(2)
            k3.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Inference + Grad-CAM</div>
                    <div class="kpi-val">{latency_ms:.1f} ms</div>
                    <div class="kpi-sub">Device: {device.type.upper()}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            k4.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Radiograph Source</div>
                    <div class="kpi-val" style="font-size:1.05rem;">{image_label[:26]}{'...' if len(image_label) > 26 else ''}</div>
                    <div class="kpi-sub">{'Dataset Label: ' + ground_truth_note if ground_truth_note else 'User Uploaded CXR'}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c_gauge:
            st.plotly_chart(render_probability_gauge(prob, active_threshold), width="stretch")

        # Diagnostic verdict + Immediate Actions / Precautions (if YES)
        if is_positive:
            render_positive_guidance(prob, active_threshold)
        else:
            render_negative_guidance(prob, active_threshold)

        # 3-Panel Visual Inspection + Grad-CAM
        st.markdown("#### Radiographic Visual Attribution (Grad-CAM Explainability)")
        img_col1, img_col2, img_col3 = st.columns(3, gap="medium")
        with img_col1:
            st.image(rgb_224, caption="1. Standardized CXR Input (224x224)", width="stretch")
        with img_col2:
            st.image(heatmap_rgb, caption=f"2. Grad-CAM Saliency Map ({selected_model_name})", width="stretch")
        with img_col3:
            st.image(overlay_rgb, caption="3. Alpha-Blended Lung Focus Overlay", width="stretch")
        st.caption(
            "Grad-CAM highlights spatial receptive-field activations supporting the Pneumonia logit (warm red/yellow regions indicate stronger positive attribution). Verify attention lies within pulmonary parenchyma rather than border labels or ECG leads."
        )

        # Live Cross-Model Consensus Comparison on the SAME Radiograph
        with st.expander("Run All 5 Trained Models Live on This Radiograph (Cross-Model Consensus)", expanded=False):
            run_consensus = st.button("Compare All Available Checkpoints on Current Image", type="primary")
            if run_consensus:
                consensus_rows = []
                skipped_models = []
                for m_name in model_order:
                    m_obj, _, _err = load_model_safe(m_name, champion_name=champion_name)
                    if m_obj is None:
                        skipped_models.append(m_name)
                        continue
                    m_prob, m_lat = predict_single_probability(m_obj, tensor_1chw, device=device)
                    m_bench = df_comparison[df_comparison["Model Name"] == m_name].iloc[0]
                    consensus_rows.append(
                        {
                            "Model Name": m_name,
                            "Predicted P(Pneumonia)": m_prob,
                            "Prediction": "Pneumonia (1)" if m_prob >= active_threshold else "Normal / Negative (0)",
                            "Test ROC-AUC": float(m_bench["ROC-AUC"]),
                            "Test Composite": float(m_bench["Clinical_Composite_Score"]),
                            "Pass Latency (ms)": m_lat,
                        }
                    )

                if skipped_models:
                    st.warning(f"Skipped models with missing checkpoints: {', '.join(skipped_models)}")

                if not consensus_rows:
                    st.error("No trained checkpoints are available in `models/` to run consensus inference.")
                else:
                    df_live = pd.DataFrame(consensus_rows)
                    pos_votes = int((df_live["Predicted P(Pneumonia)"] >= active_threshold).sum())
                    st.info(
                        f"**Cross-Architecture Consensus at tau = {active_threshold:.2f}:** "
                        f"**{pos_votes} / {len(df_live)}** available architectures flag **Pneumonia** on this radiograph."
                    )
                    fig_live = px.bar(
                        df_live,
                        x="Model Name",
                        y="Predicted P(Pneumonia)",
                        color="Prediction",
                        color_discrete_map={"Pneumonia (1)": "#DC2626", "Normal / Negative (0)": "#059669"},
                        text=df_live["Predicted P(Pneumonia)"].map(lambda v: f"{v:.1%}"),
                        title="Live Predicted Pneumonia Probability Across Available Models",
                    )
                    fig_live.add_hline(y=active_threshold, line_dash="dash", line_color="#0F172A", annotation_text=f"tau={active_threshold:.2f}")
                    fig_live.update_layout(yaxis_range=[0, 1.05], height=340, margin=dict(t=45, b=20))
                    st.plotly_chart(fig_live, width="stretch")
                    st.dataframe(
                        df_live.style.format(
                            {
                                "Predicted P(Pneumonia)": "{:.4f}",
                                "Test ROC-AUC": "{:.4f}",
                                "Test Composite": "{:.4f}",
                                "Pass Latency (ms)": "{:.2f}",
                            }
                        ),
                        width="stretch",
                        hide_index=True,
                    )


# ---------------------------------------------------------------------------
# TAB 2: 5-Model Performance & Comparison
# ---------------------------------------------------------------------------
with tab_compare:
    st.markdown("### Holdout Test Benchmark Across All 5 Trained Architectures")
    st.caption(
        f"Evaluated on the exact same untouched Holdout Test cohort (N = {eval_report['split_sizes']['test']} patients, "
        f"Training = {eval_report['split_sizes']['train']}, Validation = {eval_report['split_sizes']['val']}, zero patient leakage) at default tau = 0.50."
    )

    best_comp = df_comparison.sort_values("Clinical_Composite_Score", ascending=False).iloc[0]
    best_auc = df_comparison.sort_values("ROC-AUC", ascending=False).iloc[0]
    best_prauc = df_comparison.sort_values("PR-AUC", ascending=False).iloc[0]
    best_sens = df_comparison.sort_values("Recall (Sensitivity)", ascending=False).iloc[0]
    fastest = df_comparison.sort_values("Inference Latency (ms)", ascending=True).iloc[0]

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Top Composite Model", best_comp["Model Name"], f"{best_comp['Clinical_Composite_Score']:.4f}")
    m2.metric("Highest ROC-AUC", best_auc["Model Name"], f"{best_auc['ROC-AUC']:.4f}")
    m3.metric("Highest PR-AUC", best_prauc["Model Name"], f"{best_prauc['PR-AUC']:.4f}")
    m4.metric("Top Recall (@0.50)", best_sens["Model Name"], f"{best_sens['Recall (Sensitivity)']:.1%}")
    m5.metric("Fastest Inference", fastest["Model Name"], f"{fastest['Inference Latency (ms)']:.2f} ms/img")

    st.markdown("#### Master Comparison Table (Sorted by Clinical Composite Score)")
    df_sorted = df_comparison.sort_values("Clinical_Composite_Score", ascending=False).reset_index(drop=True).copy()
    df_sorted["Params (M)"] = df_sorted["Total Parameters"] / 1e6
    show_cols = [
        "Model Name",
        "Clinical_Composite_Score",
        "ROC-AUC",
        "PR-AUC",
        "Recall (Sensitivity)",
        "Specificity",
        "Precision",
        "F1 Score",
        "Accuracy",
        "Balanced Accuracy",
        "MCC",
        "Training Time (s)",
        "Params (M)",
        "Inference Latency (ms)",
    ]
    st.dataframe(
        df_sorted[show_cols].style.format(
            {
                "Clinical_Composite_Score": "{:.4f}",
                "ROC-AUC": "{:.4f}",
                "PR-AUC": "{:.4f}",
                "Recall (Sensitivity)": "{:.4f}",
                "Specificity": "{:.4f}",
                "Precision": "{:.4f}",
                "F1 Score": "{:.4f}",
                "Accuracy": "{:.4f}",
                "Balanced Accuracy": "{:.4f}",
                "MCC": "{:.4f}",
                "Training Time (s)": "{:.1f}",
                "Params (M)": "{:.2f}",
                "Inference Latency (ms)": "{:.2f}",
            }
        ).highlight_max(
            subset=["Clinical_Composite_Score", "ROC-AUC", "PR-AUC", "Recall (Sensitivity)", "Specificity", "F1 Score", "Accuracy", "MCC"],
            color="#DBEAFE",
        ),
        width="stretch",
        hide_index=True,
    )

    c_chart1, c_chart2 = st.columns(2, gap="medium")
    with c_chart1:
        df_melt = df_sorted.melt(
            id_vars="Model Name",
            value_vars=["Clinical_Composite_Score", "ROC-AUC", "PR-AUC", "F1 Score", "Balanced Accuracy"],
            var_name="Metric",
            value_name="Score",
        )
        fig_bar = px.bar(
            df_melt,
            x="Model Name",
            y="Score",
            color="Metric",
            barmode="group",
            title="Core Diagnostic Performance Metrics by Architecture",
            color_discrete_sequence=["#0F4C81", "#0EA5E9", "#EC4899", "#10B981", "#8B5CF6"],
        )
        fig_bar.update_layout(yaxis_range=[0.3, 0.95], height=400, legend=dict(orientation="h", y=-0.22))
        st.plotly_chart(fig_bar, width="stretch")

    with c_chart2:
        fig_eff = px.scatter(
            df_sorted,
            x="Params (M)",
            y="Inference Latency (ms)",
            size="ROC-AUC",
            color="Model Name",
            text="Model Name",
            title="Efficiency Frontier: Parameter Footprint vs GPU Latency (Bubble = ROC-AUC)",
            size_max=34,
        )
        fig_eff.update_traces(textposition="top center")
        fig_eff.update_layout(height=400, xaxis_range=[-1.5, 27.5], yaxis_range=[0.5, 8.6], showlegend=False)
        st.plotly_chart(fig_eff, width="stretch")

    st.markdown("#### Training & Validation Learning Trajectories (Stage 1 & Stage 2 Fine-Tuning)")
    hist_metric = st.selectbox(
        "Select Training Trajectory Metric",
        options=["loss", "accuracy", "auc"],
        format_func=lambda k: {"loss": "Weighted BCE Loss", "accuracy": "Classification Accuracy", "auc": "ROC-AUC"}[k],
    )
    fig_hist = make_subplots(rows=1, cols=2, subplot_titles=(f"Training {hist_metric.upper()}", f"Validation {hist_metric.upper()}"))
    palette = {
        "VGG16": "#EC4899",
        "EfficientNetB0": "#0F4C81",
        "MobileNetV2": "#F97316",
        "ResNet50": "#10B981",
        "Custom_CNN": "#6366F1",
    }
    for m_name in model_order:
        h = training_histories[m_name]
        epochs = list(range(1, len(h[hist_metric]) + 1))
        val_key = f"val_{hist_metric}"
        col_hex = palette.get(m_name, "#334155")
        fig_hist.add_trace(
            go.Scatter(x=epochs, y=h[hist_metric], mode="lines+markers", name=m_name, line=dict(color=col_hex, width=2)),
            row=1,
            col=1,
        )
        fig_hist.add_trace(
            go.Scatter(
                x=epochs,
                y=h[val_key],
                mode="lines+markers",
                name=m_name,
                line=dict(color=col_hex, width=2, dash="dash"),
                showlegend=False,
            ),
            row=1,
            col=2,
        )
    fig_hist.update_layout(height=390, margin=dict(t=45, b=20))
    st.plotly_chart(fig_hist, width="stretch")


# ---------------------------------------------------------------------------
# TAB 3: Selected Champion Deep Dive (Thresholds, Calibration, Stress Test)
# ---------------------------------------------------------------------------
with tab_audit:
    st.markdown(f"### Champion Architecture Audit: `{champion_name}`")
    champ_bench = [m for m in eval_report["model_benchmarks"] if m["Model Name"] == champion_name][0]

    col_cm, col_th = st.columns([1.0, 1.25], gap="medium")
    with col_cm:
        # Exact counts from test_size=600 (135 Pneumonia, 465 Normal) at tau=0.50: TN=335, FP=130, FN=35, TP=100
        tp = int(round(champ_bench["Recall (Sensitivity)"] * 135))
        fn = 135 - tp
        tn = int(round(champ_bench["Specificity"] * 465))
        fp = 465 - tn
        z_cm = [[tn, fp], [fn, tp]]
        txt_cm = [
            [f"TN: {tn}<br>({tn/465:.1%})", f"FP: {fp}<br>({fp/465:.1%})"],
            [f"FN: {fn}<br>({fn/135:.1%})", f"TP: {tp}<br>({tp/135:.1%})"],
        ]
        fig_cm = go.Figure(
            data=go.Heatmap(
                z=z_cm,
                x=["Predicted Normal", "Predicted Pneumonia"],
                y=["True Normal", "True Pneumonia"],
                text=txt_cm,
                texttemplate="%{text}",
                colorscale="Blues",
                showscale=False,
            )
        )
        fig_cm.update_layout(
            title=f"{champion_name} Test Confusion Matrix (N=600, tau=0.50)",
            height=360,
            yaxis_autorange="reversed",
            margin=dict(t=50, b=20),
        )
        st.plotly_chart(fig_cm, width="stretch")

    with col_th:
        st.markdown("#### Operating Thresholds & Calibration Profile")
        df_op = pd.DataFrame(
            [
                {"Operating Point": "Default / Optimal F1 (tau = 0.50)", "Test Sensitivity": 0.7407, "Test Specificity": 0.7204, "Test Precision": 0.4348, "Test F1": 0.5479},
                {"Operating Point": "Clinical High-Sensitivity Screening (tau = 0.35)", "Test Sensitivity": 0.8296, "Test Specificity": 0.6000, "Test Precision": 0.3758, "Test F1": 0.5173},
            ]
        )
        st.dataframe(
            df_op.style.format({"Test Sensitivity": "{:.4f}", "Test Specificity": "{:.4f}", "Test Precision": "{:.4f}", "Test F1": "{:.4f}"}),
            width="stretch",
            hide_index=True,
        )
        cal = eval_report["calibration_metrics"]
        c_cal1, c_cal2 = st.columns(2)
        c_cal1.metric("Brier Score Loss", f"{cal['brier_score']:.4f}", "Lower is better calibrated")
        c_cal2.metric("Expected Calibration Error (ECE)", f"{cal['expected_calibration_error']:.2%}", "Class-weighted positive shift")
        st.markdown(
            """
            **Clinical Interpretation**:
            - Training with `pos_weight = 3.4374` shifts raw probabilities upward to counteract `77.5% vs 22.5%` imbalance, making threshold selection explicit and clinically controllable.
            - Lowering `tau` from `0.50` to the validation-tuned **`0.35`** screening point lifts Holdout Test **Sensitivity from `74.07%` to `82.96%`**, reducing missed pneumonia False Negatives at triage.
            """
        )

    st.markdown("#### Artifact Perturbation & Covariate Shift Stress Testing")
    df_rob = pd.DataFrame(eval_report["robustness_audit"])
    fig_rob = px.bar(
        df_rob,
        x="Stress Condition",
        y=["ROC-AUC", "F1 Score", "Sensitivity"],
        barmode="group",
        title=f"{champion_name} Resilience Under Clinical Acquisition Artifacts",
        color_discrete_sequence=["#0F4C81", "#10B981", "#F97316"],
    )
    fig_rob.update_layout(yaxis_range=[0.4, 0.95], height=370, legend=dict(orientation="h", y=-0.22))
    st.plotly_chart(fig_rob, width="stretch")
    st.dataframe(
        df_rob.style.format(
            {
                "ROC-AUC": "{:.4f}",
                "Delta AUC": "{:+.4f}",
                "F1 Score": "{:.4f}",
                "Delta F1": "{:+.4f}",
                "Sensitivity": "{:.4f}",
                "Delta Sensitivity": "{:+.4f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )

