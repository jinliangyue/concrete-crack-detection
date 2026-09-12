"""
Streamlit demo for the concrete crack detection pipeline.

Four sections:
  1. Try it — upload an image, get a prediction with confidence
  2. Where is the model looking? — Grad-CAM heatmap overlay on the upload
  3. Model comparison — accuracy ladder across RF / CNN / ResNet18
  4. Training curves — ResNet18 train vs validation accuracy per epoch

Run locally:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.compare import comparison_table, headline_improvement  # noqa: E402
from src.data import PROJECT_ROOT as _PROJECT_ROOT  # noqa: E402
from src.infer import predict  # noqa: E402
from src.xai import gradcam_heatmap, overlay_heatmap  # noqa: E402


st.set_page_config(
    page_title="Concrete Crack Detection · Transfer Learning Demo",
    page_icon="🔍",
    layout="wide",
)


@st.cache_data
def load_resnet18_results() -> dict:
    with open(_PROJECT_ROOT / "results" / "resnet18_results.json") as f:
        return json.load(f)


@st.cache_resource
def load_resnet18_model():
    """Load ResNet18 checkpoint once and cache. Avoids re-loading on every upload."""
    import torch
    from src.infer import _load_torch_model
    from src.infer import CHECKPOINTS

    ckpt_path = CHECKPOINTS["resnet18"]
    if not ckpt_path.exists():
        return None, None
    model, meta = _load_torch_model("resnet18", ckpt_path)
    return model, meta


def render_upload_panel() -> None:
    st.subheader("Try it")
    st.markdown(
        "Upload any concrete surface photo (JPG or PNG). The model returns the predicted class "
        "and the probability for each class."
    )

    uploaded = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded is None:
        st.info("Upload an image above to run inference.")
        return

    img = Image.open(uploaded).convert("RGB")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(img, caption="Uploaded image", use_container_width=True)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        img.save(f.name, format="JPEG")
        tmp_path = f.name

    model, meta = load_resnet18_model()
    if model is None:
        st.error(
            "ResNet18 checkpoint not found at models/crack_resnet18_best.pt. "
            "Train it first: `python -m src.train --model resnet18`"
        )
        return

    with st.spinner("Classifying..."):
        try:
            result = predict(tmp_path, model_name="resnet18")
        except Exception as exc:  # noqa: BLE001 — surface any inference error
            st.error(f"Inference failed: {exc}")
            return

    with col2:
        label_color = "🔴" if result["label"] == "Crack" else "🟢"
        st.markdown(f"### {label_color} Prediction: **{result['label']}**")
        st.metric(
            label="Confidence",
            value=f"{result['confidence'] * 100:.2f}%",
            delta=f"checkpoint acc {result['checkpoint_accuracy'] * 100:.2f}%" if result.get("checkpoint_accuracy") else None,
        )

        probs_df = pd.DataFrame({
            "Class": list(result["probs"].keys()),
            "Probability": [v * 100 for v in result["probs"].values()],
        })
        fig = go.Figure(go.Bar(
            x=probs_df["Class"],
            y=probs_df["Probability"],
            marker_color=["#10b981" if c == "NoCrack" else "#ef4444" for c in probs_df["Class"]],
            text=[f"{p:.2f}%" for p in probs_df["Probability"]],
            textposition="outside",
            width=0.5,
        ))
        fig.update_layout(
            yaxis_title="Probability (%)",
            yaxis_range=[0, 105],
            height=280,
            margin=dict(l=10, r=10, t=10, b=30),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            f"Model: ResNet18 (ImageNet transfer) · "
            f"Input size: {result['img_size']}×{result['img_size']} · "
            f"Device: MPS / CUDA / CPU auto"
        )

    # Grad-CAM explanation
    _render_gradcam(img, model, result)


def _render_gradcam(
    img: Image.Image,
    model: torch.nn.Module,
    result: dict,
) -> None:
    """Render Grad-CAM heatmap for the uploaded image, alongside the model output."""
    st.markdown("---")
    st.markdown("**Where is the model looking?** — Grad-CAM on `layer4`")

    device = next(model.parameters()).device
    img_size = int(result["img_size"])
    img_resized = img.resize((img_size, img_size))
    arr = (np.asarray(img_resized, dtype=np.float32).transpose(2, 0, 1) / 255.0)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    tensor = torch.from_numpy((arr - mean) / std).unsqueeze(0).float().to(device)

    target_idx = 1 if result["label"] == "Crack" else 0

    with st.spinner("Computing Grad-CAM..."):
        try:
            heatmap = gradcam_heatmap(
                model, tensor, target_class=target_idx, target_layer="layer4",
            )
            overlay = overlay_heatmap(img, heatmap)
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Grad-CAM unavailable for this image: {exc}")
            return

    # Render heatmap as RGB for st.image (no clamp=True needed)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as cm
    heat_rgb = (cm.jet(np.clip(heatmap, 0.0, 1.0))[:, :, :3] * 255).astype(np.uint8)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(img, caption="Original", use_container_width=True)
    with col2:
        st.image(heat_rgb, caption=f"Heatmap (target={result['label']})", use_container_width=True)
    with col3:
        st.image(overlay, caption="Overlay (α=0.45)", use_container_width=True)

    st.caption(
        f"Target class: **{result['label']}** · Layer: `layer4` (last residual block) · "
        f"Red regions = strongest positive contribution. Confidence: "
        f"{result['confidence'] * 100:.2f}%."
    )


def render_comparison() -> None:
    st.subheader("Model comparison")
    rows = comparison_table()
    df = pd.DataFrame([{
        "Method": r["model"],
        "Test accuracy": f"{r['accuracy_pct']:.2f}%",
        "Training set": r["n_train"],
        "Test set": r["n_test"],
        "Source": r["source"],
    } for r in rows])
    st.dataframe(df, use_container_width=True, hide_index=True)

    best = max(rows, key=lambda r: r["accuracy_pct"])
    rf_acc = rows[0]["accuracy_pct"]
    cnn_acc = rows[1]["accuracy_pct"]
    resnet_acc = rows[2]["accuracy_pct"]
    st.success(
        f"🏆 **{best['model']}** wins with **{best['accuracy_pct']:.2f}%** test accuracy. "
        f"Transfer learning beats Random Forest by **{resnet_acc - rf_acc:.2f} percentage points** "
        f"and the self-built CNN by **{resnet_acc - cnn_acc:.2f} percentage points**."
    )

    # Bar chart
    fig = go.Figure(go.Bar(
        x=[r["accuracy_pct"] for r in rows],
        y=[r["model"] for r in rows],
        orientation="h",
        marker_color=["#94a3b8", "#3b82f6", "#1e40af"],
        text=[f"{r['accuracy_pct']:.2f}%" for r in rows],
        textposition="outside",
    ))
    fig.update_layout(
        xaxis_title="Test accuracy (%)",
        xaxis_range=[0, 100],
        height=280,
        margin=dict(l=10, r=80, t=10, b=30),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_training_curves() -> None:
    st.subheader("ResNet18 training curves")
    res = load_resnet18_results()
    history = res["history"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=history["train_acc"], mode="lines+markers", name="Train",
        line=dict(color="#3b82f6", width=2.5),
        marker=dict(size=7),
    ))
    fig.add_trace(go.Scatter(
        y=history["val_acc"], mode="lines+markers", name="Validation",
        line=dict(color="#ef4444", width=2.5),
        marker=dict(size=7),
    ))
    fig.update_layout(
        xaxis_title="Epoch",
        yaxis_title="Accuracy",
        yaxis_range=[0.5, 1.0],
        height=380,
        margin=dict(l=10, r=10, t=10, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        f"Best validation: {res['best_val_acc'] * 100:.2f}% · "
        f"Final test: {res['accuracy'] * 100:.2f}% · "
        f"Training set: {res['config']['n_train']} · "
        f"Test set: {res['config']['n_test']}"
    )


def render_about() -> None:
    with st.expander("About this project"):
        st.markdown(
            """
**Dataset.** [SDNET2018](https://digitalcommons.usu.edu/all_datasets/48/) — 56,092 real-world
concrete surface images released by Maguire, Dorafshan, and Thomas at Utah State University.
Three structure types are covered: bridge decks (D), pavements (P), and walls (W), with crack
widths ranging from 0.06 mm to 25 mm.

**Models compared.**

| Model | Approach | Parameters |
|---|---|---:|
| Random Forest | sklearn baseline on flattened grayscale pixels (4,096 features) | — |
| Self-built CNN | 4-block convolutional network trained from scratch | ~5.3M |
| ResNet18 | ImageNet pre-trained, fine-tuned end-to-end | ~11M |

**Reproduce locally.**

```bash
pip install -r requirements.txt
python -m src.train --model rf
python -m src.train --model cnn
python -m src.train --model resnet18
streamlit run app/streamlit_app.py
```

**Repository.** [github.com/jinliangyue/civil-engineering-crack-detection](https://github.com/jinliangyue/civil-engineering-crack-detection)
"""
        )


# === Page layout ===

st.title("🔍 Concrete Crack Detection")
st.markdown(
    f"**Transfer learning ablation on SDNET2018** — 56,092 real concrete images across "
    f"bridge decks, pavements, and walls. Test accuracy ladder: "
    f"**{headline_improvement()}**."
)

render_upload_panel()
st.markdown("---")
render_comparison()
st.markdown("---")
render_training_curves()
st.markdown("---")
render_about()
