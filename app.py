"""
RHEED Pattern Classifier — Public Web App
EfficientNetB2-based 4-class classifier
(Modulated / Anomalous Spots / Spotty / Streaks)
"""
import os
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RHEED Classifier",
    page_icon="🔬",
    layout="centered",
)

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_PATH   = os.path.join(os.path.dirname(__file__), "models", "Thomson_5.keras")
CLASS_NAMES  = ["Modulated", "Anomalous Spots", "Spotty", "Streaks"]
CLASS_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
CLASS_DESC   = {
    "Modulated":       "Periodic intensity modulation along streaks",
    "Anomalous Spots": "Irregular bright spots (transmission-like diffraction)",
    "Spotty":          "Discrete spots indicating 3D island growth",
    "Streaks":         "Continuous streaks — smooth 2D layer growth",
}
IMG_SIZE = (260, 260)


# ── Image preprocessing ────────────────────────────────────────────────────────
def safe_open_rgb(src) -> Image.Image:
    """Open any RHEED image (8-bit / 16-bit grayscale / RGB) as RGB."""
    img = Image.open(src)
    if img.mode in ("I", "I;16", "I;16B"):
        arr = np.array(img, dtype=np.float32)
        mn, mx = arr.min(), arr.max()
        if mx > mn:
            arr = (arr - mn) / (mx - mn) * 255.0
        img = Image.fromarray(arr.astype(np.uint8)).convert("RGB")
    elif img.mode != "RGB":
        img = img.convert("RGB")
    return img


def to_gray_stretched(img: Image.Image) -> np.ndarray:
    """Convert to single-channel float [0,1] with percentile contrast stretch.
    Uses max-channel so green phosphor signal is captured regardless of hue."""
    arr = np.array(img.convert("RGB")).astype(np.float32)
    gray = np.max(arr, axis=2)          # max-channel → brightest phosphor pixel
    p_lo = np.percentile(gray, 1)
    p_hi = np.percentile(gray, 99)
    if p_hi > p_lo:
        gray = np.clip((gray - p_lo) / (p_hi - p_lo), 0.0, 1.0)
    else:
        gray = gray / 255.0
    return gray                         # shape (H, W), float32, [0,1]


def crop_roi_by_brightest(gray: np.ndarray, roi_fraction: float = 0.55) -> np.ndarray:
    """Square-crop around the brightest region of the RHEED pattern.

    Strategy:
      1. Downsample to coarse block grid (avoids single hot pixels).
      2. Find the block with maximum mean brightness → pattern center.
      3. Cut a square of size (roi_fraction × min(H,W)) centred there.
    """
    h, w = gray.shape
    blk = max(1, min(h, w) // 30)
    hb  = (h // blk) * blk
    wb  = (w // blk) * blk
    coarse = (gray[:hb, :wb]
              .reshape(hb // blk, blk, wb // blk, blk)
              .mean(axis=(1, 3)))
    br, bc = np.unravel_index(np.argmax(coarse), coarse.shape)
    cy = int(br * blk + blk // 2)
    cx = int(bc * blk + blk // 2)
    half = int(min(h, w) * roi_fraction / 2)
    y0 = max(0, cy - half);  y1 = min(h, cy + half)
    x0 = max(0, cx - half);  x1 = min(w, cx + half)
    return gray[y0:y1, x0:x1]


def preprocess(img: Image.Image,
               lab_mode: bool = False,
               roi_fraction: float = 0.55) -> np.ndarray:
    """Prepare image for model input → (1, 260, 260, 3) float32."""
    if lab_mode:
        gray = to_gray_stretched(img)
        gray = crop_roi_by_brightest(gray, roi_fraction)
        gray8 = (gray * 255).astype(np.uint8)
        rgb   = np.stack([gray8, gray8, gray8], axis=-1)
        img   = Image.fromarray(rgb)
    img = img.resize(IMG_SIZE)
    return (np.array(img, dtype=np.float32) / 255.0)[np.newaxis]


def get_preprocessed_preview(img: Image.Image,
                              roi_fraction: float = 0.55) -> Image.Image:
    """Return the grayscale ROI crop that the model actually sees (before resize)."""
    gray  = to_gray_stretched(img)
    gray  = crop_roi_by_brightest(gray, roi_fraction)
    gray8 = (gray * 255).astype(np.uint8)
    rgb   = np.stack([gray8, gray8, gray8], axis=-1)
    return Image.fromarray(rgb)


# ── Model loading ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return keras.models.load_model(MODEL_PATH)


# ── Probability bar chart ──────────────────────────────────────────────────────
def plot_probs(probs):
    fig, ax = plt.subplots(figsize=(6, 2.8))
    bars = ax.barh(CLASS_NAMES, probs * 100, color=CLASS_COLORS)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Confidence (%)")
    ax.invert_yaxis()
    for bar, p in zip(bars, probs):
        ax.text(p * 100 + 1, bar.get_y() + bar.get_height() / 2,
                f"{p*100:.1f}%", va="center", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("🔬 RHEED Classifier")
    st.markdown(
        """
Classifies RHEED diffraction patterns into **4 categories** using a
EfficientNetB2 deep learning model trained on ~1,500 images.

---

**Classes**
| | Class | Pattern |
|---|---|---|
| 🔴 | Modulated | Periodic streak modulation |
| 🔵 | Anomalous Spots | Irregular transmission spots |
| 🟢 | Spotty | Discrete 3D island spots |
| 🟡 | Streaks | Smooth 2D growth streaks |

---

**Image type**
"""
    )

    lab_mode = st.toggle(
        "Lab image mode",
        value=False,
        help="Turn ON for raw camera images (green phosphor, dark background). "
             "Automatically extracts the bright RHEED region and converts to grayscale. "
             "Leave OFF for pre-processed grayscale images.",
    )

    if lab_mode:
        roi_fraction = st.slider(
            "ROI size",
            min_value=0.30,
            max_value=0.90,
            value=0.55,
            step=0.05,
            help="Fraction of the image used as ROI around the brightest point. "
                 "Increase if pattern is cut off; decrease to zoom in tighter.",
        )
        show_preview = st.checkbox("Show preprocessed preview", value=True,
                                   help="Display the grayscale ROI the model actually sees.")
    else:
        roi_fraction = 0.55
        show_preview = False

    st.divider()
    st.caption(f"Model: Thomson_5 · TF {tf.__version__}")
    st.caption("© 2026 rlack")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
st.header("Upload RHEED Image(s)")
st.caption("PNG / JPG / BMP / TIFF — 8-bit or 16-bit grayscale supported.")

model = load_model()
if model is None:
    st.error(f"Model file not found: `{MODEL_PATH}`")
    st.stop()

uploaded = st.file_uploader(
    "Drop files here or click to browse",
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)

if not uploaded:
    st.info("Upload one or more RHEED images to get started.")
    st.stop()

# ── Batch summary (multiple files) ────────────────────────────────────────────
if len(uploaded) > 1:
    st.divider()
    st.subheader(f"Summary — {len(uploaded)} images")
    rows = []
    for f in uploaded:
        try:
            img  = safe_open_rgb(f)
            arr  = preprocess(img, lab_mode, roi_fraction)
            prob = model.predict(arr, verbose=0)[0]
            top  = int(np.argmax(prob))
            rows.append({
                "File": f.name,
                "Prediction": CLASS_NAMES[top],
                "Confidence": f"{prob[top]*100:.1f}%",
                **{c: f"{p*100:.1f}%" for c, p in zip(CLASS_NAMES, prob)},
            })
            f.seek(0)
        except Exception as ex:
            rows.append({"File": f.name, "Prediction": "Error",
                         "Confidence": str(ex),
                         **{c: "-" for c in CLASS_NAMES}})

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Download CSV",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name="rheed_results.csv",
        mime="text/csv",
    )
    st.divider()

# ── Per-image detail ──────────────────────────────────────────────────────────
for f in uploaded:
    try:
        img = safe_open_rgb(f)
    except Exception as ex:
        st.error(f"`{f.name}` could not be opened: {ex}")
        continue

    arr  = preprocess(img, lab_mode, roi_fraction)
    prob = model.predict(arr, verbose=0)[0]
    top  = int(np.argmax(prob))
    cls  = CLASS_NAMES[top]
    conf = float(prob[top])
    col  = CLASS_COLORS[top]

    with st.container(border=True):
        # ── Image columns ──────────────────────────────────────────────────
        if lab_mode and show_preview:
            c_orig, c_pre, c_res = st.columns([1, 1, 1.4])
            with c_orig:
                st.image(img, caption=f"Original: {f.name}", use_container_width=True)
            with c_pre:
                prev = get_preprocessed_preview(img, roi_fraction)
                st.image(prev, caption="Model input (ROI)", use_container_width=True)
        else:
            c_img, c_res = st.columns([1, 1.4])
            with c_img:
                st.image(img, caption=f.name, use_container_width=True)

        with c_res:
            st.markdown(
                f"""
                <div style="padding:14px;border-radius:8px;
                            background:{col}18;border-left:5px solid {col};
                            margin-bottom:10px;">
                    <div style="font-size:11px;color:#888;letter-spacing:.5px;">
                        PREDICTION
                    </div>
                    <div style="font-size:26px;font-weight:700;color:{col};">
                        {cls}
                    </div>
                    <div style="font-size:13px;color:#555;margin-top:2px;">
                        Confidence: <b>{conf*100:.1f}%</b>
                    </div>
                    <div style="font-size:11px;color:#888;margin-top:6px;">
                        {CLASS_DESC[cls]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            fig = plot_probs(prob)
            st.pyplot(fig)
            plt.close(fig)

            if conf < 0.6:
                st.warning(
                    f"Low confidence ({conf*100:.1f}%). "
                    "The image may be ambiguous or outside the training distribution."
                )
