"""
RHEED Pattern Classifier — Public Web App
EfficientNetB2-based 4-class classifier
(Modulated / Anomalous Spots / Spotty / Streaks)
"""
import os
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps, ImageFilter
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
    """Convert lab image to [0,1] grayscale matching training data distribution.

    Steps:
      1. Max-channel grayscale (captures green phosphor regardless of hue).
      2. Background subtraction: gray - 0.85 × GaussianBlur(gray)
         → removes circular vignette while preserving streak intensity gradient.
         (Division was replaced with subtraction to avoid halo/diamond artefacts
          where streak tops appeared as isolated bright lozenges.)
      3. Percentile (p2–p98) contrast stretch → maps active range to [0, 1].
    """
    arr  = np.array(img.convert("RGB")).astype(np.float32)
    gray = np.max(arr, axis=2)                        # (H, W) max-channel

    # ── Background subtraction ─────────────────────────────────────────────
    # Large Gaussian blur estimates the slowly-varying vignette background.
    # Subtracting 85 % of it removes the circular gradient while keeping the
    # relative brightness variation along each streak (top bright → bottom dim),
    # which is what the model learned to recognise as a streak.
    radius = max(40, min(gray.shape) // 4)
    gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
    bg = np.array(
        Image.fromarray(gray_u8).filter(ImageFilter.GaussianBlur(radius=radius)),
        dtype=np.float32,
    )
    gray = np.clip(gray - bg * 0.92, 0.0, None)      # subtract background trend

    # ── Percentile stretch ─────────────────────────────────────────────────
    p_lo = np.percentile(gray, 2)
    p_hi = np.percentile(gray, 98)
    if p_hi > p_lo:
        gray = np.clip((gray - p_lo) / (p_hi - p_lo), 0.0, 1.0)
    else:
        mx   = gray.max()
        gray = (gray / mx) if mx > 0 else gray

    # ── Gamma correction ───────────────────────────────────────────────────
    # γ > 1 suppresses residual background (dark pixels → darker) while
    # keeping streak peaks close to 1.0, matching training data contrast.
    gray = np.power(gray, 1.5)

    return gray.astype(np.float32)                    # (H, W), [0, 1]


def crop_roi_by_brightest(gray: np.ndarray,
                          roi_fraction: float = 0.55,
                          skip_top: float = 0.25) -> np.ndarray:
    """Square-crop capturing the full streak length in the RHEED pattern.

    X (horizontal): brightness-weighted centroid of columns below the gun area,
                    with a Gaussian centre-bias to ignore the phosphor screen rim.
    Y (vertical):   fixed at the image centre — RHEED streaks always span the
                    centre of the circular screen, so anchoring on the brightest
                    row (streak top) is avoided; the full streak length is captured.
    """
    h, w = gray.shape

    # ── Horizontal: find the centre of the streak columns ─────────────────────
    sigma_x = w * 0.35
    weight_x = np.exp(
        -((np.arange(w, dtype=np.float32) - w / 2) ** 2) / (2 * sigma_x ** 2)
    )  # (W,) — down-weights phosphor rim

    start = int(h * skip_top)
    col_brightness = (gray[start:, :] * weight_x).mean(axis=0)   # (W,)

    # Smooth to avoid locking onto a single bright pixel
    k = max(3, w // 40)
    col_brightness = np.convolve(col_brightness, np.ones(k) / k, mode='same')
    cx = int(np.argmax(col_brightness))

    # ── Vertical: always image centre (captures full streak, not just bright top)
    cy = h // 2

    # ── Crop ──────────────────────────────────────────────────────────────────
    half = int(min(h, w) * roi_fraction / 2)
    y0 = max(0, cy - half);  y1 = min(h, cy + half)
    x0 = max(0, cx - half);  x1 = min(w, cx + half)
    return gray[y0:y1, x0:x1]


def preprocess(img: Image.Image,
               lab_mode: bool = False,
               roi_fraction: float = 0.55,
               skip_top: float = 0.25) -> np.ndarray:
    """Prepare image for model input → (1, 260, 260, 3) float32."""
    if lab_mode:
        gray = to_gray_stretched(img)
        gray = crop_roi_by_brightest(gray, roi_fraction, skip_top)
        gray8 = (gray * 255).astype(np.uint8)
        rgb   = np.stack([gray8, gray8, gray8], axis=-1)
        img   = Image.fromarray(rgb)
    img = img.resize(IMG_SIZE)
    return (np.array(img, dtype=np.float32) / 255.0)[np.newaxis]


def get_preprocessed_preview(img: Image.Image,
                              roi_fraction: float = 0.55,
                              skip_top: float = 0.25) -> Image.Image:
    """Return the grayscale ROI crop that the model actually sees (before resize)."""
    gray  = to_gray_stretched(img)
    gray  = crop_roi_by_brightest(gray, roi_fraction, skip_top)
    gray8 = (gray * 255).astype(np.uint8)
    rgb   = np.stack([gray8, gray8, gray8], axis=-1)
    return Image.fromarray(rgb)


# ── Model loading ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return keras.models.load_model(MODEL_PATH)


# ── Surface reconstruction from streak spacing ────────────────────────────────
def detect_reconstruction(model_input: np.ndarray) -> str:
    """Estimate surface reconstruction from horizontal streak spacing.
    model_input: (260, 260, 3) float32 — the array the model already received.
    Returns e.g. '1×1', '2×1', '√2×√2 R45°', or '—'.
    """
    gray = model_input.mean(axis=2)          # (H, W)
    h, w = gray.shape

    # Horizontal profile: mean over middle rows (skip gun shadow & bottom noise)
    r0, r1 = int(h * 0.25), int(h * 0.70)
    profile = gray[r0:r1].mean(axis=0)       # (W,)

    # Smooth with ~3 % of width
    k = max(3, w // 35)
    profile = np.convolve(profile, np.ones(k) / k, mode='same')

    pmax = profile.max()
    if pmax < 0.03:
        return "—"
    profile /= pmax

    # Local maxima above 15 % of peak
    peaks = [i for i in range(1, w - 1)
             if profile[i] > profile[i - 1]
             and profile[i] > profile[i + 1]
             and profile[i] > 0.15]

    if len(peaks) < 2:
        return "—"

    peaks.sort()
    gaps = [peaks[j + 1] - peaks[j] for j in range(len(peaks) - 1)]
    d_max = max(gaps)
    d_min = min(gaps)

    if d_max == 0:
        return "—"

    ratio = d_min / d_max

    if ratio > 0.82:      # evenly spaced → bulk periodicity
        return "1×1"
    elif ratio > 0.60:    # ≈ 1/√2 ≈ 0.707
        return "√2×√2 R45°"
    elif ratio > 0.35:    # ≈ 1/2 = 0.50
        return "2×1"
    else:
        return "—"


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

    skip_top     = 0.25   # exclude top 25 % (electron gun area)
    roi_fraction = 0.55   # ROI = 55 % of min(H, W) around brightest point
    show_preview = lab_mode  # always show preprocessing preview in lab mode

    st.divider()
    st.caption(f"Model: Thomson_5 · TF {tf.__version__}")
    st.caption("© 2026 rlack")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
st.header("Upload RHEED Image(s)")
# Hide Streamlit's auto-generated "50MB per file • …" helper text
st.markdown(
    "<style>small.st-emotion-cache-1b2d9b5, "
    "[data-testid='stFileUploaderDropzoneInstructions'] small { display:none !important; }"
    "</style>",
    unsafe_allow_html=True,
)

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
            arr  = preprocess(img, lab_mode, roi_fraction, skip_top)
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

    arr  = preprocess(img, lab_mode, roi_fraction, skip_top)
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
                prev = get_preprocessed_preview(img, roi_fraction, skip_top)
                st.image(prev, caption="Model input (ROI)", use_container_width=True)
        else:
            c_img, c_res = st.columns([1, 1.4])
            with c_img:
                st.image(img, caption=f.name, use_container_width=True)

        with c_res:
            # Surface reconstruction (only for ordered 2D surfaces)
            recon_html = ""
            if cls in ("Streaks", "Modulated"):
                recon = detect_reconstruction(arr[0])
                recon_html = (
                    f'<div style="font-size:12px;color:#555;margin-top:8px;">'
                    f'<b>Surface reconstruction:</b> {recon}</div>'
                )

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
                    {recon_html}
                </div>
                """,
                unsafe_allow_html=True,
            )
            fig = plot_probs(prob)
            st.pyplot(fig)
            plt.close(fig)

