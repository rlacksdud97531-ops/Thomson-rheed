"""
RHEED Pattern Classifier — Public Web App
EfficientNetB2-based 4-class classifier
(Mixed / Unclear / Spotty / Streaks)
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
CLASS_NAMES  = ["Mixed", "Unclear", "Spotty", "Streaks"]
CLASS_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
CLASS_DESC   = {
    "Mixed":   "Periodic intensity modulation along streaks",
    "Unclear": "Irregular bright spots (transmission-like diffraction)",
    "Spotty":  "Discrete spots indicating 3D island growth",
    "Streaks": "Continuous streaks — smooth 2D layer growth",
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


def crop_black_top(img: Image.Image, threshold: float = 0.04) -> Image.Image:
    """위 절반이 어두우면 (평균 밝기 < threshold) 상단 절반 제거.

    RHEED 실험실 이미지에서 전자총 그림자로 인한 검은 상단 영역을 제거.
    threshold: 0~1 범위, 기본값 0.04 (4% 밝기)
    """
    arr = np.array(img.convert("L"), dtype=np.float32) / 255.0
    h = arr.shape[0]
    top_brightness = arr[:h // 2].mean()
    if top_brightness < threshold:
        return img.crop((0, h // 2, img.width, h))
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
    # p_lo = p50: median and below → 0 (black background).
    # Background pixels sit near the median after subtraction;
    # streak pixels are above it → only streaks survive.
    p_lo = np.percentile(gray, 50)
    p_hi = np.percentile(gray, 99)
    if p_hi > p_lo:
        gray = np.clip((gray - p_lo) / (p_hi - p_lo), 0.0, 1.0)
    else:
        mx   = gray.max()
        gray = (gray / mx) if mx > 0 else gray

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

    # Gaussian centre-bias weight (horizontal) — suppresses phosphor rim
    sigma_x = w * 0.35
    weight_x = np.exp(
        -((np.arange(w, dtype=np.float32) - w / 2) ** 2) / (2 * sigma_x ** 2)
    )  # (W,)

    start = int(h * skip_top)
    active = gray[start:, :]          # region below gun shadow

    # ── Horizontal: brightest column = streak centre ───────────────────────
    col_brightness = (active * weight_x).mean(axis=0)
    k = max(3, w // 40)
    col_brightness = np.convolve(col_brightness, np.ones(k) / k, mode='same')
    cx = int(np.argmax(col_brightness))

    # ── Vertical: centroid of the top-25 % brightest rows ─────────────────
    # Streaks live in the brightest rows; their centroid gives the vertical
    # centre of the actual pattern (not the fixed image centre which often
    # pulls the crop into the bright phosphor background below the streaks).
    row_brightness = (active * weight_x).mean(axis=1)
    thresh_row = np.percentile(row_brightness, 75)
    bright_rows = np.where(row_brightness >= thresh_row)[0]
    if len(bright_rows) > 0:
        cy = int(np.mean(bright_rows)) + start
    else:
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


def predict_auto(model, img: Image.Image,
                 roi_fraction: float = 0.55,
                 skip_top: float = 0.25):
    """Normal mode와 Lab mode 단일 예측 후 confidence 높은 쪽 반환.
    위 절반이 어두우면 자동으로 상단 크롭 적용.

    Returns:
        prob      : 선택된 예측 확률 벡터
        used_lab  : Lab mode가 선택됐으면 True
    """
    img = crop_black_top(img)   # 위 절반 검정이면 자동 제거

    arr_normal = preprocess(img, lab_mode=False,
                            roi_fraction=roi_fraction, skip_top=skip_top)
    arr_lab    = preprocess(img, lab_mode=True,
                            roi_fraction=roi_fraction, skip_top=skip_top)

    prob_normal = model.predict(arr_normal, verbose=0)[0]
    prob_lab    = model.predict(arr_lab,    verbose=0)[0]

    if prob_normal.max() >= prob_lab.max():
        return prob_normal, False
    else:
        return prob_lab, True


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
| 🔴 | Mixed | Periodic streak modulation |
| 🔵 | Unclear | Irregular transmission spots |
| 🟢 | Spotty | Discrete 3D island spots |
| 🟡 | Streaks | Smooth 2D growth streaks |

---
"""
    )

    skip_top     = 0.25
    roi_fraction = 0.55

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
            prob, _ = predict_auto(model, img, roi_fraction, skip_top)
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

    prob, used_lab = predict_auto(model, img, roi_fraction, skip_top)
    top  = int(np.argmax(prob))
    cls  = CLASS_NAMES[top]
    conf = float(prob[top])
    col  = CLASS_COLORS[top]

    with st.container(border=True):
        # ── Image columns ──────────────────────────────────────────────────
        if used_lab:
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
            if cls in ("Streaks", "Mixed"):
                arr = preprocess(img, used_lab, roi_fraction, skip_top)
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

