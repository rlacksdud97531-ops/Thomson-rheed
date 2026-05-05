"""
RHEED Pattern Classifier — Public Web App
EfficientNetB2-based 4-class classifier
(Mixed / Unclear / Spotty / Streaks)
"""
import os
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import cv2
from skimage.transform import radon

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RHEED Classifier",
    page_icon="🔬",
    layout="centered",
)

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_PATH   = os.path.join(os.path.dirname(__file__), "models", "Thomson_42_best_model.keras")
CLASS_NAMES  = ["Mixed", "Unclear", "Spotty", "Streaks"]
CLASS_COLORS = ["#374151", "#374151", "#374151", "#374151"]  # 단색 (gray-700)
IMG_SIZE     = (260, 260)


# ── Image loading ──────────────────────────────────────────────────────────────
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


# ── Crop dark top ────────────────────────────────────────────────────────────
def crop_dark_top(img: Image.Image) -> Image.Image:
    """위쪽 어두운 영역(깨진 검정 / 전자총 그림자) 제거.

    핵심 아이디어: 짧은 leakage / scratch 영역은 무시하고, **가장 긴 밝은 띠**
    (= 실제 RHEED 패턴 + glow)의 시작점에서 자른다.

    1) 행별 평균 밝기 → smoothing
    2) 적응형 threshold (mean + 0.5*std) 위/아래 binary
    3) 연속된 bright 구간(run)을 모두 찾아 가장 긴 구간의 시작점에서 crop
    """
    arr = np.array(img.convert("L"), dtype=np.float32)
    h, w = arr.shape

    row_mean = arr.mean(axis=1)

    # Smoothing
    k = max(15, h // 20)
    smoothed = np.convolve(row_mean, np.ones(k) / k, mode="same")

    # Threshold
    threshold = smoothed.mean() + smoothed.std() * 0.5
    bright = smoothed >= threshold

    # 연속 bright 구간 (run-length)
    padded = np.concatenate([[False], bright, [False]])
    diffs = np.diff(padded.astype(np.int32))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]

    if len(starts) == 0:
        return img

    lengths = ends - starts
    longest_idx = int(lengths.argmax())
    start = int(starts[longest_idx])

    # 너무 작으면 (전체의 5% 미만) 자르지 않음 (학습 이미지 보호)
    if start < h * 0.05:
        return img

    return img.crop((0, start, w, h))


# ── Grayscale conversion + auto-contrast ─────────────────────────────────────
def to_grayscale_rgb(img: Image.Image) -> Image.Image:
    """초록 인광 → 회색조 → autocontrast → RGB(R=G=B).
    autocontrast: 최소→0, 최대→255 매핑. 학습 이미지(고대비) 분포에 가까워짐."""
    gray = img.convert("L")
    gray = ImageOps.autocontrast(gray)
    return gray.convert("RGB")


# ── Preprocessing ──────────────────────────────────────────────────────────────
def preprocess(img: Image.Image) -> tuple[np.ndarray, Image.Image]:
    """검은 상단 제거 → grayscale → autocontrast → resize → normalize.
    Returns (array, processed_img) — processed_img는 시각화용."""
    cropped = crop_dark_top(img)
    gray = to_grayscale_rgb(cropped)
    resized = gray.resize(IMG_SIZE)
    arr = (np.array(resized, dtype=np.float32) / 255.0)[np.newaxis]
    return arr, gray


# ── Model loading ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return keras.models.load_model(MODEL_PATH)


# ── Surface reconstruction estimation ────────────────────────────────────────
def detect_reconstruction(model_input: np.ndarray) -> str:
    """Streak 간 가로 간격으로 표면 재구성(reconstruction) 추정.

    model_input: (260, 260, 3) float32 — 모델이 본 그 배열.
    Returns: '1×1', '2×1', '√2×√2 R45°', or '—'
    """
    gray = model_input.mean(axis=2)                   # (H, W)
    h, w = gray.shape

    # 가로 brightness profile (위/아래 노이즈 제외, 가운데 영역만 사용)
    r0, r1 = int(h * 0.25), int(h * 0.70)
    profile = gray[r0:r1].mean(axis=0)                # (W,)

    # Smoothing (~3% of width)
    k = max(3, w // 35)
    profile = np.convolve(profile, np.ones(k) / k, mode="same")

    pmax = profile.max()
    if pmax < 0.03:
        return "—"
    profile = profile / pmax

    # 15% 이상의 local maxima만 유효 streak peak
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

    if ratio > 0.82:        # 균등 간격 → bulk periodicity
        return "1×1"
    elif ratio > 0.60:      # ≈ 1/√2 ≈ 0.707
        return "√2×√2 R45°"
    elif ratio > 0.35:      # ≈ 1/2 = 0.50
        return "2×1"
    else:
        return "—"


# ── Kikuchi-like line detector ────────────────────────────────────────────────
def detect_kikuchi(gray_pil: Image.Image) -> tuple[np.ndarray, int, bool]:
    """Kikuchi-like band 검출 — Radon transform 방식.

    Pipeline:
      1. Aggressive CLAHE (large tile)        ← diffuse band 대비 강화
      2. Background subtraction (Gaussian σ ~ h/8)  ← 큰 구조만 남김
      3. ROI crop (상하단 + 좌우 마진)
      4. Downsample to 256px (Radon 속도)
      5. Radon transform on diagonal angles only (15-75°, 105-165°)
      6. Robust z-score (median + MAD) → top-K peak 검출
      7. (theta, rho) → 원본 이미지 라인 좌표 → cv2.line으로 그리기

    Radon은 line integral 기반이라 diffuse band도 잘 잡고,
    노이즈에 강함 (적분이 noise 평균화).

    Returns: (overlay_rgb, n_peaks, detected)
    """
    gray = np.array(gray_pil.convert("L"), dtype=np.uint8)
    h, w = gray.shape

    # ── 1. Aggressive CLAHE (큰 tile = 큰 구조 contrast 살림) ────────────
    clahe    = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16))
    enhanced = clahe.apply(gray)

    # ── 2. Background subtraction (large Gaussian = 배경 제거) ─────────
    sigma     = max(15.0, h / 8.0)
    bg        = cv2.GaussianBlur(enhanced.astype(np.float32), (0, 0), sigma)
    flat      = enhanced.astype(np.float32) - bg
    flat_pos  = np.clip(flat, 0, None)         # 밝은 band만 (양의 편차)
    flat_neg  = np.clip(-flat, 0, None)        # 어두운 band (음의 편차)
    flat_band = flat_pos + flat_neg            # 둘 다 합산 (Kikuchi는 양쪽)

    # ── 3. ROI ────────────────────────────────────────────────────────────
    y1, y2   = int(h * 0.10), int(h * 0.90)
    x1m, x2m = int(w * 0.08), int(w * 0.92)
    roi      = flat_band[y1:y2, x1m:x2m]
    rh, rw   = roi.shape

    # ── 4. Downsample (Radon은 O(N²·n_theta), 256px가 적절) ──────────
    target = 256
    if max(rh, rw) > target:
        scale  = target / float(max(rh, rw))
        new_h  = int(rh * scale); new_w = int(rw * scale)
        roi_sm = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        scale  = 1.0
        roi_sm = roi

    # ── 5. Radon transform — 대각 각도만 (양방향) ────────────────────
    theta_pos = np.linspace(15, 75,  31)       # / 방향 (positive slope)
    theta_neg = np.linspace(105, 165, 31)      # \ 방향 (negative slope)
    theta_all = np.concatenate([theta_pos, theta_neg])
    sino      = radon(roi_sm.astype(np.float32), theta=theta_all, circle=False)

    # ── 6. 각 angle column에서 robust z-score (median + MAD) ─────────
    col_med = np.median(sino, axis=0, keepdims=True)
    col_mad = np.median(np.abs(sino - col_med), axis=0, keepdims=True)
    zscore  = (sino - col_med) / (col_mad + 1e-6)

    # Top-K peak 검출 (non-max suppression)
    Z_THRESH    = 5.0      # 5 MAD above median
    MAX_PEAKS   = 6
    NMS_RHO     = 12       # ±12 row 억제
    NMS_THETA   = 3        # ±3 angle 억제
    z_work = zscore.copy()
    peaks  = []
    sh     = sino.shape[0]

    for _ in range(MAX_PEAKS):
        idx = int(np.argmax(z_work))
        r, c = np.unravel_index(idx, z_work.shape)
        if z_work[r, c] < Z_THRESH:
            break
        rho_small = r - sh / 2.0
        rho       = rho_small / scale          # downsample 보정
        peaks.append((float(theta_all[c]), float(rho), float(zscore[r, c])))
        # NMS
        r0, r1 = max(0, r - NMS_RHO), min(sh, r + NMS_RHO + 1)
        c0, c1 = max(0, c - NMS_THETA), min(z_work.shape[1], c + NMS_THETA + 1)
        z_work[r0:r1, c0:c1] = -1e9

    # ── 7. (theta, rho) → 라인 endpoint → 오버레이에 그리기 ──────────
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    # ROI 중심 (원본 좌표)
    roi_cx = x1m + rw / 2.0
    roi_cy = y1  + rh / 2.0
    L      = float(max(w, h)) * 1.5

    for theta_deg, rho, _z in peaks:
        # skimage radon 규약: theta 회전 후 column 합 = 그 angle column = projection
        # → line 방향: (sin θ, cos θ),  normal: (cos θ, -sin θ)
        th = np.radians(theta_deg)
        ct, st = np.cos(th), np.sin(th)
        # Foot of perpendicular from ROI center
        px = roi_cx + rho * ct
        py = roi_cy - rho * st
        # Line direction (perpendicular to normal)
        dx, dy = st, ct
        p1 = (int(px - L * dx), int(py - L * dy))
        p2 = (int(px + L * dx), int(py + L * dy))
        cv2.line(overlay, p1, p2, (220, 60, 60), 2, cv2.LINE_AA)

    detected = len(peaks) >= 1
    return overlay, len(peaks), detected


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
| Class | Pattern |
|---|---|
| Mixed | Periodic streak modulation |
| Unclear | Irregular transmission spots |
| Spotty | Discrete 3D island spots |
| Streaks | Smooth 2D growth streaks |

---
"""
    )
    st.caption(f"Model: Thomson_42 · TF {tf.__version__}")
    st.caption("© 2026 rlack")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
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
            arr, _gray = preprocess(img)
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

    arr, gray_img = preprocess(img)
    prob = model.predict(arr, verbose=0)[0]
    top  = int(np.argmax(prob))
    cls  = CLASS_NAMES[top]
    conf = float(prob[top])
    col  = CLASS_COLORS[top]

    with st.container(border=True):
        c_orig, c_gray, c_res = st.columns([1, 1, 1.4])
        with c_orig:
            st.image(img, caption=f"Original: {f.name}", use_container_width=True)
        with c_gray:
            st.image(gray_img, use_container_width=True)
        with c_res:
            fig = plot_probs(prob)
            st.pyplot(fig)
            plt.close(fig)

            # Streak / Mixed 일 때만 reconstruction 표시
            if cls in ("Streaks", "Mixed"):
                recon = detect_reconstruction(arr[0])
                st.markdown(
                    f'<div style="font-size:14px;color:#555;margin-top:6px;">'
                    f'<b>Surface reconstruction:</b> '
                    f'<span style="font-family:monospace;color:#000;'
                    f'font-weight:600;">{recon}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # ── Kikuchi-like line overlay ─────────────────────────────────────────
        st.divider()
        st.caption("🔍 Kikuchi-like line analysis (candidate detection only)")
        kk_overlay, kk_count, kk_detected = detect_kikuchi(gray_img)
        c_kk, c_kk_info = st.columns([2, 1])
        with c_kk:
            st.image(
                kk_overlay,
                caption=f"Red lines = Radon-detected band candidates ({kk_count} band(s))",
                use_container_width=True,
            )
        with c_kk_info:
            if kk_detected:
                st.success(f"✅ {kk_count} Kikuchi-like band(s) detected")
            else:
                st.info("⚪ Not detected")
            st.caption(
                "Detection: Radon transform on background-subtracted image. "
                "Diagonal angles (15–75°, 105–165°) only. "
                "Candidate indicator — physical confirmation requires expert review."
            )
