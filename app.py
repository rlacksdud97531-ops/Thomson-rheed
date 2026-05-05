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
    """Kikuchi-like band 검출 — 2-track 방식.

    Track 1 (Hough): CLAHE → Bilateral → Canny → Hough 로 대각선 라인 세그먼트 검출
    Track 2 (Gradient): Sobel gradient direction 분석으로 대각 gradient 비율 측정

    Kikuchi bands = diffuse intensity gradients이므로 낮은 threshold 사용.
    둘 중 하나라도 감지되면 detected=True.

    gray_pil : 전처리된 grayscale PIL 이미지 (full-res, before 260×260 resize)
    Returns  : (overlay_rgb: np.ndarray uint8, n_hough_lines: int, detected: bool)

    ⚠️ 후보 검출 (geometry 기반) — 물리적 Kikuchi 확정이 아님.
    """
    gray = np.array(gray_pil.convert("L"), dtype=np.uint8)
    h, w = gray.shape

    # ── 공통 전처리 ──────────────────────────────────────────────────────────
    # Aggressive CLAHE — diffuse band 강화
    clahe    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # ROI: 상단 10%, 하단 10% 제외
    y1, y2   = int(h * 0.10), int(h * 0.90)

    # ── Track 1: Hough line detection ────────────────────────────────────────
    # Bilateral filter — edge 보존하면서 noise 제거 (Gaussian보다 diffuse band에 유리)
    bilateral = cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)
    edges     = cv2.Canny(bilateral, 15, 50)          # 낮은 threshold (diffuse용)
    roi_edges = edges[y1:y2, :]

    min_len = max(20, int(0.08 * min(h, w)))          # 더 짧은 선도 검출
    lines   = cv2.HoughLinesP(
        roi_edges, rho=1, theta=np.pi / 180,
        threshold=15,                                  # 낮은 vote threshold
        minLineLength=min_len, maxLineGap=30,          # 큰 gap 허용
    )

    kikuchi = []
    if lines is not None:
        for seg in lines:
            x1, ly1, x2, ly2 = seg[0]
            dx, dy = x2 - x1, ly2 - ly1
            angle  = abs(np.degrees(np.arctan2(dy, dx)))
            if angle > 90:
                angle = 180 - angle
            if 15 < angle < 75:             # 수평·수직 제외, 대각선만
                kikuchi.append((x1, ly1 + y1, x2, ly2 + y1))

    hough_detected = len(kikuchi) >= 2

    # ── Track 2: Gradient direction analysis ─────────────────────────────────
    # Sobel gradient — Kikuchi band = 대각 방향 intensity gradient
    roi_enh = enhanced[y1:y2, :].astype(np.float32)
    gx      = cv2.Sobel(roi_enh, cv2.CV_64F, 1, 0, ksize=3)
    gy      = cv2.Sobel(roi_enh, cv2.CV_64F, 0, 1, ksize=3)
    mag     = np.sqrt(gx ** 2 + gy ** 2)
    ang     = np.abs(np.degrees(np.arctan2(gy, gx)))
    ang[ang > 90] = 180 - ang[ang > 90]

    # 상위 20% gradient 중 대각 방향 비율
    thresh       = np.percentile(mag, 80)
    strong_mask  = mag > thresh
    diag_mask    = (ang > 15) & (ang < 75)
    diag_ratio   = float((strong_mask & diag_mask).mean())
    grad_detected = diag_ratio > 0.12    # 강한 gradient의 12% 이상이 대각 방향

    detected = hough_detected or grad_detected

    # ── 오버레이: grayscale + 빨간 Hough 라인 ────────────────────────────────
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    for (x1, ly1, x2, ly2) in kikuchi:
        cv2.line(overlay, (x1, ly1), (x2, ly2), (220, 60, 60), 2)

    return overlay, len(kikuchi), detected


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
                caption=f"Red lines = Hough diagonal candidates ({kk_count} line(s))",
                use_container_width=True,
            )
        with c_kk_info:
            if kk_detected:
                st.success("✅ Kikuchi-like features detected")
            else:
                st.info("⚪ Not detected")
            st.caption(
                "Two-track detection: Hough line geometry + "
                "Sobel gradient direction analysis. "
                "Candidate indicator only — physical confirmation requires expert review."
            )
