"""
RHEED Pattern Classifier — Public Web App
EfficientNetB2-based 4-class classifier
(Mixed / Unclear / Spotty / Streaks)
"""
import os
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps, ImageDraw
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from streamlit_image_coordinates import streamlit_image_coordinates
from scipy.optimize import curve_fit

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


# ── Peak detection (used by Spot Box Tool) ─────────────────────────────────────
def _find_valleys(y_smooth: np.ndarray, peak_idx: int,
                  all_peak_indices: list) -> tuple[int, int]:
    """현재 peak 의 좌측 / 우측 valley 인덱스 반환.

    좌측 valley: 직전 peak (없으면 시작) 부터 현 peak 까지의 argmin
    우측 valley: 현 peak 부터 직후 peak (없으면 끝) 까지의 argmin
    """
    n = len(y_smooth)
    left_peaks = [p for p in all_peak_indices if p < peak_idx]
    left_bound = max(left_peaks) if left_peaks else 0
    left_valley = left_bound + int(np.argmin(y_smooth[left_bound:peak_idx + 1]))

    right_peaks = [p for p in all_peak_indices if p > peak_idx]
    right_bound = min(right_peaks) if right_peaks else n - 1
    right_valley = peak_idx + int(np.argmin(y_smooth[peak_idx:right_bound + 1]))

    return left_valley, right_valley


def _local_baseline(y_smooth: np.ndarray, distances: np.ndarray,
                    peak_idx: int, left_valley: int, right_valley: int) -> float:
    """좌측 valley 와 우측 valley 직선의 peak 위치 값 (linear baseline subtraction)."""
    y_l, y_r = float(y_smooth[left_valley]),  float(y_smooth[right_valley])
    x_l, x_r = float(distances[left_valley]), float(distances[right_valley])
    x_p      = float(distances[peak_idx])
    if abs(x_r - x_l) < 1e-9:
        return (y_l + y_r) / 2.0
    t = (x_p - x_l) / (x_r - x_l)
    return y_l + t * (y_r - y_l)


def _gauss_func(x, amp, mu, sigma, bl):
    """Single Gaussian: bl + amp · exp(-(x-mu)² / (2σ²))."""
    return bl + amp * np.exp(-(x - mu) ** 2 / (2.0 * sigma ** 2))


def _fit_gaussian(distances: np.ndarray, intensities: np.ndarray,
                  peak_idx: int, baseline_init: float,
                  left_valley: int, right_valley: int) -> dict:
    """단일 Gaussian fit (baseline 포함 4-parameter).

    Fit window: [left_valley, right_valley] — 인접 peak 영향 회피.
    Returns dict:
        success     : 수렴 성공 여부
        fwhm        : 2√(2 ln 2) · |σ| ≈ 2.355 σ
        r2          : 적합도 (1=완벽)
        mu, sigma, amp, baseline : fit 파라미터
    """
    x_fit = distances[left_valley:right_valley + 1].astype(np.float64)
    y_fit = intensities[left_valley:right_valley + 1].astype(np.float64)

    if len(x_fit) < 5:
        return {"success": False, "fwhm": np.nan, "r2": np.nan,
                "mu": np.nan, "sigma": np.nan, "amp": np.nan,
                "baseline": baseline_init}

    # Initial guesses
    peak_y    = float(intensities[peak_idx])
    peak_x    = float(distances[peak_idx])
    amp_init  = max(peak_y - baseline_init, 1.0)
    # σ 초기값 — 데이터 위 half-max 폭에서 추정
    half_init = baseline_init + amp_init / 2.0
    above     = y_fit > half_init
    if above.any():
        idx_a    = np.where(above)[0]
        rough_fw = float(x_fit[idx_a[-1]] - x_fit[idx_a[0]])
        sigma_init = max(rough_fw / 2.355, 1.0)
    else:
        sigma_init = max((x_fit[-1] - x_fit[0]) / 6.0, 1.0)

    try:
        popt, _ = curve_fit(
            _gauss_func, x_fit, y_fit,
            p0=[amp_init, peak_x, sigma_init, baseline_init],
            bounds=([0.0,      float(x_fit[0]),  0.1,                       -np.inf],
                    [np.inf,   float(x_fit[-1]), float(x_fit[-1] - x_fit[0]) * 2.0,
                                                                              np.inf]),
            maxfev=2000,
        )
    except (RuntimeError, ValueError):
        return {"success": False, "fwhm": np.nan, "r2": np.nan,
                "mu": np.nan, "sigma": np.nan, "amp": np.nan,
                "baseline": baseline_init}

    amp_f, mu_f, sigma_f, bl_f = (float(v) for v in popt)
    y_pred  = _gauss_func(x_fit, *popt)
    ss_res  = float(np.sum((y_fit - y_pred) ** 2))
    ss_tot  = float(np.sum((y_fit - y_fit.mean()) ** 2))
    r2      = 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else 0.0
    fwhm    = 2.0 * np.sqrt(2.0 * np.log(2.0)) * abs(sigma_f)

    return {
        "success": True,
        "fwhm":    float(fwhm),
        "r2":      float(r2),
        "mu":      mu_f,
        "sigma":   abs(sigma_f),
        "amp":     amp_f,
        "baseline": bl_f,
    }


def _compute_fwhm(distances: np.ndarray, y_smooth: np.ndarray,
                  peak_idx: int, baseline: float) -> dict:
    """Peak에서 양쪽으로 half-max 만나는 지점까지 걸어가서 FWHM 계산.

    Sub-pixel 정밀도: half-max 교차 지점은 선형 보간.
    인접 peak 침범 방지: intensity가 다시 상승하면 그 직전 valley를 경계로.

    Returns: dict — left_x, right_x, fwhm, half_max
    """
    n        = len(y_smooth)
    peak_y   = float(y_smooth[peak_idx])
    half_max = baseline + (peak_y - baseline) / 2.0

    # ── 왼쪽 ────────────────────────────────────────────────────────
    left_x  = float(distances[0])
    prev_y  = peak_y
    for i in range(peak_idx - 1, -1, -1):
        yi = float(y_smooth[i])
        if yi <= half_max:
            # 선형 보간: y[i] (below) → y[i+1] (above) 사이에서 half_max 교차
            y1, y2 = yi, float(y_smooth[i + 1])
            x1, x2 = float(distances[i]), float(distances[i + 1])
            if abs(y2 - y1) > 1e-9:
                t = (half_max - y1) / (y2 - y1)
                left_x = x1 + t * (x2 - x1)
            else:
                left_x = x1
            break
        if yi > prev_y:
            # 다시 상승 → 인접 peak 침범. 직전 valley 지점을 경계로.
            left_x = float(distances[i + 1])
            break
        prev_y = yi

    # ── 오른쪽 ──────────────────────────────────────────────────────
    right_x = float(distances[-1])
    prev_y  = peak_y
    for i in range(peak_idx + 1, n):
        yi = float(y_smooth[i])
        if yi <= half_max:
            y1, y2 = float(y_smooth[i - 1]), yi
            x1, x2 = float(distances[i - 1]), float(distances[i])
            if abs(y2 - y1) > 1e-9:
                t = (half_max - y1) / (y2 - y1)
                right_x = x1 + t * (x2 - x1)
            else:
                right_x = x2
            break
        if yi > prev_y:
            right_x = float(distances[i - 1])
            break
        prev_y = yi

    return {
        "left_x":   left_x,
        "right_x":  right_x,
        "fwhm":     right_x - left_x,
        "half_max": half_max,
    }


def detect_peaks(distances: np.ndarray, intensities: np.ndarray,
                 min_height_ratio:     float = 0.25,
                 min_prominence_ratio: float = 0.15,
                 min_dist_px:          float = 20.0) -> list:
    """Profile에서 모든 의미있는 local peak 검출 + 각 peak의 FWHM 계산.

    Args:
        distances             : 1D distance array (px)
        intensities           : 1D intensity array
        min_height_ratio      : peak 최소 절대 높이 (동적 범위 비율)
        min_prominence_ratio  : peak 사이 valley 깊이 (동적 범위 비율)
        min_dist_px           : peak 간 최소 거리 (px)

    Returns: list of dicts (위치 순) with keys:
        x          : peak 위치 (px)
        y          : peak intensity (원본)
        fwhm       : FWHM (px)
        fwhm_left  : FWHM 왼쪽 경계 x (px)
        fwhm_right : FWHM 오른쪽 경계 x (px)
        half_max   : 반높이 강도 (시각화용)
    """
    n = len(intensities)
    if n < 5:
        return []

    y = intensities.astype(np.float64)

    # Smoothing (kernel 크게 — 평탄한 봉우리 정상부 진동 제거)
    k = max(5, n // 40)
    if k % 2 == 0:
        k += 1
    # Edge padding: mode="same" 은 0-padding 으로 가장자리 값이 인위적으로
    # 낮아지는 artifact 발생. edge value 로 패딩 후 'valid' convolve 사용.
    pad      = k // 2
    y_padded = np.pad(y, pad, mode="edge")
    y_smooth = np.convolve(y_padded, np.ones(k) / k, mode="valid")

    # 동적 threshold
    baseline  = float(np.percentile(y_smooth, 20))
    peak_max  = float(y_smooth.max())
    dyn_range = peak_max - baseline
    if dyn_range < 1e-6:
        return []
    height_thresh  = baseline + dyn_range * min_height_ratio
    min_prominence = dyn_range * min_prominence_ratio

    # Strict local maxima
    candidates = [
        i for i in range(1, n - 1)
        if y_smooth[i] > y_smooth[i - 1]
        and y_smooth[i] > y_smooth[i + 1]
        and y_smooth[i] > height_thresh
    ]
    if not candidates:
        return []

    # Prominence 필터
    candidates.sort(key=lambda i: -y_smooth[i])
    accepted = []
    for c in candidates:
        if any(abs(distances[c] - distances[a]) < min_dist_px for a in accepted):
            continue
        is_isolated = True
        for a in accepted:
            lo, hi = min(c, a), max(c, a)
            valley = float(y_smooth[lo:hi + 1].min())
            smaller_top = min(y_smooth[c], y_smooth[a])
            if smaller_top - valley < min_prominence:
                is_isolated = False
                break
        if is_isolated:
            accepted.append(c)

    accepted.sort(key=lambda i: distances[i])

    # 각 accepted peak 마다 valleys → local baseline → walk FWHM + Gaussian fit
    peaks = []
    for idx in accepted:
        left_v, right_v = _find_valleys(y_smooth, idx, accepted)
        local_bl        = _local_baseline(y_smooth, distances, idx, left_v, right_v)
        fw              = _compute_fwhm(distances, y_smooth, idx, local_bl)  # 시각용
        gauss           = _fit_gaussian(distances, y, idx, local_bl, left_v, right_v)

        # 메인 FWHM = Gaussian fit 성공 시 Gaussian, 실패 시 walk fallback
        fwhm_primary = gauss["fwhm"] if gauss["success"] else fw["fwhm"]

        peaks.append({
            "x":          float(distances[idx]),
            "y":          float(y[idx]),
            # ── 메인 결과 ──
            "fwhm":       float(fwhm_primary),
            "r2":         float(gauss["r2"]) if gauss["success"] else np.nan,
            "fit_ok":     bool(gauss["success"]),
            "gauss":      gauss,                # full fit params (그래프 overlay용)
            # ── 시각 보조 (빨간 가로선용, 숫자 표시 X) ──
            "fwhm_left":  fw["left_x"],
            "fwhm_right": fw["right_x"],
            "half_max":   fw["half_max"],
            "baseline":   local_bl,
            # ── valley 인덱스 (그래프 baseline LINE 그릴 때 사용) ──
            "left_valley_x":  float(distances[left_v]),
            "left_valley_y":  float(y_smooth[left_v]),
            "right_valley_x": float(distances[right_v]),
            "right_valley_y": float(y_smooth[right_v]),
        })
    return peaks


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

        # ── Spot Box Tool — 1-click horizontal profile + peak detection ───────
        with st.expander("🎯 Spot Box Tool — 1-click streak profile", expanded=False):
            sk_center = f"sb_center_{f.name}"
            sk_last_b = f"sb_last_{f.name}"
            for k in (sk_center, sk_last_b):
                if k not in st.session_state:
                    st.session_state[k] = None
            center = st.session_state[sk_center]

            c_ctrl_b, c_img_b = st.columns([1, 2])

            with c_ctrl_b:
                if center is None:
                    st.info("👆 Click on the image")
                else:
                    st.success(f"Center: ({center[0]}, {center[1]})")

                if st.button("🔄 Reset", key=f"sb_reset_{f.name}"):
                    st.session_state[sk_center] = None
                    st.session_state[sk_last_b] = None
                    st.rerun()

                box_w = st.slider(
                    "Box width (px)",
                    min_value=100, max_value=2000, value=800, step=20,
                    key=f"sb_w_{f.name}",
                    help="Horizontal extent — how far left/right to scan.",
                )
                box_h = st.slider(
                    "Box height (px)",
                    min_value=5, max_value=100, value=20, step=1,
                    key=f"sb_h_{f.name}",
                    help="Vertical extent — pixels averaged for noise reduction.",
                )

            with c_img_b:
                # Box 오버레이 그림
                display_b = img.copy().convert("RGB")
                drw_b     = ImageDraw.Draw(display_b)

                if center is not None:
                    cx, cy = center
                    hw, hh = box_w // 2, box_h // 2
                    iw, ih = img.size
                    bx0 = max(0,  cx - hw); bx1 = min(iw, cx + hw)
                    by0 = max(0,  cy - hh); by1 = min(ih, cy + hh)
                    drw_b.rectangle([bx0 - 1, by0 - 1, bx1 + 1, by1 + 1],
                                    outline="white", width=4)
                    drw_b.rectangle([bx0, by0, bx1, by1],
                                    outline="red",   width=2)
                    drw_b.ellipse([cx - 5, cy - 5, cx + 5, cy + 5],
                                  fill="red", outline="white", width=1)

                state_id_b = f"{center}_{box_w}_{box_h}"
                value_b = streamlit_image_coordinates(
                    display_b,
                    key=f"sb_clicker_{f.name}_{state_id_b}",
                    use_column_width=True,
                )

                if value_b is not None and value_b != st.session_state[sk_last_b]:
                    st.session_state[sk_last_b] = value_b
                    natural_w, natural_h = img.size
                    disp_w = value_b.get("width",  natural_w)
                    disp_h = value_b.get("height", natural_h)
                    if disp_w > 0 and disp_h > 0:
                        nx = int(round(value_b["x"] * natural_w / disp_w))
                        ny = int(round(value_b["y"] * natural_h / disp_h))
                    else:
                        nx = int(value_b["x"]);  ny = int(value_b["y"])
                    nx = max(0, min(natural_w - 1, nx))
                    ny = max(0, min(natural_h - 1, ny))
                    st.session_state[sk_center] = (nx, ny)
                    st.rerun()

            # ── Auto profile + peak detection ────────────────────────────────
            if center is not None:
                gray_arr = np.array(img.convert("L"), dtype=np.float64)
                cx, cy   = center
                hw, hh   = box_w // 2, box_h // 2
                ih, iw   = gray_arr.shape
                x0 = max(0,  cx - hw); x1 = min(iw, cx + hw + 1)
                y0 = max(0,  cy - hh); y1 = min(ih, cy + hh + 1)
                roi = gray_arr[y0:y1, x0:x1]

                if roi.size == 0:
                    st.warning("Box is out of image bounds.")
                else:
                    # 수직 평균 → 1D 수평 profile
                    profile   = roi.mean(axis=0)
                    distances = np.arange(len(profile), dtype=np.float64)
                    peaks     = detect_peaks(distances, profile)

                    fig, ax = plt.subplots(figsize=(8, 3))
                    ax.plot(distances, profile, "b-", linewidth=1.2,
                            label="Profile (avg)")
                    if peaks:
                        px = [p["x"] for p in peaks]
                        py = [p["y"] for p in peaks]
                        # Sloped local baseline lines (valley → valley, 회색 점선)
                        for p in peaks:
                            ax.plot([p["left_valley_x"], p["right_valley_x"]],
                                    [p["left_valley_y"], p["right_valley_y"]],
                                    color="gray", linestyle="--",
                                    linewidth=1.0, alpha=0.6)
                        # Gaussian fit 곡선 overlay (주황색)
                        for p in peaks:
                            if not p["fit_ok"]:
                                continue
                            g = p["gauss"]
                            x_g = np.linspace(p["left_valley_x"], p["right_valley_x"], 100)
                            y_g = _gauss_func(x_g, g["amp"], g["mu"],
                                              g["sigma"], g["baseline"])
                            ax.plot(x_g, y_g, color="darkorange",
                                    linewidth=1.8, alpha=0.85)
                        # 시각 보조: 반높이 빨간 가로선 (숫자는 안 보고함)
                        for p in peaks:
                            ax.hlines(y=p["half_max"],
                                      xmin=p["fwhm_left"], xmax=p["fwhm_right"],
                                      colors="red", linewidth=1.5, alpha=0.5)
                        # Peak 점선 (위치 표시) + 빨간 점
                        for x_p in px:
                            ax.axvline(x=x_p, color="red", linestyle=":",
                                       linewidth=0.8, alpha=0.3)
                        ax.plot(px, py, "ro", markersize=8,
                                markeredgecolor="white", markeredgewidth=1.5,
                                label=f"Peaks ({len(peaks)})")
                        # Legend entries (dummy)
                        ax.plot([], [], color="darkorange", linewidth=1.8,
                                label="Gaussian fit")
                        ax.plot([], [], color="gray", linestyle="--",
                                linewidth=1.0, label="Local baseline")
                    ax.set_xlabel(
                        f"Distance from left edge of box (px)  ·  "
                        f"Box: {x1 - x0} × {y1 - y0}"
                    )
                    ax.set_ylabel("Intensity (avg)")
                    ax.grid(True, alpha=0.3)
                    ax.legend(loc="upper right", fontsize=9)
                    ax.spines[["top", "right"]].set_visible(False)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                    # ── Metrics ──────────────────────────────────────────
                    if len(peaks) == 0:
                        st.warning(
                            "No peaks detected — try larger box width or "
                            "move center."
                        )
                    elif len(peaks) == 1:
                        p0 = peaks[0]
                        r2_str = (f"R²={p0['r2']:.2f}" if p0["fit_ok"]
                                  else "fit failed")
                        st.info(
                            f"**1 peak detected** at **{p0['x']:.1f} px** — "
                            f"FWHM **{p0['fwhm']:.1f} px** ({r2_str}). "
                            f"Increase box width to capture more streaks."
                        )
                    else:
                        positions = [p["x"]    for p in peaks]
                        fwhms     = [p["fwhm"] for p in peaks]
                        r2s       = [p["r2"]   for p in peaks if p["fit_ok"]]
                        spacings  = [positions[j + 1] - positions[j]
                                     for j in range(len(positions) - 1)]
                        mean_sp = float(np.mean(spacings))
                        std_sp  = float(np.std(spacings))
                        mean_fw = float(np.mean(fwhms))
                        std_fw  = float(np.std(fwhms))
                        mean_r2 = float(np.mean(r2s)) if r2s else float("nan")

                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Peaks detected", f"{len(peaks)}")
                        m2.metric("Mean spacing",   f"{mean_sp:.1f} px",
                                  delta=f"± {std_sp:.1f} px", delta_color="off")
                        m3.metric("Mean FWHM",      f"{mean_fw:.1f} px",
                                  delta=f"± {std_fw:.1f} px", delta_color="off")
                        m4.metric("Mean Fit R²",
                                  f"{mean_r2:.3f}" if not np.isnan(mean_r2) else "—",
                                  help="Gaussian fit quality (1.0 = perfect). "
                                       "≥0.95 reliable, 0.7–0.95 cautious, "
                                       "<0.7 suspect (non-Gaussian / overlap).")

                        # Per-peak 상세 — R² 기반 신뢰도 색 코딩
                        def _rel_marker(r2):
                            if np.isnan(r2):  return "⚪ fit failed"
                            if r2 >= 0.95:    return "🟢 reliable"
                            if r2 >= 0.70:    return "🟡 cautious"
                            return "🔴 suspect"

                        rows = []
                        for j, p in enumerate(peaks, start=1):
                            rows.append(
                                f"  Peak {j}:  x = {p['x']:>6.1f} px  ·  "
                                f"FWHM = {p['fwhm']:>5.1f} px  ·  "
                                f"R² = {p['r2']:.3f}  {_rel_marker(p['r2'])}"
                                if p["fit_ok"]
                                else
                                f"  Peak {j}:  x = {p['x']:>6.1f} px  ·  "
                                f"FWHM = {p['fwhm']:>5.1f} px (walk fallback)  "
                                f"⚪ fit failed"
                            )
                        sp_str = "  ·  ".join(f"{s:.0f}" for s in spacings)

                        st.markdown(
                            f'<div style="font-size:13px;color:#555;'
                            f'margin-top:8px;line-height:1.7;font-family:monospace;">'
                            + "<br>".join(rows) +
                            f'<br><br><b>Spacings (px):</b> {sp_str}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
