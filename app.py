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


# ── Line Scan Tool ─────────────────────────────────────────────────────────────
def snap_to_axis(p1: tuple, p2: tuple, snap_deg: float = 5.0) -> tuple:
    """라인이 수평/수직에서 snap_deg 이내면 정확히 일직선으로 보정.

    수평 snap: |angle| < snap_deg 또는 |angle - 180| < snap_deg → y2 = y1
    수직 snap: |angle - 90| < snap_deg                            → x2 = x1
    그 외: 그대로 반환.
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    if dx == 0 and dy == 0:
        return p2
    angle = abs(np.degrees(np.arctan2(dy, dx)))
    # 수평 (0° 또는 180°)
    if angle < snap_deg or angle > (180 - snap_deg):
        return (int(p2[0]), int(p1[1]))
    # 수직 (90°)
    if abs(angle - 90) < snap_deg:
        return (int(p1[0]), int(p2[1]))
    return p2


def line_scan(img_array: np.ndarray, p1: tuple, p2: tuple, width: int = 0):
    """이미지 위 p1→p2 라인 따라 intensity profile 추출.

    img_array : 2D grayscale numpy array
    p1, p2    : (x, y) — 픽셀 좌표
    width     : 라인 수직 방향 적분 폭 (±픽셀). 0 = 단일 픽셀 라인.

    Returns: (distances_px, intensities)
    """
    x0, y0 = p1
    x1, y1 = p2
    length_px = float(np.hypot(x1 - x0, y1 - y0))
    if length_px < 1.0:
        return np.array([]), np.array([])

    n = max(int(length_px), 50)
    xs = np.linspace(x0, x1, n)
    ys = np.linspace(y0, y1, n)

    h, w_img = img_array.shape

    if width == 0:
        xs_c = np.clip(xs.astype(int), 0, w_img - 1)
        ys_c = np.clip(ys.astype(int), 0, h - 1)
        intensities = img_array[ys_c, xs_c].astype(np.float64)
    else:
        # 라인의 단위 수직 벡터
        dx, dy = x1 - x0, y1 - y0
        perp_x, perp_y = -dy / length_px, dx / length_px
        intensities = np.zeros(n, dtype=np.float64)
        for k in range(-width, width + 1):
            xs_w = np.clip((xs + k * perp_x).astype(int), 0, w_img - 1)
            ys_w = np.clip((ys + k * perp_y).astype(int), 0, h - 1)
            intensities += img_array[ys_w, xs_w].astype(np.float64)
        intensities /= float(2 * width + 1)

    distances = np.linspace(0.0, length_px, n)
    return distances, intensities


def detect_peaks(distances: np.ndarray, intensities: np.ndarray,
                 min_height_ratio:     float = 0.25,
                 min_prominence_ratio: float = 0.15,
                 min_dist_px:          float = 20.0) -> list:
    """Profile에서 모든 의미있는 local peak 검출 (prominence 기반).

    Args:
        distances             : 1D distance array (px)
        intensities           : 1D intensity array
        min_height_ratio      : peak 최소 절대 높이 (동적 범위 비율)
        min_prominence_ratio  : peak 사이 valley 깊이 (동적 범위 비율) —
                                같은 봉우리의 sub-peak 거름
        min_dist_px           : peak 간 최소 거리 (px)

    Returns: list of (distance, intensity) tuples, 위치 순.
    """
    n = len(intensities)
    if n < 5:
        return []

    y = intensities.astype(np.float64)

    # Smoothing (kernel 크게 — 평탄한 봉우리 정상부 진동 제거)
    k = max(5, n // 40)
    if k % 2 == 0:
        k += 1
    y_smooth = np.convolve(y, np.ones(k) / k, mode="same")

    # 동적 threshold
    baseline = float(np.percentile(y_smooth, 20))
    peak_max = float(y_smooth.max())
    dyn_range = peak_max - baseline
    if dyn_range < 1e-6:
        return []
    height_thresh   = baseline + dyn_range * min_height_ratio
    min_prominence  = dyn_range * min_prominence_ratio

    # Strict local maxima (>, 평탄 정상부 미세 진동 회피)
    candidates = [
        i for i in range(1, n - 1)
        if y_smooth[i] > y_smooth[i - 1]
        and y_smooth[i] > y_smooth[i + 1]
        and y_smooth[i] > height_thresh
    ]
    if not candidates:
        return []

    # ── Prominence 필터 ──────────────────────────────────────────────
    # 높이 내림차순으로 검토하면서, 이미 accept된 peak과 사이의 valley가
    # 충분히 깊은지 확인. 얕으면 같은 봉우리의 sub-peak → 거름.
    candidates.sort(key=lambda i: -y_smooth[i])
    accepted = []
    for c in candidates:
        # 거리 필터
        if any(abs(distances[c] - distances[a]) < min_dist_px for a in accepted):
            continue
        # Prominence 필터 (accept된 peak과 사이의 가장 깊은 valley 검사)
        is_isolated = True
        for a in accepted:
            lo, hi = min(c, a), max(c, a)
            valley = float(y_smooth[lo:hi + 1].min())
            smaller_top = min(y_smooth[c], y_smooth[a])
            if smaller_top - valley < min_prominence:
                is_isolated = False  # 같은 봉우리
                break
        if is_isolated:
            accepted.append(c)

    accepted.sort(key=lambda i: distances[i])
    return [(float(distances[i]), float(y[i])) for i in accepted]


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

        # ── Line Scan Tool ─────────────────────────────────────────────────────
        with st.expander("📏 Line Scan Tool — measure intensity profile", expanded=False):
            sk_p1   = f"ls_p1_{f.name}"
            sk_p2   = f"ls_p2_{f.name}"
            sk_last = f"ls_last_{f.name}"
            for k in (sk_p1, sk_p2, sk_last):
                if k not in st.session_state:
                    st.session_state[k] = None

            p1 = st.session_state[sk_p1]
            p2 = st.session_state[sk_p2]

            c_ctrl, c_img = st.columns([1, 2])

            with c_ctrl:
                st.markdown("**How to use:**")
                if p1 is None:
                    st.info("Click **first point** on image →")
                elif p2 is None:
                    st.info("Click **second point** →")
                else:
                    st.success(
                        f"Line: ({p1[0]}, {p1[1]}) → ({p2[0]}, {p2[1]})\n\n"
                        f"Length: {np.hypot(p2[0]-p1[0], p2[1]-p1[1]):.1f} px"
                    )

                if st.button("🔄 Reset", key=f"ls_reset_{f.name}"):
                    st.session_state[sk_p1]   = None
                    st.session_state[sk_p2]   = None
                    st.session_state[sk_last] = None
                    st.rerun()

                # Snap toggle: 수평/수직에서 5° 이내면 자동 정렬
                snap_on = st.checkbox(
                    "📐 Snap to horizontal/vertical",
                    value=True,
                    key=f"ls_snap_{f.name}",
                    help="If your second click is within 5° of horizontal or "
                         "vertical, the line is auto-aligned perfectly straight.",
                )

                # Integration width: 라인 수직 방향 평균 픽셀 수
                # (노이즈 완화 — 3px 정도가 일반적인 sweet spot)
                width_px = 3

            with c_img:
                # 원본 이미지 위에 markers + line 직접 그림
                display = img.copy().convert("RGB")
                drw     = ImageDraw.Draw(display)

                # 1) LINE (white outline + red core) — circles 아래
                if p1 is not None and p2 is not None:
                    drw.line([p1, p2], fill="white", width=8)
                    drw.line([p1, p2], fill="red",   width=4)

                # 2) Markers on top
                for pt in (p1, p2):
                    if pt is None:
                        continue
                    x, y = pt
                    drw.ellipse([x-11, y-11, x+11, y+11],
                                outline="white", width=3)
                    drw.ellipse([x-7,  y-7,  x+7,  y+7],
                                fill="red", outline="white", width=1)

                # State 변화마다 key 다르게 → client-side 이미지 cache 무효화
                state_id = f"{p1}_{p2}"
                value = streamlit_image_coordinates(
                    display,
                    key=f"ls_clicker_{f.name}_{state_id}",
                    use_column_width=True,
                )
                # Detect new click (key 바뀌면 value 가 None 으로 reset 됨 → 안전)
                if value is not None and value != st.session_state[sk_last]:
                    st.session_state[sk_last] = value

                    # ── 좌표 스케일링: 표시 좌표 → 원본 픽셀 좌표 ───────────
                    natural_w, natural_h = img.size
                    disp_w = value.get("width",  natural_w)
                    disp_h = value.get("height", natural_h)
                    if disp_w > 0 and disp_h > 0:
                        nx = int(round(value["x"] * natural_w / disp_w))
                        ny = int(round(value["y"] * natural_h / disp_h))
                    else:
                        nx = int(value["x"]);  ny = int(value["y"])
                    nx = max(0, min(natural_w - 1, nx))
                    ny = max(0, min(natural_h - 1, ny))
                    new_pt = (nx, ny)

                    if p1 is None:
                        st.session_state[sk_p1] = new_pt
                    elif p2 is None:
                        # 수평/수직 snap (toggle ON일 때)
                        final_pt = snap_to_axis(p1, new_pt) if snap_on else new_pt
                        st.session_state[sk_p2] = final_pt
                    st.rerun()

            # Profile + multi-peak detection
            if p1 is not None and p2 is not None:
                gray_arr = np.array(img.convert("L"), dtype=np.float64)
                distances, intensities = line_scan(gray_arr, p1, p2, width=width_px)

                if len(distances) > 0:
                    peaks = detect_peaks(distances, intensities)

                    # ── Plot: profile + 빨간 점으로 peak 표시 ─────────────
                    fig, ax = plt.subplots(figsize=(8, 3))
                    ax.plot(distances, intensities, "b-", linewidth=1.2,
                            label="Profile")
                    if peaks:
                        px = [p[0] for p in peaks]
                        py = [p[1] for p in peaks]
                        ax.plot(px, py, "ro", markersize=8,
                                markeredgecolor="white", markeredgewidth=1.5,
                                label=f"Peaks ({len(peaks)})")
                        # peak 위치에 세로 점선
                        for x_p in px:
                            ax.axvline(x=x_p, color="red", linestyle=":",
                                       linewidth=0.8, alpha=0.5)
                    ax.set_xlabel("Distance along line (px)")
                    ax.set_ylabel("Intensity")
                    ax.grid(True, alpha=0.3)
                    ax.legend(loc="upper right", fontsize=9)
                    ax.spines[["top", "right"]].set_visible(False)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                    # ── Metrics: peak count, positions, spacings ─────────
                    if len(peaks) == 0:
                        st.warning(
                            "No peaks detected — try a different line "
                            "or longer scan."
                        )
                    elif len(peaks) == 1:
                        st.info(
                            f"**1 peak detected** at position "
                            f"**{peaks[0][0]:.1f} px** (intensity "
                            f"{peaks[0][1]:.0f}). Draw a longer line "
                            f"across multiple streaks to measure spacing."
                        )
                    else:
                        positions = [p[0] for p in peaks]
                        spacings  = [positions[j + 1] - positions[j]
                                     for j in range(len(positions) - 1)]
                        mean_sp = float(np.mean(spacings))
                        std_sp  = float(np.std(spacings))

                        m1, m2, m3 = st.columns(3)
                        m1.metric("Peaks detected", f"{len(peaks)}")
                        m2.metric("Mean spacing",   f"{mean_sp:.1f} px")
                        m3.metric("Std (regularity)", f"± {std_sp:.1f} px")

                        # 자세한 정보
                        pos_str = "  ·  ".join(f"{p:.0f}" for p in positions)
                        sp_str  = "  ·  ".join(f"{s:.0f}" for s in spacings)
                        st.markdown(
                            f'<div style="font-size:13px;color:#555;'
                            f'margin-top:8px;line-height:1.7;">'
                            f'<b>Positions (px):</b> <span style="font-family:monospace;">{pos_str}</span><br>'
                            f'<b>Spacings (px):</b>  <span style="font-family:monospace;">{sp_str}</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
