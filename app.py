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
IMG_SIZE     = (260, 260)

# Substrate database — bulk in-plane lattice constant (Å)
SUBSTRATES = {
    "Sapphire (Al₂O₃) c-plane":  4.785,
    "Si (100)":                  5.431,
    "Si (111)":                  5.431,
    "GaAs (100)":                5.653,
    "SrTiO₃ (100)":              3.905,
    "MgO (100)":                 4.213,
    "LaAlO₃ (100)":              3.789,
    "MgAl₂O₄ (100)":             8.083,
    "Custom":                    None,
}


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


# ── Streak detection (peaks + spacing) ──────────────────────────────────────
def detect_streaks(img: Image.Image,
                   edge_margin: float = 0.10,
                   min_dist_ratio: float = 0.04) -> dict:
    """이미지에서 streak peak 위치 + 간격 검출.

    img: cropped 또는 grayscale image
    edge_margin: 좌우 가장자리에서 무시할 비율 (기본 10%)
    min_dist_ratio: peak 사이 최소 거리 (분석 창 너비 대비, 기본 4%)
                   → 같은 클러스터의 중복 peak 제거

    Returns dict:
      - peaks:   list of column indices (전체 이미지 좌표)
      - spacing: median gap (px) or None
      - r0, r1, c0, c1: 분석에 사용한 영역
    """
    arr = np.array(img.convert("L"), dtype=np.float32) / 255.0
    h, w = arr.shape

    r0, r1 = int(h * 0.25), int(h * 0.70)
    c0, c1 = int(w * edge_margin), int(w * (1 - edge_margin))

    profile = arr[r0:r1, c0:c1].mean(axis=0)
    win_w = c1 - c0

    k = max(3, win_w // 35)
    profile = np.convolve(profile, np.ones(k) / k, mode="same")

    pmax = profile.max()
    if pmax < 0.03:
        return {"peaks": [], "spacing": None,
                "r0": r0, "r1": r1, "c0": c0, "c1": c1}
    profile = profile / pmax

    # 후보 local maxima (높이 0.15 이상)
    candidates = [(i, float(profile[i])) for i in range(1, win_w - 1)
                  if profile[i] > profile[i - 1]
                  and profile[i] > profile[i + 1]
                  and profile[i] > 0.15]

    # 같은 클러스터 안에서는 가장 밝은 peak 하나만 유지
    # (밝기 내림차순으로 보면서, 이미 선택된 peak 와 min_dist 이상 떨어진 것만 add)
    min_dist = max(3, int(win_w * min_dist_ratio))
    selected = []
    for pos, height in sorted(candidates, key=lambda x: -x[1]):
        if all(abs(pos - p) >= min_dist for p, _ in selected):
            selected.append((pos, height))

    # 위치 순으로 정렬 → 전체 이미지 좌표로 변환
    selected.sort(key=lambda x: x[0])
    peaks = [p + c0 for p, _ in selected]

    spacing = None
    if len(peaks) >= 2:
        gaps = [peaks[j + 1] - peaks[j] for j in range(len(peaks) - 1)]
        if gaps:
            spacing = float(np.median(gaps))

    return {"peaks": peaks, "spacing": spacing,
            "r0": r0, "r1": r1, "c0": c0, "c1": c1}


def detect_streak_spacing_px(img: Image.Image):
    """Backward-compat wrapper. Returns spacing only."""
    cropped = crop_dark_top(img)
    return detect_streaks(cropped)["spacing"]


def annotate_streaks(img: Image.Image, streaks: dict) -> Image.Image:
    """Streak peak 위치를 빨간 세로줄로, 분석 영역을 노란 사각형 윤곽으로 표시."""
    annotated = img.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)
    w, h = annotated.size

    # 분석 영역 윤곽 (노란 사각형)
    r0 = streaks.get("r0")
    r1 = streaks.get("r1")
    c0 = streaks.get("c0", 0)
    c1 = streaks.get("c1", w)
    if r0 is not None and r1 is not None:
        draw.rectangle([(c0, r0), (c1 - 1, r1 - 1)], outline="yellow", width=2)

    # Peak 위치 (빨간 세로줄, 분석 행 범위 안에서만)
    line_w = max(2, w // 200)
    y0 = r0 if r0 is not None else 0
    y1 = (r1 - 1) if r1 is not None else (h - 1)
    for p in streaks.get("peaks", []):
        draw.line([(p, y0), (p, y1)], fill="red", width=line_w)

    return annotated


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

    # ── Lattice constant calibration ─────────────────────────────────────
    st.subheader("📐 Lattice Calibration")

    substrate = st.selectbox(
        "Substrate",
        options=list(SUBSTRATES.keys()),
        index=0,
    )
    a_sub_default = SUBSTRATES[substrate]
    if a_sub_default is None:
        a_sub = st.number_input(
            "Custom a (Å)", min_value=0.1, max_value=20.0,
            value=4.0, step=0.001, format="%.3f")
    else:
        a_sub = a_sub_default
        st.caption(f"Bulk a = **{a_sub} Å**")

    cal_file = st.file_uploader(
        "Bare substrate RHEED",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        key="cal_upload",
        label_visibility="collapsed",
        help="사파이어 substrate 이미지를 업로드해서 calibration",
    )

    if cal_file is not None:
        try:
            cal_img = safe_open_rgb(cal_file)
            cal_cropped = crop_dark_top(cal_img)
            cal_gray = to_grayscale_rgb(cal_cropped)
            cal_streaks = detect_streaks(cal_gray)
            cal_spacing = cal_streaks["spacing"]

            # Annotated thumbnail (peaks visualized)
            cal_annotated = annotate_streaks(cal_gray, cal_streaks)
            n_cal_peaks = len(cal_streaks["peaks"])
            st.image(
                cal_annotated,
                caption=f"Substrate · {n_cal_peaks} peaks detected",
                use_container_width=True,
            )

            if cal_spacing is not None:
                # a × d_px = constant
                st.session_state["cal_const"] = a_sub * cal_spacing
                st.session_state["cal_substrate"] = substrate
                st.session_state["cal_spacing_px"] = cal_spacing
                st.success(
                    f"✓ Calibrated\n\n"
                    f"Spacing: **{cal_spacing:.1f} px** = {a_sub} Å"
                )
            else:
                st.error("Streak detection failed in calibration image")
        except Exception as ex:
            st.error(f"Calibration failed: {ex}")
    elif "cal_const" in st.session_state:
        st.info(
            f"Using stored calibration\n"
            f"({st.session_state.get('cal_substrate', '?')}, "
            f"{st.session_state.get('cal_spacing_px', 0):.1f} px)"
        )

    st.divider()
    st.caption(f"Model: Thomson_5 · TF {tf.__version__}")
    st.caption("© 2026 rlack")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
st.header("Upload RHEED Image(s)")

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

    # Streak detection on the gray image (post-crop, post-autocontrast)
    streaks = detect_streaks(gray_img)
    annotated = annotate_streaks(gray_img, streaks)

    with st.container(border=True):
        c_orig, c_gray, c_res = st.columns([1, 1, 1.4])
        with c_orig:
            st.image(img, caption=f"Original: {f.name}", use_container_width=True)
        with c_gray:
            n_peaks = len(streaks["peaks"])
            cap = f"Detected {n_peaks} peaks" if n_peaks else "No peaks detected"
            st.image(annotated, caption=cap, use_container_width=True)
        with c_res:
            fig = plot_probs(prob)
            st.pyplot(fig)
            plt.close(fig)

            # Streak / Mixed 일 때만 reconstruction & lattice constant 표시
            if cls in ("Streaks", "Mixed"):
                recon = detect_reconstruction(arr[0])
                st.markdown(
                    f'<div style="font-size:14px;color:#555;margin-top:6px;">'
                    f'<b>Surface reconstruction:</b> '
                    f'<span style="font-family:monospace;color:#1e293b;'
                    f'font-weight:600;">{recon}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # Lattice constant (calibration이 있을 때만)
                if "cal_const" in st.session_state:
                    spacing_px = streaks["spacing"]
                    if spacing_px is not None:
                        a_growth = st.session_state["cal_const"] / spacing_px
                        st.markdown(
                            f'<div style="font-size:14px;color:#555;margin-top:4px;">'
                            f'<b>Lattice constant:</b> '
                            f'<span style="font-family:monospace;color:#1e293b;'
                            f'font-weight:600;">{a_growth:.3f} Å</span>'
                            f'<span style="font-size:11px;color:#999;margin-left:6px;">'
                            f'({spacing_px:.1f} px)</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            '<div style="font-size:13px;color:#999;margin-top:4px;">'
                            'Lattice: streak detection failed'
                            '</div>',
                            unsafe_allow_html=True,
                        )
