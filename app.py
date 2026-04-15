"""
RHEED Image Classifier — Public Web App
Thomson_5 모델 기반 4-class 분류기
(Modulated / Anomalous Spots / Spotty / Streaks)
"""
import os
import io
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# ─── 페이지 설정 ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RHEED Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── 상수 ───────────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "Thomson_5.keras")
CLASS_NAMES = ["Modulated", "Anomalous Spots", "Spotty", "Streaks"]
CLASS_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
CLASS_DESCRIPTIONS = {
    "Modulated":       "Periodic intensity modulation along the streaks",
    "Anomalous Spots": "Irregular bright spots (transmission-like diffraction)",
    "Spotty":          "Discrete diffraction spots (3D island growth)",
    "Streaks":         "Continuous streaks (smooth 2D layer growth)",
}
IMG_SIZE = (260, 260)

# ─── 이미지 전처리 (16-bit PNG 지원) ────────────────────────────────────────────
def safe_open_rgb(file_or_path) -> Image.Image:
    """16-bit 흑백 PNG / RGBA / L 모드 모두 처리해서 RGB 반환"""
    img = Image.open(file_or_path)
    if img.mode in ("I", "I;16", "I;16B"):
        arr = np.array(img, dtype=np.float32)
        mn, mx = arr.min(), arr.max()
        if mx > mn:
            arr = (arr - mn) / (mx - mn) * 255.0
        img = Image.fromarray(arr.astype(np.uint8)).convert("RGB")
    elif img.mode == "L":
        img = img.convert("RGB")
    elif img.mode == "RGBA":
        img = img.convert("RGB")
    elif img.mode != "RGB":
        img = img.convert("RGB")
    return img


def find_shadow_edge_row(img: Image.Image) -> int:
    """RHEED 특화: Shadow edge(검은 가로 띠) 위치를 찾음.
    이미지를 가로줄마다 평균 밝기로 스캔해서 가장 어두운 띠를 찾음.
    RHEED 패턴은 항상 이 shadow edge의 아래쪽에 있음."""
    arr = np.array(img.convert("RGB"))
    bright = np.max(arr, axis=2).astype(np.float32)
    row_means = bright.mean(axis=1)  # 각 가로줄의 평균 밝기

    h = len(row_means)
    # 너무 작은 이미지 예외처리
    if h < 30:
        return 0

    # 이동평균으로 스무딩 (단일 픽셀 노이즈 무시)
    window = max(5, h // 50)
    kernel = np.ones(window, dtype=np.float32) / window
    smoothed = np.convolve(row_means, kernel, mode="same")

    # 이미지 상단 20% ~ 70% 구간에서 가장 어두운 가로줄이 shadow edge
    # (맨 위 20%는 테두리일 가능성 ↑, 70% 이하는 RHEED 본체일 가능성 ↑)
    s_start = int(h * 0.20)
    s_end = int(h * 0.70)
    if s_end <= s_start + 1:
        return h // 3

    local = smoothed[s_start:s_end]
    edge_idx = int(np.argmin(local)) + s_start
    return edge_idx


def auto_crop_bright_region(img: Image.Image, padding: int = 10,
                             threshold_percentile: float = 5,
                             use_shadow_edge: bool = True) -> Image.Image:
    """이미지에서 RHEED 패턴만 자동 크롭.
    use_shadow_edge=True면: 먼저 shadow edge를 찾아 그 아래만 사용 → 위쪽 노이즈/깨진 phosphor 제거.
    그 다음 밝기 기준 crop."""
    arr = np.array(img.convert("RGB"))
    h, w = arr.shape[:2]

    # 1단계: Shadow edge 아래만 선택 (위쪽 노이즈 제거)
    if use_shadow_edge:
        edge = find_shadow_edge_row(img)
        # Edge + 약간의 여유(window만큼 아래)부터 시작
        safety = max(5, h // 60)
        edge = min(edge + safety, h - 10)
        arr = arr[edge:, :]  # 아래쪽만
        if arr.size == 0 or arr.shape[0] < 20:
            # 예외: edge 감지 실패 시 아래 절반 사용
            arr = np.array(img.convert("RGB"))[h // 2:, :]

    # 2단계: 남은 영역에서 밝은 부분만 crop (검은 테두리 제거)
    bright = np.max(arr, axis=2).astype(np.float32)
    threshold = max(np.percentile(bright, 100 - threshold_percentile) * 0.1, 10)
    mask = bright > threshold

    rows_any = np.any(mask, axis=1)
    cols_any = np.any(mask, axis=0)
    if not rows_any.any() or not cols_any.any():
        return Image.fromarray(arr.astype(np.uint8))

    r_min, r_max = np.where(rows_any)[0][[0, -1]]
    c_min, c_max = np.where(cols_any)[0][[0, -1]]

    # 패딩 추가
    hh, ww = arr.shape[:2]
    r_min = max(0, r_min - padding)
    r_max = min(hh - 1, r_max + padding)
    c_min = max(0, c_min - padding)
    c_max = min(ww - 1, c_max + padding)

    cropped = arr[r_min:r_max + 1, c_min:c_max + 1]
    return Image.fromarray(cropped.astype(np.uint8))


def manual_bottom_crop(img: Image.Image, bottom_fraction: float = 0.5) -> Image.Image:
    """이미지의 아래쪽 N% 만 잘라서 반환. bottom_fraction=0.5 → 아래 절반."""
    arr = np.array(img.convert("RGB"))
    h = arr.shape[0]
    start = int(h * (1 - bottom_fraction))
    return Image.fromarray(arr[start:, :])


def to_grayscale_rgb(img: Image.Image) -> Image.Image:
    """컬러 이미지를 '학습 분포와 같은' 그레이스케일 RGB로 변환.
    초록 phosphor RHEED 이미지를 R=G=B로 만들어 도메인 시프트 완화."""
    arr = np.array(img.convert("RGB")).astype(np.float32)
    # max channel (phosphor green에서 가장 강함) — 또는 luminance
    gray = np.max(arr, axis=2)
    # 0-255로 재정규화
    mn, mx = gray.min(), gray.max()
    if mx > mn:
        gray = (gray - mn) / (mx - mn) * 255.0
    # R=G=B
    rgb = np.stack([gray, gray, gray], axis=-1).astype(np.uint8)
    return Image.fromarray(rgb)


def apply_crop(img: Image.Image, crop_mode: str, manual_fraction: float = 0.55) -> Image.Image:
    """크롭 모드에 따라 이미지 자름"""
    if crop_mode == "Smart (Shadow Edge)":
        return auto_crop_bright_region(img, use_shadow_edge=True)
    elif crop_mode == "Auto (밝은 영역)":
        return auto_crop_bright_region(img, use_shadow_edge=False)
    elif crop_mode == "Manual (아래쪽 %)":
        return manual_bottom_crop(img, bottom_fraction=manual_fraction)
    else:
        return img


def preprocess(img: Image.Image, crop_mode: str = "None",
               manual_fraction: float = 0.55,
               grayscale: bool = False) -> np.ndarray:
    """PIL 이미지 → 모델 입력용 (1, 260, 260, 3) float32 배열"""
    img = apply_crop(img, crop_mode, manual_fraction)
    if grayscale:
        img = to_grayscale_rgb(img)
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0  # 모델 내부 Rescaling(255)이 다시 [0,255]로
    return arr[np.newaxis]


# ─── 모델 로드 (캐시) ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🔬 모델 로딩 중... (최초 1회만)")
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return keras.models.load_model(MODEL_PATH)


# ─── 예측 플롯 ─────────────────────────────────────────────────────────────────
def plot_probabilities(probs, class_names, colors):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.barh(class_names, probs * 100, color=colors)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Probability (%)")
    ax.invert_yaxis()
    for i, p in enumerate(probs):
        ax.text(p * 100 + 1, i, f"{p*100:.2f}%", va="center", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════════════════════
st.title("🔬 RHEED Image Classifier")
st.caption(
    "Reflection High-Energy Electron Diffraction 패턴을 4개 클래스로 분류합니다. "
    "*Powered by EfficientNet-based deep learning (Thomson_5 model).*"
)

# ─── 사이드바: 모델 정보 + 사용법 ───────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown(
        """
**Model:** Thomson_5 (EfficientNetB2 + Transfer Learning)
**Input size:** 260 × 260 RGB
**Classes:** 4
- 🔴 **Modulated** — periodic intensity modulation
- 🔵 **Anomalous Spots** — irregular transmission spots
- 🟢 **Spotty** — discrete 3D diffraction spots
- 🟡 **Streaks** — smooth 2D streaks

---

**How to use**
1. Upload RHEED image(s) below
2. Model predicts class + confidence
3. See probability bars for all 4 classes

**Supported formats:**
PNG / JPG / BMP / TIFF (8-bit 또는 16-bit grayscale OK)
        """
    )
    st.divider()
    st.subheader("⚙️ Preprocessing")
    st.caption("실험실 이미지(초록/검정 배경/깨진 phosphor)의 경우 아래 옵션 사용")

    crop_mode = st.radio(
        "📐 크롭 모드",
        options=["None", "Smart (Shadow Edge)", "Auto (밝은 영역)", "Manual (아래쪽 %)"],
        index=1,
        help=(
            "**None**: 크롭 없음 — 학습 데이터와 같은 형식일 때\n\n"
            "**Smart (Shadow Edge)**: RHEED 전용. 가로 검은 띠 아래만 사용 → "
            "위쪽 깨진 phosphor / 긁힘 자동 제거 ⭐ 추천\n\n"
            "**Auto (밝은 영역)**: 밝은 픽셀이 있는 bounding box만. "
            "단순 검정 테두리만 있을 때\n\n"
            "**Manual (아래쪽 %)**: 이미지 아래쪽 N%만 강제 사용 (확실한 방법)"
        ),
    )

    manual_fraction = 0.55
    if crop_mode == "Manual (아래쪽 %)":
        manual_fraction = st.slider(
            "아래쪽 비율",
            min_value=0.30, max_value=0.90, value=0.55, step=0.05,
            help="0.55 = 이미지 아래 55%만 사용",
        )

    grayscale = st.checkbox(
        "⚫ 그레이스케일 변환 (초록 → 흑백)",
        value=True,
        help="초록 phosphor 이미지를 학습 분포와 같은 R=G=B 흑백으로 변환.",
    )
    show_preprocessed = st.checkbox(
        "🔍 전처리 결과 미리보기 표시",
        value=True,
        help="모델에 실제 들어가는 이미지를 확인",
    )

    # 내부용 플래그 (코드 호환성)
    auto_crop = crop_mode != "None"

    st.divider()
    st.caption(f"TensorFlow: {tf.__version__}")
    st.caption("Developed by rlack · 2026")

# ─── 메인: 파일 업로더 ─────────────────────────────────────────────────────────
model = load_model()
if model is None:
    st.error(f"❌ 모델 파일을 찾을 수 없습니다: `{MODEL_PATH}`")
    st.stop()

uploaded_files = st.file_uploader(
    "RHEED 이미지를 업로드하세요 (여러 장 한꺼번에 OK)",
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
    accept_multiple_files=True,
    help="드래그&드롭 또는 클릭해서 선택. 여러 장 한꺼번에 업로드 가능합니다.",
)

if not uploaded_files:
    st.info("👆 위에서 이미지를 업로드해주세요.")
    st.stop()

# ─── 일괄 예측 ─────────────────────────────────────────────────────────────────
st.divider()
st.subheader(f"📊 예측 결과 ({len(uploaded_files)}장)")

# 여러 장이면 먼저 요약 테이블
if len(uploaded_files) > 1:
    summary_rows = []
    for f in uploaded_files:
        try:
            img = safe_open_rgb(f)
            arr = preprocess(img, crop_mode=crop_mode,
                             manual_fraction=manual_fraction,
                             grayscale=grayscale)
            probs = model.predict(arr, verbose=0)[0]
            top_i = int(np.argmax(probs))
            summary_rows.append({
                "파일": f.name,
                "예측": CLASS_NAMES[top_i],
                "신뢰도": f"{probs[top_i]*100:.2f}%",
                **{cn: f"{p*100:.1f}%" for cn, p in zip(CLASS_NAMES, probs)},
            })
            f.seek(0)  # 다시 읽을 수 있게 포인터 리셋
        except Exception as e:
            summary_rows.append({
                "파일": f.name, "예측": "오류", "신뢰도": "-",
                **{cn: "-" for cn in CLASS_NAMES},
            })
    df = pd.DataFrame(summary_rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # 다운로드 버튼
    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "📥 결과를 CSV로 다운로드",
        data=csv,
        file_name="rheed_predictions.csv",
        mime="text/csv",
    )
    st.divider()

# 각 파일 상세
for f in uploaded_files:
    try:
        img = safe_open_rgb(f)
    except Exception as e:
        st.error(f"❌ `{f.name}`: 이미지를 열 수 없습니다 ({e})")
        continue

    # 전처리된 이미지도 만들기 (미리보기 + 예측용)
    preprocessed_img = apply_crop(img, crop_mode, manual_fraction)
    if grayscale:
        preprocessed_img = to_grayscale_rgb(preprocessed_img)

    arr = preprocess(img, crop_mode=crop_mode,
                     manual_fraction=manual_fraction,
                     grayscale=grayscale)
    probs = model.predict(arr, verbose=0)[0]
    top_i = int(np.argmax(probs))
    top_cls = CLASS_NAMES[top_i]
    top_conf = probs[top_i]
    top_color = CLASS_COLORS[top_i]

    with st.container(border=True):
        st.markdown(f"### 📄 `{f.name}`")
        col_img, col_res = st.columns([1, 1.3])

        with col_img:
            st.image(img, caption=f"원본: {img.size[0]}×{img.size[1]} px",
                     use_container_width=True)
            if show_preprocessed and (auto_crop or grayscale):
                st.image(
                    preprocessed_img,
                    caption=f"전처리 후: {preprocessed_img.size[0]}×{preprocessed_img.size[1]} px "
                            f"→ 260×260으로 리사이즈되어 모델 입력",
                    use_container_width=True,
                )

        with col_res:
            # 큼직한 Top-1 결과
            st.markdown(
                f"""
                <div style="padding:16px;border-radius:10px;
                            background:{top_color}22;border-left:6px solid {top_color};
                            margin-bottom:12px;">
                    <div style="font-size:13px;color:#666;">Predicted class</div>
                    <div style="font-size:28px;font-weight:700;color:{top_color};">
                        {top_cls}
                    </div>
                    <div style="font-size:15px;color:#333;margin-top:4px;">
                        Confidence: <b>{top_conf*100:.2f}%</b>
                    </div>
                    <div style="font-size:12px;color:#666;margin-top:6px;">
                        {CLASS_DESCRIPTIONS[top_cls]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # 확률 막대
            fig = plot_probabilities(probs, CLASS_NAMES, CLASS_COLORS)
            st.pyplot(fig)
            plt.close(fig)

            # 낮은 신뢰도 경고
            if top_conf < 0.6:
                st.warning(
                    f"⚠️ 낮은 신뢰도 ({top_conf*100:.1f}%). "
                    "이미지가 모호하거나 학습 분포와 다를 수 있습니다."
                )
