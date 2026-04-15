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


def preprocess(img: Image.Image) -> np.ndarray:
    """PIL 이미지 → 모델 입력용 (1, 260, 260, 3) float32 배열"""
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
            arr = preprocess(img)
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

    arr = preprocess(img)
    probs = model.predict(arr, verbose=0)[0]
    top_i = int(np.argmax(probs))
    top_cls = CLASS_NAMES[top_i]
    top_conf = probs[top_i]
    top_color = CLASS_COLORS[top_i]

    with st.container(border=True):
        st.markdown(f"### 📄 `{f.name}`")
        col_img, col_res = st.columns([1, 1.3])

        with col_img:
            st.image(img, caption=f"{img.size[0]}×{img.size[1]} px",
                     use_container_width=True)

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
