# 🔬 RHEED Image Classifier

A deep learning web application for classifying **Reflection High-Energy Electron Diffraction (RHEED)** patterns into 4 categories.

**🌐 Live Demo:** [your-app-name.streamlit.app](#) *(update this link after deploy)*

---

## 🎯 Classes

| Class | Description |
|-------|-------------|
| 🔴 **Modulated**       | Periodic intensity modulation along the streaks |
| 🔵 **Anomalous Spots** | Irregular bright spots (transmission-like diffraction) |
| 🟢 **Spotty**          | Discrete diffraction spots (3D island growth) |
| 🟡 **Streaks**         | Continuous streaks (smooth 2D layer growth) |

## 🧠 Model

- **Architecture:** EfficientNetB2 + Transfer Learning + Custom Dense head
- **Input size:** 260 × 260 RGB
- **Training:** Label Smoothing + Mixup + Cosine Annealing + 2-Phase Fine-tuning
- **Test Accuracy:** ~93% on stratified 20% test split
- **File:** `models/Thomson_5.keras` (~61 MB)

## ✨ Features

- 📤 Multi-image upload (drag & drop)
- 🔢 Supports 8-bit and 16-bit grayscale PNG/TIFF
- 📊 Probability bar chart for all 4 classes
- 📥 Batch prediction results as CSV download
- ⚠️ Low-confidence warnings

## 🚀 Run Locally

```bash
# 1. Clone this repo
git clone https://github.com/YOUR_USERNAME/rheed_public.git
cd rheed_public

# 2. Install dependencies (Python 3.10 or 3.11 recommended)
pip install -r requirements.txt

# 3. Run
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## ☁️ Deploy to Streamlit Community Cloud (Free, Always-On)

1. Push this repo to GitHub (public)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click **New app** → select this repo / branch / `app.py`
5. Click **Deploy** → get a public URL like `https://YOUR-APP.streamlit.app`

That's it. The app stays online 24/7 even when your computer is off.

## 📁 Project Structure

```
rheed_public/
├── app.py                  # Main Streamlit app (inference only)
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── .gitignore
├── .streamlit/
│   └── config.toml         # Theme & server config
└── models/
    └── Thomson_5.keras     # Trained model (61 MB)
```

## 📝 License

MIT
