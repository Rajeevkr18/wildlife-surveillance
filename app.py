import sys
import os

# ================================================
# FIX IMPORT PATH FOR STREAMLIT CLOUD
# ================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.join(current_dir, "utils")

if utils_dir not in sys.path:
    sys.path.append(utils_dir)

# ================================================
# IMPORTS
# ================================================
import streamlit as st
import numpy as np
import cv2
from PIL import Image
from inference import predict_animal, predict_fire, detect_poaching
import base64
import time

# ================================================
# PAGE CONFIG
# ================================================
st.set_page_config(
    page_title="Wildlife Surveillance AI",
    layout="wide",
)

# ================================================
# PREMIUM UI THEME (Glassmorphism + Gradient)
# ================================================
st.markdown("""
    <style>

    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
        color: #e2e8f0;
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border-radius: 18px;
        padding: 25px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.4);
        margin-bottom: 20px;
    }

    .main-title {
        text-align: center;
        font-size: 48px;
        font-weight: 800;
        color: #38bdf8;
        text-shadow: 0 0 12px rgba(56, 189, 248, 0.6);
        margin-top: 10px;
    }

    .sub-title {
        text-align: center;
        font-size: 20px;
        color: #cbd5e1;
        margin-top: -12px;
        margin-bottom: 20px;
    }

    .stButton>button {
        background: linear-gradient(135deg, #0ea5e9, #3b82f6);
        color: white;
        padding: 12px 22px;
        border-radius: 12px;
        border: none;
        font-size: 18px;
        font-weight: 600;
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 15px rgba(59,130,246,0.6);
    }

    .result-box {
        background: rgba(59,130,246,0.15);
        padding: 15px;
        border-radius: 12px;
        text-align: center;
        font-size: 22px;
        color: #f8fafc;
        border: 1px solid rgba(59,130,246,0.4);
        margin-top: 15px;
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.1);
        padding: 12px 20px;
        border-radius: 8px;
        margin-right: 10px;
        color: #e2e8f0 !important;
        font-size: 18px !important;
        font-weight: 600 !important;
        transition: 0.2s;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255,255,255,0.25);
    }

    </style>
""", unsafe_allow_html=True)

# ================================================
# HEADER
# ================================================
st.markdown("<h1 class='main-title'>🦁 Wildlife Surveillance AI System</h1>", unsafe_allow_html=True)
st.markdown("<h4 class='sub-title'>Animal Detection • Poaching Alerts • Fire Monitoring • Real-Time Vision</h4>", unsafe_allow_html=True)

# ================================================
# SIDEBAR
# ================================================
st.sidebar.header("🌿 Navigation")
mode = st.sidebar.radio(
    "Choose Mode:",
    ["Animal Classification", "Poaching Detection", "Fire Detection", "Live Webcam"]
)

uploaded_file = st.sidebar.file_uploader("📤 Upload Image", type=["png", "jpg", "jpeg"])

# ================================================
# IMAGE MODES
# ================================================
if mode != "Live Webcam":

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    # LEFT — IMAGE
    with col1:
        if uploaded_file:
            pil_image = Image.open(uploaded_file).convert("RGB")
            image = np.array(pil_image)
            st.image(image, caption="📷 Uploaded Image", use_container_width=True)
        else:
            st.warning("⬅ Please upload an image.")

    # RIGHT — ANALYSIS
    with col2:
        if uploaded_file and st.button("🚀 Run Analysis"):
            with st.spinner("Processing... Please wait..."):
                time.sleep(1)

                tab1, tab2, tab3 = st.tabs(
                    ["🦌 Animal Classification", "🎯 Poaching Detection", "🔥 Fire Detection"]
                )

                # ANIMAL CLASSIFICATION
                with tab1:
                    label, conf = predict_animal(image)
                    st.markdown(
                        f"<div class='result-box'>🐾 Species: <b>{label.title()}</b></div>",
                        unsafe_allow_html=True
                    )
                    st.info(f"Confidence: {conf*100:.2f}%")

                # POACHING
                with tab2:
                    result_img, boxes = detect_poaching(image)
                    st.image(result_img, use_container_width=True)
                    st.markdown(
                        f"<div class='result-box'>Detected Objects: <b>{len(boxes)}</b></div>",
                        unsafe_allow_html=True
                    )

                # FIRE
                with tab3:
                    fire_label, fire_conf = predict_fire(image)
                    if "Fire" in fire_label:
                        st.error(f"🔥 FIRE DETECTED ({fire_conf*100:.2f}%)")
                    else:
                        st.success(f"🌿 No Fire ({fire_conf*100:.2f}%)")

    st.markdown("</div>", unsafe_allow_html=True)

# ================================================
# LIVE WEBCAM MODE
# ================================================
if mode == "Live Webcam":

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("🎥 Live Webcam Detection")

    start = st.button("▶ Start Webcam")
    stop = st.button("⏹ Stop Webcam")
    FRAME_WINDOW = st.image([])

    if "cam_active" not in st.session_state:
        st.session_state.cam_active = False

    if start:
        st.session_state.cam_active = True
    if stop:
        st.session_state.cam_active = False
        st.warning("Webcam stopped.")

    if st.session_state.cam_active:
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("❌ Unable to access webcam.")
            st.session_state.cam_active = False

        while st.session_state.cam_active:
            ret, frame = cap.read()
            if not ret:
                st.error("Could not read frame.")
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            label, conf = predict_animal(rgb)

            cv2.putText(
                rgb,
                f"{label.title()} ({conf*100:.1f}%)",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (56, 189, 248),
                2
            )

            FRAME_WINDOW.image(rgb)

        cap.release()

    st.markdown("</div>", unsafe_allow_html=True)
