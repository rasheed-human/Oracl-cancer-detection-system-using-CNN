import streamlit as st
import joblib
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model

from preprocess import preprocess_image_from_array
from utils import make_gradcam_heatmap, overlay_heatmap

# =========================
# CONFIGURATION
# =========================
MODEL_PATH = 'saved_models/hybrid_model.h5'
SCALER_PATH = 'saved_models/clinical_data_scaler.pkl'
CNN_MODEL_PATH = 'saved_models/cnn_branch_model.h5'

CNN_LAST_CONV_LAYER = 'conv5_block3_out'  # verify layer name
IMAGE_SIZE = (224, 224)

# =========================
# PAGE SETUP
# =========================
st.set_page_config(
    page_title="Oral Cancer Detection",
    layout="wide"
)

st.title("🩺 Oral Cancer Detection using CNN + Clinical Data")

# =========================
# MODEL LOADING (CACHED)
# =========================
@st.cache_resource
def load_all_models():
    hybrid_model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    cnn_branch = load_model(CNN_MODEL_PATH)
    return hybrid_model, scaler, cnn_branch

try:
    hybrid_model, scaler, cnn_branch = load_all_models()
except Exception as e:
    st.error(f"❌ Model loading failed: {e}")
    st.stop()

# =========================
# SIDEBAR – CLINICAL DATA
# =========================
st.sidebar.title("Patient Clinical Data")

age = st.sidebar.slider("Age", 18, 100, 50)

smoking_status = st.sidebar.selectbox(
    "Smoking Status", (0, 1),
    format_func=lambda x: "No" if x == 0 else "Yes"
)

alcohol_use = st.sidebar.selectbox(
    "Alcohol Use", (0, 1),
    format_func=lambda x: "No" if x == 0 else "Yes"
)

user_clinical_data = np.array([[age, smoking_status, alcohol_use]])

# =========================
# IMAGE UPLOAD
# =========================
st.header("Lesion Image Analysis")

uploaded_file = st.file_uploader(
    "Upload an image of the oral lesion",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is None:
    st.info("Please upload an image to continue.")
    st.stop()

image = Image.open(uploaded_file).convert("RGB")
original_img_array = np.array(image)

col1, col2 = st.columns(2)

with col1:
    st.image(original_img_array, caption="Uploaded Image", use_container_width=True)

# =========================
# RUN ANALYSIS
# =========================
if st.button("🔍 Run Analysis"):
    with st.spinner("Analyzing image and clinical data..."):

        processed_img = preprocess_image_from_array(original_img_array)
        img_for_model = np.expand_dims(processed_img, axis=0)

        scaled_clinical_data = scaler.transform(user_clinical_data)

        prediction = hybrid_model.predict(
            [img_for_model, scaled_clinical_data],
            verbose=0
        )

        prob = float(prediction[0][0])

    # =========================
    # RESULTS
    # =========================
    with col2:
        st.subheader("Analysis Results")

        st.metric("Malignancy Probability", f"{prob * 100:.2f}%")
        st.progress(prob)

        if prob >= 0.80:
            st.error("⚠️ High likelihood of Malignancy")
        elif prob <= 0.20:
            st.success("✅ Low likelihood of Malignancy")
        else:
            st.warning("⚖️ Inconclusive – Further examination recommended")

    # =========================
    # GRAD-CAM
    # =========================
    st.subheader("🧠 Explainable AI (Grad-CAM)")

    try:
        heatmap = make_gradcam_heatmap(
            img_for_model,
            cnn_branch,
            CNN_LAST_CONV_LAYER
        )

        original_bgr = cv2.cvtColor(original_img_array, cv2.COLOR_RGB2BGR)
        original_resized = cv2.resize(original_bgr, IMAGE_SIZE)

        overlay_bgr = overlay_heatmap(original_resized, heatmap)
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

        st.image(overlay_rgb, caption="Grad-CAM Heatmap", use_container_width=True)

    except Exception as e:
        st.error(f"Grad-CAM failed: {e}")
