import streamlit as st
import numpy as np
import cv2
import os
import gdown
from tensorflow import keras
from tensorflow.keras.applications.resnet import preprocess_input
from PIL import Image

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Deep Learning for Brain Tumor MRI",
    layout="wide",
)

# --- CONSTANTES ---
MODEL_FILENAME = "best_resnet_model.h5"
GDRIVE_ID = "1LdaSNQbdHSLJdEpB0Jogqwk0Y2g98W5e"
IMG_SIZE = (224, 224)
CLASS_NAMES = ['Healthy', 'Tumor']

# --- FUNCIONES ---
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_FILENAME):
        url = f"https://drive.google.com/uc?id={GDRIVE_ID}"
        gdown.download(url, MODEL_FILENAME, quiet=False)
    model = keras.models.load_model(MODEL_FILENAME)
    return model

def preprocess_image(image: Image.Image):
    img = image.resize(IMG_SIZE)
    img_array = np.array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded)

# --- CARGA DEL MODELO UNA VEZ ---
model = load_model()

# --- TÍTULO ---
st.markdown(
    "<h1 style='text-align: center;'>🧠 Deep Learning for Brain Tumor</h1><br>",
    unsafe_allow_html=True
)

# --- DISEÑO EN 3 COLUMNAS ---
col1, col_mid, col2 = st.columns([1, 0.1, 1])

with col1:
    uploaded_file = st.file_uploader("📤 Sube una imagen de resonancia magnética", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        predict_btn = st.button("🔍 Predecir")

with col2:
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_column_width=True)
        if predict_btn:
            img_preprocessed = preprocess_image(image)
            prediction_probs = model.predict(img_preprocessed)
            prob_tumor = prediction_probs[0][0]

            if prob_tumor >= 0.5:
                predicted_class = CLASS_NAMES[1]
                confidence = prob_tumor * 100
            else:
                predicted_class = CLASS_NAMES[0]
                confidence = (1 - prob_tumor) * 100

            # Mostrar resultados justo debajo de la imagen
            st.markdown("#### 🧾 Resultado del diagnóstico")
            st.markdown(f"**Predicción:** {predicted_class}")
            st.markdown(f"**Confianza:** {confidence:.2f}%")

# --- ESTILO PARA IMAGEN ---
st.markdown("""
<style>
img {
    max-height: 300px !important;
    object-fit: contain;
}
</style>
""", unsafe_allow_html=True)
