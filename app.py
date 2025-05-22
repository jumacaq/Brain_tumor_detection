import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.resnet import preprocess_input
import tempfile
import requests
from PIL import Image

# --- CONFIGURACIÓN ---
CLASS_NAMES = ['Healthy', 'Tumor']
IMG_SIZE = (224, 224)

# URL pública de Google Drive (asegúrate que sea pública y directa)
MODEL_URL = "https://drive.google.com/uc?export=download&id=1LdaSNQbdHSLJdEpB0Jogqwk0Y2g98W5e"
MODEL_FILENAME = "best_resnet_model.h5"

@st.cache_resource
def load_model():
    response = requests.get(MODEL_URL)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name
    model = keras.models.load_model(tmp_path)
    return model

model = load_model()

st.title("Clasificación de Tumores Cerebrales")
st.write("Sube una imagen de resonancia magnética para predecir si es saludable o muestra un tumor.")

uploaded_file = st.file_uploader("Sube una imagen", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen subida", use_column_width=True)

    if st.button("Realizar Predicción"):
        img_array = np.array(image)
        img_resized = cv2.resize(img_array, IMG_SIZE)
        img_array_expanded = np.expand_dims(img_resized, axis=0)
        img_preprocessed = preprocess_input(img_array_expanded)

        prediction_probs = model.predict(img_preprocessed)
        prob_tumor = prediction_probs[0][0]

        if prob_tumor >= 0.5:
            predicted_class = "Tumor"
            confidence = prob_tumor * 100
        else:
            predicted_class = "Healthy"
            confidence = (1 - prob_tumor) * 100

        st.success(f"Predicción: **{predicted_class}** con {confidence:.2f}% de confianza.")
        st.progress(int(confidence))
