import streamlit as st
import numpy as np
import cv2
import os
import gdown
from tensorflow import keras
from tensorflow.keras.applications.resnet import preprocess_input
from PIL import Image

# Constantes
MODEL_FILENAME = "best_resnet_model.h5"
GDRIVE_ID = "1LdaSNQbdHSLJdEpB0Jogqwk0Y2g98W5e"
IMG_SIZE = (224, 224)
CLASS_NAMES = ['Healthy', 'Tumor']

# Descargar y cargar modelo
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_FILENAME):
        url = f"https://drive.google.com/uc?id={GDRIVE_ID}"
        gdown.download(url, MODEL_FILENAME, quiet=False)
    model = keras.models.load_model(MODEL_FILENAME)
    return model

# App Streamlit
st.title("🧠 Clasificador de Tumores Cerebrales")

uploaded_file = st.file_uploader("📤 Sube una imagen", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Mostrar imagen
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen cargada", use_column_width=True)

    if st.button("🔍 Predecir"):
        # Preprocesar
        img = image.resize(IMG_SIZE)
        img_array = np.array(img)
        img_array_expanded = np.expand_dims(img_array, axis=0)
        img_preprocessed = preprocess_input(img_array_expanded)

        # Predecir
        model = load_model()
        prediction_probs = model.predict(img_preprocessed)
        prob_tumor = prediction_probs[0][0]

        # Interpretar resultado
        if prob_tumor >= 0.5:
            predicted_class = CLASS_NAMES[1]
            confidence = prob_tumor * 100
        else:
            predicted_class = CLASS_NAMES[0]
            confidence = (1 - prob_tumor) * 100

        # Mostrar resultado
        st.subheader("🧾 Resultado")
        st.write(f"**Predicción:** {predicted_class}")
        st.write(f"**Confianza:** {confidence:.2f}%")
