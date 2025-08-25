import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# ============================
# Load model and preprocessing tools
# ============================
MODEL_PATH = "dna_species_model.h5"
SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "label_encoder.pkl"

# Load trained model
model = load_model(MODEL_PATH)

# Load scaler
scaler = joblib.load(SCALER_PATH)

# Load label encoder (to get original class names)
label_encoder = joblib.load(ENCODER_PATH)
class_labels = list(label_encoder.classes_)

# ============================
# Streamlit UI
# ============================
st.set_page_config(page_title="DNA Species Prediction", layout="centered")

st.title("🧬 DNA Species Prediction App")
st.write("Enter DNA sequence features below to predict the species.")

# Input fields for DNA features
gc_content = st.number_input("GC_Content (%)", min_value=0.0, max_value=100.0, value=50.0)
at_content = st.number_input("AT_Content (%)", min_value=0.0, max_value=100.0, value=50.0)
seq_length = st.number_input("Sequence Length", min_value=1, value=100)
num_a = st.number_input("Num_A", min_value=0, value=25)
num_t = st.number_input("Num_T", min_value=0, value=25)
num_c = st.number_input("Num_C", min_value=0, value=25)
num_g = st.number_input("Num_G", min_value=0, value=25)
kmer_3 = st.number_input("k-mer 3 Freq", min_value=0.0, max_value=1.0, value=0.5)
mutation_flag = st.selectbox("Mutation Flag", [0, 1])

# Convert input to array
features = np.array([[gc_content, at_content, seq_length, num_a, num_t, num_c,
                      num_g, kmer_3, mutation_flag]])

# Scale features
features_scaled = scaler.transform(features)

# ============================
# Prediction
# ============================
if st.button("Predict Species"):
    prediction = model.predict(features_scaled)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_label = class_labels[predicted_class]

    st.success(f"✅ Predicted Species: **{predicted_label}**")
