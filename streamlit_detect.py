# Import library yang diperlukan
import streamlit as st
import tensorflow as tf
import numpy as np
import joblib
from PIL import Image


# Judul Aplikasi
st.title("Deteksi Gangguan Listrik")
st.write("Masukkan informasi untuk mendeteksi gangguan")

# Load scaler dan label encoder
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Load model TFLite
interpreter = tf.lite.Interpreter(model_path="detecting-electrical-faults.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Form untuk input
Ia = st.number_input("Nilai Ia")
Ib = st.number_input("Nilai Ib")
Ic = st.number_input("Nilai Ic")
Va = st.number_input("Nilai Va")
Vb = st.number_input("Nilai Vb")
Vc = st.number_input("Nilai Vc")

if st.button("Deteksi Gangguan"):
    # Preprocessing input
    input_data = np.array([[Ia, Ib, Ic, Va, Vb, Vc]])
    input_scaled = scaler.transform(input_data).astype(np.float32)
    
    interpreter.set_tensor(input_details[0]['index'], input_scaled)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])

    predicted_label_index = np.argmax(prediction)
    predicted_probability = prediction[0][predicted_label_index]
    predicted_label = label_encoder.inverse_transform([predicted_label_index])[0]
    
    st.success(f"**Deteksi Gangguan: {predicted_label}**")
    
