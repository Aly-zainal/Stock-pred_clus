import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Muat model DBSCAN dan scaler
try:
    dbsc = joblib.load('dbscan_model.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    st.error("Model 'dbscan_model.pkl' atau scaler 'scaler.pkl' tidak ditemukan. Pastikan Anda sudah menjalankan sel sebelumnya untuk melatih dan menyimpan model.")
    st.stop()

st.title('Prediksi Klaster Harga Saham dengan DBSCAN')
st.write('Aplikasi ini memprediksi klaster untuk nilai Daily Returns dan Volatility yang diberikan menggunakan model DBSCAN.')

st.header('Masukkan Fitur Saham')
daily_returns = st.number_input('Daily Returns (contoh: 0.005 untuk 0.5%)', value=0.001, format="%.6f")
volatility = st.number_input('Volatility (contoh: 0.015)', value=0.01, format="%.6f")

if st.button('Prediksi Klaster'):
    # Buat DataFrame dari input pengguna
    input_data = pd.DataFrame([[daily_returns, volatility]], columns=['Daily Returns', 'Volatility'])

    # Skala input data menggunakan scaler yang sudah dilatih
    input_scaled = scaler.transform(input_data)

    # Prediksi klaster
    predicted_cluster = dbsc.fit_predict(input_scaled)

    st.subheader('Hasil Prediksi')
    if predicted_cluster[0] == -1:
        st.warning('Titik data ini diklasifikasikan sebagai *noise* (bukan bagian dari klaster manapun).')
    else:
        st.success(f'Titik data ini termasuk dalam Klaster: {predicted_cluster[0]}')

    st.write("\n--- Untuk menjalankan aplikasi Streamlit ini:\n1. Simpan kode di atas sebagai `app.py` di lingkungan lokal Anda.\n2. Buka terminal di direktori yang sama.\n3. Jalankan perintah: `streamlit run app.py`")
