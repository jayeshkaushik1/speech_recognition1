import streamlit as st
import os
import requests
import numpy as np
import librosa
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import LSTM


# --- 1. CUSTOM LAYER FIX ---
# This handles the "time_major" error by stripping the bad argument
class FixedLSTM(LSTM):
    def __init__(self, **kwargs):
        if 'time_major' in kwargs:
            kwargs.pop('time_major')
        super().__init__(**kwargs)

# --- 2. CONFIG ---
FRAME_LENGTH = 2048
HOP_LENGTH = 512
TARGET_LENGTH = 180000
EMOTION_LABELS = ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust']

# --- 3. DIRECT DOWNLOAD & LOAD ---
@st.cache_resource
def download_and_load_models():
    # *** REPLACE WITH YOUR REAL LINKS ***
    female_url = "https://github.com/YOUR_USERNAME/YOUR_REPO/releases/download/v1.0/model_female.h5"
    male_url = "https://github.com/YOUR_USERNAME/YOUR_REPO/releases/download/v1.0/model_male.h5"

    status_text = st.empty()
    
    # Download checks
    if not os.path.exists("model_female.h5"):
        status_text.info("⏳ Downloading Female Model...")
        try:
            r = requests.get(female_url)
            r.raise_for_status()
            with open("model_female.h5", "wb") as f: f.write(r.content)
        except Exception as e:
            st.error(f"Failed to download Female model: {e}")
            return None, None

    if not os.path.exists("model_male.h5"):
        status_text.info("⏳ Downloading Male Model...")
        try:
            r = requests.get(male_url)
            r.raise_for_status()
            with open("model_male.h5", "wb") as f: f.write(r.content)
        except Exception as e:
            st.error(f"Failed to download Male model: {e}")
            return None, None

    # Load with Custom Object
    status_text.info("⚙️ Loading TensorFlow models...")
    try:
        # We register FixedLSTM as "LSTM" so Keras uses it automatically
        custom_objects = {'LSTM': FixedLSTM}
        
        model_f = tf.keras.models.load_model("model_female.h5", custom_objects=custom_objects, compile=False)
        model_m = tf.keras.models.load_model("model_male.h5", custom_objects=custom_objects, compile=False)
        
        status_text.empty()
        return model_f, model_m
    except Exception as e:
        st.error(f"💥 Error loading models: {e}")
        return None, None

# --- 4. PREPROCESSING ---
def process_audio(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    trimmed, _ = librosa.effects.trim(y, top_db=25)
    
    if len(trimmed) > TARGET_LENGTH:
        padded = trimmed[:TARGET_LENGTH]
    else:
        padding = TARGET_LENGTH - len(trimmed)
        padded = np.pad(trimmed, (0, padding), 'constant')
    
    zcr = librosa.feature.zero_crossing_rate(padded, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
    rms = librosa.feature.rms(y=padded, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
    mfccs = librosa.feature.mfcc(y=padded, sr=sr, n_mfcc=13, hop_length=HOP_LENGTH)
    
    features = np.concatenate((zcr.T, rms.T, mfccs.T), axis=1)
    return np.expand_dims(features, axis=0).astype('float32')

# --- 5. UI ---
st.title("🎙️ Speech Emotion Recognition")

model_female, model_male = download_and_load_models()

if model_female:
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        audio = st.file_uploader("Upload .wav", type=['wav'])
    
    with col2:
        if audio and st.button("Analyze"):
            with open("temp.wav", "wb") as f: f.write(audio.getbuffer())
            try:
                feats = process_audio("temp.wav")
                model = model_female if gender == 'Female' else model_male
                preds = model.predict(feats)
                idx = np.argmax(preds)
                st.success(f"Emotion: **{EMOTION_LABELS[idx].upper()}**")
                st.bar_chart(pd.DataFrame(preds[0], index=EMOTION_LABELS, columns=["Probability"]))
            except Exception as e:
                st.error(f"Analysis error: {e}")
