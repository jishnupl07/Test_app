import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks
import tempfile

# Import your new blood pressure logic
import blood_pressure

# --- Model Loading ---
# Cache the BP model so it only loads once
@st.cache_resource
def load_bp_model():
    model = blood_pressure.load_model()
    return model

# Load the model
try:
    bp_model = load_bp_model()
    st.sidebar.success("BP Model Loaded Successfully")
except Exception as e:
    st.sidebar.error(f"Error loading BP model: {e}")
    st.sidebar.error("Please ensure 'bp_model_lstm.pth' is in your GitHub repo.")
    bp_model = None

# --- Core BPM/PPG Functions (Your Original) ---
def extract_red_intensity(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error: Could not open video file.")
        return None, 0, 0
        
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 30 # Default fallback
    
    red_intensities = []
    
    progress_bar = st.progress(0, text="Analyzing video for BPM plot...")
    
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        red_channel = frame[:, :, 2] 
        avg_red_intensity = np.mean(red_channel)
        red_intensities.append(avg_red_intensity)
        
        progress_bar.progress((i + 1) / frame_count, text=f"Analyzing frame {i+1}/{frame_count}")
            
    cap.release()
    progress_bar.empty()
    return red_intensities, frame_count, fps

def normalize_intensities(intensities):
    min_intensity = np.min(intensities)
    max_intensity = np.max(intensities)
    normalized_intensities = (intensities - min_intensity) / (max_intensity - min_intensity)
    return normalized_intensities

def find_peaks_in_intensity(intensities, distance=15, prominence=0.05):
    peaks, properties = find_peaks(intensities, distance=distance, height=None, threshold=None, prominence=prominence)
    return peaks, properties

def calculate_bpm(peaks, duration_in_seconds):
    if duration_in_seconds == 0:
        return 0
    num_beats = len(peaks)
    bpm = (num_beats / duration_in_seconds) * 60
    return bpm

def create_bpm_plot(intensities, peaks, video_name):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(intensities, label='Normalized Red Intensity')
    ax.plot(peaks, np.array(intensities)[peaks], "x", label=f'Detected Peaks ({len(peaks)})')
    ax.set_title(f'Red Intensity and Peaks for {video_name}')
    ax.set_xlabel('Frame Number')
    ax.set_ylabel('Normalized Average Red Intensity')
    ax.legend()
    return fig

def signal_quality_check(intensities, peaks, properties, fps):
    if len(peaks) == 0:
        return False, 0
        
    signal_power = np.mean(np.array(intensities)[peaks] ** 2)
    noise_power = np.var(intensities)
    snr = signal_power / (noise_power + 1e-10) 
    
    if fps == 0: fps = 30 # Avoid division by zero
    min_peaks = (len(intensities) / fps) * (30 / 60) # 30 BPM
    if len(peaks) < min_peaks: 
        st.warning(f"Low peak count ({len(peaks)}). Result may be inaccurate.")
        return False, snr
    
    if snr < 2: 
        return False, snr
    
    return True, snr

# --- Streamlit App Interface ---

st.set_page_config(layout="wide", page_title="Health Monitor (BPM & BP)")
st.title("Health Monitor: BPM & Blood Pressure Calculator")
st.write("Upload a video file (e.g., from a finger-tip recording) to calculate BPM and Blood Pressure.")

uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi", "mkv"])

if uploaded_file is not None and bp_model is not None:
    video_name = uploaded_file.name
    
    # Save to a temporary file path
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video_name)[1]) as tmpfile:
        tmpfile.write(uploaded_file.getvalue())
        video_path = tmpfile.name

    st.write(f"Processing video: `{video_name}`")
    
    # --- 1. BPM Calculation (for Plot) ---
    with st.spinner("Analyzing signal for BPM plot..."):
        intensities, frame_count, fps = extract_red_intensity(video_path)
    
    if intensities:
        with st.spinner("Finding peaks for BPM..."):
            normalized_intensities = normalize_intensities(intensities)
            peaks, properties = find_peaks_in_intensity(normalized_intensities)
            
            is_valid_signal, snr = signal_quality_check(normalized_intensities, peaks, properties, fps)
            
            if not is_valid_signal:
                st.error(f"The video '{video_name}' has poor signal quality (SNR: {snr:.2f}). Please use a better video with a clearer signal.")
            else:
                duration_in_seconds = frame_count / fps
                bpm = calculate_bpm(peaks, duration_in_seconds)
                
                # --- 2. Blood Pressure Calculation ---
                try:
                    with st.spinner("Calculating Blood Pressure..."):
                        # Call the function from your blood_pressure.py
                        # It uses the video_path to do its own processing
                        systolic, diastolic = blood_pressure.predict_bp(bp_model, video_path)
                    
                    st.success("Processing Complete!")
                    
                    # --- 3. Display All Results ---
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Results")
                        st.metric(label="Calculated BPM", value=f"{bpm:.2f}")
                        st.metric(label="Predicted Systolic BP", value=f"{systolic:.0f}")
                        st.metric(label="Predicted Diastolic BP", value=f"{diastolic:.0f}")
                        
                        st.subheader("Video Info")
                        st.metric(label="Video Length", value=f"{duration_in_seconds:.2f} s")
                        st.metric(label="Video FPS", value=f"{fps:.2f}")
                        st.metric(label="Signal-to-Noise Ratio (SNR)", value=f"{snr:.2f}")
                    
                    with col2:
                        st.subheader("Signal Plot (for BPM)")
                        plot = create_bpm_plot(normalized_intensities, peaks, video_name)
                        st.pyplot(plot)

                except Exception as e:
                    st.error(f"An error occurred during Blood Pressure calculation: {e}")
                    st.exception(e) # Show full error details

    # Clean up the temporary file
    os.remove(video_path)

elif bp_model is None:
    st.error("The Blood Pressure model could not be loaded. The app cannot proceed.")