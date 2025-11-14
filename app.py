import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks
import tempfile

# --- Core Processing Functions (Your Original Logic) ---
# These functions are your original logic, unchanged except for UI parts.

def extract_red_intensity(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error: Could not open video file.")
        return None, 0, 0
        
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    red_intensities = []
    
    # Add a progress bar for long videos
    progress_bar = st.progress(0, text="Analyzing video...")
    
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        # Your original logic: use the red channel
        red_channel = frame[:, :, 2] 
        avg_red_intensity = np.mean(red_channel)
        red_intensities.append(avg_red_intensity)
        
        # Update progress bar
        progress_bar.progress((i + 1) / frame_count, text=f"Analyzing frame {i+1}/{frame_count}")
            
    cap.release()
    progress_bar.empty() # Remove progress bar when done
    return red_intensities, frame_count, fps

def normalize_intensities(intensities):
    min_intensity = np.min(intensities)
    max_intensity = np.max(intensities)
    normalized_intensities = (intensities - min_intensity) / (max_intensity - min_intensity)
    return normalized_intensities

def find_peaks_in_intensity(intensities, distance=30, height=None, threshold=None, prominence=None):
    # Your original parameters from bpm.py (distance=15, prominence=0.05)
    peaks, properties = find_peaks(intensities, distance=15, height=height, threshold=threshold, prominence=0.05)
    return peaks, properties

def calculate_bpm(peaks, duration_in_seconds):
    num_beats = len(peaks)
    bpm = (num_beats / duration_in_seconds) * 60
    return bpm

def create_bpm_plot(intensities, peaks, video_name):
    # This replaces plt.show()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(intensities, label='Normalized Red Intensity')
    ax.plot(peaks, np.array(intensities)[peaks], "x", label=f'Detected Peaks ({len(peaks)})')
    ax.set_title(f'Red Intensity and Peaks for {video_name}')
    ax.set_xlabel('Frame Number')
    ax.set_ylabel('Normalized Average Red Intensity')
    ax.legend()
    return fig

def signal_quality_check(intensities, peaks, properties, fps):
    # Your original logic
    if len(peaks) == 0:
        return False, 0
        
    signal_power = np.mean(np.array(intensities)[peaks] ** 2)
    noise_power = np.var(intensities)
    snr = signal_power / (noise_power + 1e-10) 
    
    # Your check (modified slightly for robustness)
    min_peaks = (len(intensities) / fps) * (30 / 60) # 30 BPM
    if len(peaks) < min_peaks: 
        st.warning(f"Low peak count ({len(peaks)}). Result may be inaccurate.")
        return False, snr
    
    if snr < 2: 
        return False, snr
    
    return True, snr

# --- Streamlit App Interface (Replaces tkinter) ---

st.set_page_config(layout="wide", page_title="Heart Rate (BPM) Calculator")
st.title("Heart Rate (BPM) Calculator from Video")
st.write("Upload a video file (e.g., from a finger-tip recording) to calculate the BPM.")

# 1. Replaces tkinter file dialog
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    video_name = uploaded_file.name
    
    # Handle the uploaded file correctly for cv2
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video_name)[1]) as tmpfile:
        tmpfile.write(uploaded_file.getvalue())
        video_path = tmpfile.name

    st.write(f"Processing video: `{video_name}`")
    
    with st.spinner("Extracting signal..."):
        # Call your original function
        intensities, frame_count, fps = extract_red_intensity(video_path)
    
    # Clean up the temporary file
    os.remove(video_path)

    if intensities:
        with st.spinner("Analyzing signal and finding peaks..."):
            # Call your original functions
            normalized_intensities = normalize_intensities(intensities)
            peaks, properties = find_peaks_in_intensity(normalized_intensities)
            
            is_valid_signal, snr = signal_quality_check(normalized_intensities, peaks, properties, fps)
            
            # 2. Replaces tkinter messagebox
            if not is_valid_signal:
                st.error(f"The video '{video_name}' has poor signal quality (SNR: {snr:.2f}). Please use a better video with a clearer signal.")
            else:
                duration_in_seconds = frame_count / fps
                # Call your original function
                bpm = calculate_bpm(peaks, duration_in_seconds)
                
                st.success("Processing Complete!")
                
                # 3. Display results in Streamlit (replaces messagebox)
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(label="Calculated BPM", value=f"{bpm:.2f}")
                    st.metric(label="Video Length", value=f"{duration_in_seconds:.2f} s")
                    st.metric(label="Total Frames", value=frame_count)
                    st.metric(label="Video FPS", value=f"{fps:.2f}")
                    st.metric(label="Signal-to-Noise Ratio (SNR)", value=f"{snr:.2f}")
                
                with col2:
                    st.subheader("Signal Plot")
                    # Call your plot function
                    plot = create_bpm_plot(normalized_intensities, peaks, video_name)
                    # Display the plot in Streamlit (replaces plt.show())
                    st.pyplot(plot)

