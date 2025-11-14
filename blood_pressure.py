import cv2
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import find_peaks, butter, filtfilt, detrend
import os

# --- THIS IS THE CRUCIAL CHANGE ---
# This block finds the correct path to the model file on the server
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "bp_model_lstm.pth")
# --- END OF CHANGE ---

def extract_red_intensity(video_path):
    cap = cv2.VideoCapture(video_path)
    red_intensities = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        red_channel = frame[:, :, 2] 
        avg_red_intensity = np.mean(red_channel)
        red_intensities.append(avg_red_intensity)
    cap.release()
    return red_intensities

def preprocess_signal(intensities, fps):
    detrended_signal = detrend(intensities)
    nyquist = 0.5 * fps
    low = 0.5 / nyquist
    high = 4 / nyquist
    b, a = butter(4, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, detrended_signal)
    return filtered_signal

def calculate_bpm(peaks, frame_count, fps):
    num_beats = len(peaks)
    duration_in_seconds = frame_count / fps
    bpm = (num_beats / duration_in_seconds) * 60
    return bpm

def extract_bpm(video_path):
    intensities = extract_red_intensity(video_path)
    # Assuming a fixed FPS if not available, which your code does.
    # We'll try to get it from the main app, but this is a fallback.
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 30 # Default fallback
    cap.release()
    
    filtered_signal = preprocess_signal(intensities, fps)
    peaks, _ = find_peaks(filtered_signal, distance=fps//2, prominence=0.01)
    return calculate_bpm(peaks, len(intensities), fps)

def load_video_frames(video_path, num_frames=100):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_resized = cv2.resize(frame, (64, 64))
        frames.append(frame_resized[:, :, 2] / 255.0)  # Red channel normalization
        count += 1
    cap.release()
    
    if len(frames) < num_frames:
        # Pad with zeros if video is too short
        padding = np.zeros((num_frames - len(frames), 64, 64))
        if len(frames) > 0:
            frames = np.concatenate((np.stack(frames, axis=0), padding), axis=0)
        else:
            frames = padding
    
    frames = np.stack(frames, axis=0) if isinstance(frames, list) else frames
    # Returns a 4D tensor: (batch=1, frames=100, height=64, width=64)
    return torch.tensor(frames, dtype=torch.float32).unsqueeze(0)

class BPRegressionModel(nn.Module):
    def __init__(self):
        super(BPRegressionModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.Flatten()
        )
        # Fix input size for LSTM: 32 * 16 * 16 = 8192
        self.lstm = nn.LSTM(input_size=8192, hidden_size=128, num_layers=1, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(128 + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 2) 
        )

    def forward(self, x, bpm):
        # x is (1, 100, 64, 64)
        
        # We need to add the channel dimension: (1, 1, 100, 64, 64)
        x = x.unsqueeze(1) # Add channel dimension
        
        # --- THE BUG WAS HERE ---
        # The line below was trying to unpack 5 dims from a 4-dim tensor
        # It's no longer needed as we unsqueeze *before* this.
        # batch_size, channels, frames, height, width = x.size()
        
        # cnn_out shape: (1, 32, 25, 16, 16)
        cnn_out = self.cnn(x)
        
        # We need to permute and flatten for LSTM
        # (batch, channels, frames, h, w) -> (batch, frames, features)
        cnn_out = cnn_out.permute(0, 2, 1, 3, 4) # (1, 25, 32, 16, 16)
        
        # Flatten the last 3 dims
        batch_size, frames_dim, c, h, w = cnn_out.size()
        cnn_out = cnn_out.reshape(batch_size, frames_dim, -1) # (1, 25, 32*16*16)
        
        # Your LSTM input_size is 32 * 16 * 16 = 8192. This matches.
        
        lstm_out, _ = self.lstm(cnn_out)
        lstm_last_out = lstm_out[:, -1, :] 
        combined = torch.cat((lstm_last_out, bpm.view(-1, 1)), dim=1)
        output = self.fc(combined)
        return output


def load_model():
    # It now uses the MODEL_PATH variable
    model = BPRegressionModel()
    
    # Load model state dict, mapping to CPU
    # This ensures it works on Streamlit's CPU-only servers
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model


def predict_bp(model, video_path):
    bpm = extract_bpm(video_path)
    video_data = load_video_frames(video_path) # Shape: (1, 100, 64, 64)
    with torch.no_grad():
        prediction = model(video_data, torch.tensor([bpm]).float())
    systolic, diastolic = prediction.squeeze().tolist()
    return systolic, diastolic
