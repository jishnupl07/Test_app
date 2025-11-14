import cv2
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import find_peaks, butter, filtfilt, detrend
import os

# --- THIS FINDS THE MODEL FILE ON THE SERVER ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "bp_model_lstm.pth")
# ---

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
            nn.MaxPool3d(kernel_size=(2, 2, 2)), # 100 -> 50
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)), # 50 -> 25
            nn.Flatten()
        )
        self.lstm = nn.LSTM(input_size=32 * 16 * 16, hidden_size=128, num_layers=1, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(128 + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 2) 
        )

    # --- THIS IS THE CORRECTED FORWARD FUNCTION ---
    def forward(self, x, bpm):
        # x arrives as a 4D tensor: (batch=1, frames=100, height=64, width=64)
        
        # Add the 'channel' dimension to make it 5D for Conv3d
        # Shape becomes: (1, 1, 100, 64, 64)
        x = x.unsqueeze(1) 
        
        batch_size = x.size(0)
        
        # self.cnn(x) applies Conv, Pool, Conv, Pool, Flatten
        # Output shape is (batch_size, 32 * 25 * 16 * 16) = (1, 204800)
        cnn_out = self.cnn(x)
        
        # Reshape for LSTM: (batch, seq_len, features)
        # The pooling layers reduced the frame dimension from 100 to 25.
        # We must reshape to (1, 25, 8192)
        # 8192 = 32 * 16 * 16
        cnn_out = cnn_out.view(batch_size, 25, -1)
        
        # Pass to LSTM
        lstm_out, _ = self.lstm(cnn_out)
        lstm_last_out = lstm_out[:, -1, :] 
        
        # Combine with BPM
        combined = torch.cat((lstm_last_out, bpm.view(-1, 1)), dim=1)
        output = self.fc(combined)
        return output
    # --- END OF CORRECTED FUNCTION ---


def load_model():
    model = BPRegressionModel()
    # --- THIS IS THE FIX ---
    # Load model state dict, mapping to CPU and using the correct path
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    # --- END OF FIX ---
    model.eval()
    return model


def predict_bp(model, video_path):
    bpm = extract_bpm(video_path)
    # video_data is 4D: (1, 100, 64, 64)
    video_data = load_video_frames(video_path) 
    
    with torch.no_grad():
        # --- THIS IS THE FIX ---
        # We pass the 4D tensor directly to the model.
        # The 'forward' function will handle adding the 5th dimension.
        prediction = model(video_data, torch.tensor([bpm]).float())
        # --- END OF FIX ---
        
    systolic, diastolic = prediction.squeeze().tolist()
    
    # This returns 3 values as expected by app.py
    return systolic, diastolic, bpm
