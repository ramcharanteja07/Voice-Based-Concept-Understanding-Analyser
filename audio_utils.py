import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np


def save_temp_audio(audio_bytes, path="temp_audio.wav"):
    with open(path, "wb") as f:
        f.write(audio_bytes)
    return path


def load_audio(path):
    y, sr = librosa.load(path, sr=None, mono=True)
    return y, sr


def normalize_audio(y):
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak
    return y


def plot_waveform(y, sr, save_path=None):
    fig, ax = plt.subplots(figsize=(6, 2))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title("Waveform Visualization")
    if save_path:
        fig.savefig(save_path)
    return fig


def read_audio_with_soundfile(path):
    y, sr = sf.read(path)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    return y, sr


def compute_pause_ratio(y, threshold=0.01):
    return round(np.sum(np.abs(y) < threshold) / len(y), 2)


def compute_rms_energy(y):
    rms = librosa.feature.rms(y=y)[0]
    return round(float(np.mean(rms)), 4)