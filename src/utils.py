"""
utils.py
--------
Shared audio loading, preprocessing, and augmentation utilities
used across all 3 models in the Messy Mashup project.
"""

import os
import random
import numpy as np
import librosa
import soundfile as sf
import torch

# ── Constants ─────────────────────────────────────────────────
SR      = 16000
GENRES  = ["blues","classical","country","disco","hiphop",
           "jazz","metal","pop","reggae","rock"]
STEMS   = ["drums.wav", "bass.wav", "vocals.wav", "other.wav"]

genre_to_idx = {g: i for i, g in enumerate(GENRES)}
idx_to_genre = {i: g for g, i in genre_to_idx.items()}


def load_audio(path, sr=SR):
    """
    Load an audio file, convert to mono, resample to target SR.
    
    Args:
        path (str): Path to audio file
        sr   (int): Target sample rate (default 16000)
    
    Returns:
        np.ndarray: Audio waveform as float32
    """
    try:
        y, file_sr = sf.read(path)
        if len(y.shape) > 1:
            y = y.mean(axis=1)
        y = y.astype(np.float32)
        if file_sr != sr:
            y = librosa.resample(y=y, orig_sr=file_sr, target_sr=sr)
    except:
        y, _ = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float32)


def normalize(y):
    """
    Peak normalize audio to [-1, 1].
    
    Args:
        y (np.ndarray): Audio waveform
    
    Returns:
        np.ndarray: Normalized waveform
    """
    mx = np.max(np.abs(y))
    return y / mx if mx > 0 else y


def pad_or_trim(y, samples):
    """
    Pad or trim audio to exact length.
    
    Args:
        y       (np.ndarray): Audio waveform
        samples (int)       : Target length in samples
    
    Returns:
        np.ndarray: Fixed length waveform
    """
    if len(y) < samples:
        y = np.pad(y, (0, samples - len(y)))
    return y[:samples].astype(np.float32)


def mix_song(song_path, sr=SR):
    """
    Load and sum all 4 stems of a song into one mixed track.
    
    Args:
        song_path (str): Path to song directory containing stem files
        sr        (int): Target sample rate
    
    Returns:
        np.ndarray or None: Mixed normalized audio, or None if failed
    """
    mixed = None
    for stem in STEMS:
        p = os.path.join(song_path, stem)
        if not os.path.exists(p):
            continue
        try:
            y = load_audio(p, sr=sr)
            if mixed is None:
                mixed = y
            else:
                min_len = min(len(mixed), len(y))
                mixed   = mixed[:min_len] + y[:min_len]
        except:
            continue
    return normalize(mixed) if mixed is not None else None


def add_noise(audio, noise_files, snr_db=None):
    """
    Add ESC-50 environmental noise to audio at a random SNR.
    
    Args:
        audio       (np.ndarray): Input audio
        noise_files (list)      : List of noise file paths
        snr_db      (float)     : Signal-to-noise ratio in dB (random if None)
    
    Returns:
        np.ndarray: Noisy audio
    """
    if snr_db is None:
        snr_db = random.uniform(5, 20)
    try:
        noise = load_audio(random.choice(noise_files))
        if len(noise) < len(audio):
            noise = np.tile(noise, int(np.ceil(len(audio) / len(noise))))
        noise     = noise[:len(audio)]
        sig_rms   = np.sqrt(np.mean(audio**2)) + 1e-8
        noise_rms = np.sqrt(np.mean(noise**2))  + 1e-8
        noise     = noise * (sig_rms / noise_rms) / (10 ** (snr_db / 20))
        return audio + noise
    except:
        return audio


def waveform_to_logmel(audio, sr=SR, n_fft=1024,
                        hop_length=320, n_mels=128):
    """
    Convert waveform to Log-Mel Spectrogram.
    
    Args:
        audio      (np.ndarray): Input waveform
        sr         (int)       : Sample rate
        n_fft      (int)       : FFT window size
        hop_length (int)       : Hop length
        n_mels     (int)       : Number of mel filter banks
    
    Returns:
        np.ndarray: Log-mel spectrogram of shape (time_frames, n_mels)
    """
    mel    = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=n_fft,
        hop_length=hop_length, n_mels=n_mels
    )
    logmel = librosa.power_to_db(mel)
    return logmel.T  # (time, mel)


def time_shift(y, max_shift=0.1):
    """Randomly shift waveform forward/backward."""
    shift = int(random.uniform(-max_shift, max_shift) * len(y))
    return np.roll(y, shift).astype(np.float32)


def time_stretch(y, samples):
    """Slight speed change 0.9x-1.1x, re-fixed to target length."""
    rate      = random.uniform(0.9, 1.1)
    stretched = librosa.effects.time_stretch(y, rate=rate)
    return pad_or_trim(stretched, samples)


def pitch_shift(y, sr=SR):
    """Random pitch shift ±2 semitones."""
    steps = random.uniform(-2, 2)
    return librosa.effects.pitch_shift(
        y, sr=sr, n_steps=steps
    ).astype(np.float32)
