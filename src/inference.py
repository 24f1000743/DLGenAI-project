"""
inference.py
------------
Inference pipeline for all 3 trained models in the Messy Mashup project.
Loads saved weights and generates genre predictions on test audio files.

Usage:
    python inference.py --model resnet --test_csv /path/to/test.csv
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torchvision import models, transforms
import torchaudio.transforms as T
from transformers import ASTFeatureExtractor, ASTForAudioClassification
from tqdm import tqdm

from utils import (
    load_audio, normalize, pad_or_trim,
    waveform_to_logmel, time_shift, time_stretch, add_noise,
    SR, GENRES, idx_to_genre
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Model definitions ─────────────────────────────────────────
class CNNGenre(nn.Module):
    """4-block CNN from scratch for genre classification."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32),
            nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),
            nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128),
            nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256),
            nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def build_resnet():
    """ResNet18 with custom classification head."""
    model    = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(512, 10)
    )
    return model


def build_ast():
    """AST fine-tuned for 10-genre classification."""
    return ASTForAudioClassification.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593",
        num_labels=10,
        ignore_mismatched_sizes=True
    )


# ── Preprocessing ─────────────────────────────────────────────
def cnn_preprocess(path):
    """Load audio → log-mel → normalize → CNN tensor."""
    samples = SR * 30
    y       = pad_or_trim(load_audio(path), samples)
    logmel  = waveform_to_logmel(y)
    logmel  = (logmel - np.mean(logmel)) / (np.std(logmel) + 1e-6)
    return torch.tensor(logmel).unsqueeze(0).unsqueeze(0).float()


def resnet_preprocess(path):
    """Load audio → mel spectrogram → 3-channel 224x224 tensor."""
    samples      = SR * 5
    y            = load_audio(path)
    if len(y) < samples:
        y = np.pad(y, (0, samples - len(y)))
    start = max(0, (len(y) - samples) // 2)
    y     = y[start:start + samples].astype(np.float32)

    mel_transform = T.MelSpectrogram(
        sample_rate=SR, n_fft=1024, hop_length=512, n_mels=128
    )
    db_transform = T.AmplitudeToDB()
    resize       = transforms.Resize((224, 224))

    t   = torch.FloatTensor(y).unsqueeze(0)
    mel = mel_transform(t)
    mel = db_transform(mel)
    mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-8)
    mel = mel.repeat(3, 1, 1)
    mel = resize(mel)
    return mel.unsqueeze(0)


def ast_preprocess(path, feature_extractor):
    """Load audio → AST feature extractor → tensor."""
    samples = SR * 10
    try:
        import soundfile as sf
        y, file_sr = sf.read(path)
        if len(y.shape) > 1:
            y = y.mean(axis=1)
        y = y.astype(np.float32)
        if file_sr != SR:
            import librosa
            y = librosa.resample(y=y, orig_sr=file_sr, target_sr=SR)
    except:
        import librosa
        y, _ = librosa.load(path, sr=SR, mono=True)
    y      = pad_or_trim(y, samples)
    inputs = feature_extractor(y, sampling_rate=SR, return_tensors="pt")
    return inputs["input_values"]


# ── TTA Inference ─────────────────────────────────────────────
def predict_resnet_tta(path, model, n_tta=4):
    """ResNet inference with 4-crop TTA."""
    model.eval()
    samples      = SR * 5
    y            = load_audio(path)
    if len(y) < samples:
        y = np.pad(y, (0, samples - len(y)))

    mel_transform = T.MelSpectrogram(
        sample_rate=SR, n_fft=1024, hop_length=512, n_mels=128
    )
    db_transform = T.AmplitudeToDB()
    resize       = transforms.Resize((224, 224))

    def to_mel(audio):
        t   = torch.FloatTensor(audio).unsqueeze(0)
        mel = mel_transform(t)
        mel = db_transform(mel)
        mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-8)
        mel = mel.repeat(3, 1, 1)
        return resize(mel).unsqueeze(0)

    import random
    all_logits = []
    for _ in range(n_tta):
        start = random.randint(0, max(len(y) - samples, 0))
        crop  = y[start:start + samples]
        crop  = normalize(crop)
        x     = to_mel(crop).to(DEVICE)
        with torch.no_grad():
            all_logits.append(model(x).cpu())

    avg = torch.stack(all_logits).mean(0)
    return torch.argmax(avg, dim=1).item()


def predict_ast_tta(path, model, feature_extractor, noise_files, n_tta=4):
    """AST inference with 4-view TTA."""
    model.eval()
    samples = SR * 10
    y       = pad_or_trim(load_audio(path), samples)

    views = [
        normalize(y),
        normalize(add_noise(y.copy(), noise_files, snr_db=10)),
        normalize(time_shift(y.copy())),
        normalize(time_stretch(y.copy(), samples)),
    ]
    logits_all = []
    for v in views[:n_tta]:
        inp = feature_extractor(v, sampling_rate=SR, return_tensors="pt")
        x   = inp["input_values"].to(DEVICE)
        with torch.no_grad():
            out = model(x)
        logits_all.append(out.logits.cpu())
    avg = torch.stack(logits_all).mean(0)
    return torch.argmax(avg, dim=1).item()


# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet",
                        choices=["cnn", "resnet", "ast"])
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to model weights .pth file")
    parser.add_argument("--test_csv", type=str, required=True,
                        help="Path to test.csv")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to data root directory")
    parser.add_argument("--output", type=str, default="submission.csv")
    args = parser.parse_args()

    test_df     = pd.read_csv(args.test_csv)
    predictions = []

    if args.model == "cnn":
        model = CNNGenre().to(DEVICE)
        model.load_state_dict(torch.load(args.weights, map_location=DEVICE))
        model.eval()
        for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
            path = os.path.join(args.data_root, row["filename"])
            try:
                x    = cnn_preprocess(path).to(DEVICE)
                pred = torch.argmax(model(x), dim=1).item()
            except:
                pred = 0
            predictions.append(idx_to_genre[pred])

    elif args.model == "resnet":
        model = build_resnet().to(DEVICE)
        model.load_state_dict(torch.load(args.weights, map_location=DEVICE))
        for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
            path = os.path.join(args.data_root, row["filename"])
            try:
                pred = predict_resnet_tta(path, model)
            except:
                pred = 0
            predictions.append(idx_to_genre[pred])

    elif args.model == "ast":
        fe    = ASTFeatureExtractor.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593"
        )
        model = build_ast().to(DEVICE)
        model.load_state_dict(torch.load(args.weights, map_location=DEVICE))
        for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
            path = os.path.join(args.data_root, row["filename"])
            try:
                pred = predict_ast_tta(path, model, fe, [])
            except:
                pred = 0
            predictions.append(idx_to_genre[pred])

    submission = pd.DataFrame({
        "id"   : test_df["id"],
        "genre": predictions
    })
    submission.to_csv(args.output, index=False)
    print(f"Submission saved → {args.output}")
    print(submission["genre"].value_counts())
