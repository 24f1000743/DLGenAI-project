"""
train.py
--------
Training logic for all 3 models in the Messy Mashup project:
1. CNNGenre       — CNN from scratch on log-mel spectrograms
2. ResNet18       — Pretrained ImageNet ResNet18 on mel spectrograms
3. AST            — Audio Spectrogram Transformer fine-tuned on AudioSet

Usage:
    python train.py --model cnn
    python train.py --model resnet
    python train.py --model ast
"""

import os
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import torchaudio.transforms as T
from transformers import ASTFeatureExtractor, ASTForAudioClassification
from tqdm import tqdm
import wandb

from utils import (
    load_audio, normalize, pad_or_trim, mix_song,
    add_noise, waveform_to_logmel, time_shift,
    time_stretch, pitch_shift,
    SR, GENRES, STEMS, genre_to_idx, idx_to_genre
)

# ── Config ────────────────────────────────────────────────────
DATA_ROOT  = "/kaggle/input/competitions/jan-2026-dl-gen-ai-project/messy_mashup"
GENRE_ROOT = os.path.join(DATA_ROOT, "genres_stems")
NOISE_ROOT = os.path.join(DATA_ROOT, "ESC-50-master", "audio")

noise_files = [os.path.join(NOISE_ROOT, f)
               for f in os.listdir(NOISE_ROOT) if f.endswith(".wav")]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── CNN Model ─────────────────────────────────────────────────
class CNNGenre(nn.Module):
    """
    4-block CNN for genre classification from log-mel spectrograms.
    Built entirely from scratch — no pretrained weights.
    """
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


# ── CNN Dataset ───────────────────────────────────────────────
class CNNDataset(Dataset):
    """Generates synthetic mashups for CNN training."""
    def __init__(self, genre_root, noise_files, synthetic_size=5000):
        self.noise_files    = noise_files
        self.synthetic_size = synthetic_size
        self.genre_dirs     = {}
        for genre in GENRES:
            gpath = os.path.join(genre_root, genre)
            self.genre_dirs[genre] = [
                os.path.join(gpath, s) for s in os.listdir(gpath)
            ]

    def __len__(self):
        return self.synthetic_size

    def __getitem__(self, idx):
        genre     = random.choice(GENRES)
        song_dirs = random.sample(self.genre_dirs[genre], 4)
        samples   = SR * 30
        stems     = []
        for sd in song_dirs:
            stem_name = random.choice(STEMS)
            p         = os.path.join(sd, stem_name)
            try:
                stems.append(pad_or_trim(load_audio(p), samples))
            except:
                stems.append(np.zeros(samples, dtype=np.float32))

        audio  = np.mean(stems, axis=0)
        if random.random() < 0.7:
            audio = add_noise(audio, self.noise_files)
        logmel = waveform_to_logmel(audio)
        logmel = (logmel - np.mean(logmel)) / (np.std(logmel) + 1e-6)
        x      = torch.tensor(logmel).unsqueeze(0).float()
        return x, genre_to_idx[genre]


# ── ResNet Dataset ────────────────────────────────────────────
class ResNetDataset(Dataset):
    """Wraps precomputed audio crops for ResNet training."""
    def __init__(self, chunks, augment_fn=None):
        self.chunks  = chunks
        self.augment = augment_fn
        self.mel     = T.MelSpectrogram(
            sample_rate=SR, n_fft=1024, hop_length=512, n_mels=128
        )
        self.db      = T.AmplitudeToDB()
        self.resize  = transforms.Resize((224, 224))

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        audio, label = self.chunks[idx]
        audio = audio.copy()
        if self.augment:
            audio = self.augment(audio)
        t   = torch.FloatTensor(audio).unsqueeze(0)
        mel = self.mel(t)
        mel = self.db(mel)
        mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-8)
        mel = mel.repeat(3, 1, 1)
        mel = self.resize(mel)
        return mel, label


# ── AST Dataset ───────────────────────────────────────────────
class ASTDataset(Dataset):
    """Wraps precomputed audio chunks for AST training."""
    def __init__(self, chunks, feature_extractor, augment_fn=None):
        self.chunks  = chunks
        self.fe      = feature_extractor
        self.augment = augment_fn

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        audio, label = self.chunks[idx]
        audio = audio.copy()
        if self.augment:
            audio = self.augment(audio)
        inputs = self.fe(audio, sampling_rate=SR, return_tensors="pt")
        x      = inputs["input_values"].squeeze(0)
        return x, label


# ── Training function ─────────────────────────────────────────
def train_model(model, loader, optimizer, scheduler=None,
                epochs=10, model_name="model"):
    """
    Generic training loop with W&B logging.

    Args:
        model      : PyTorch model
        loader     : DataLoader
        optimizer  : Optimizer
        scheduler  : LR scheduler (optional)
        epochs     : Number of epochs
        model_name : Name for W&B logging
    """
    model.train()
    for epoch in range(epochs):
        total_loss, correct, total = 0, 0, 0

        for x, y in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            x, y = x.to(DEVICE), y.to(DEVICE)

            if hasattr(model, 'logits'):
                outputs = model(x).logits
            else:
                outputs = model(x)

            loss = F.cross_entropy(outputs, y, label_smoothing=0.1)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            correct    += (outputs.argmax(1) == y).sum().item()
            total      += y.size(0)

        if scheduler:
            scheduler.step()

        acc = 100 * correct / total
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(loader):.4f} | Acc: {acc:.1f}%")
        wandb.log({
            "epoch"     : epoch + 1,
            "train_loss": total_loss / len(loader),
            "train_acc" : acc
        })

    return model


# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="cnn",
                        choices=["cnn", "resnet", "ast"])
    args = parser.parse_args()

    wandb.init(
        project="dlgenai-project",
        entity="24f1000743-iit-madras",
        name=f"train-{args.model}"
    )

    if args.model == "cnn":
        dataset   = CNNDataset(GENRE_ROOT, noise_files)
        loader    = DataLoader(dataset, batch_size=16,
                               shuffle=True, num_workers=0)
        model     = CNNGenre().to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        model     = train_model(model, loader, optimizer, epochs=10,
                                model_name="cnn")
        torch.save(model.state_dict(), "cnn_scratch_model.pth")

    elif args.model == "resnet":
        model     = models.resnet18(weights="IMAGENET1K_V1")
        model.fc  = nn.Sequential(nn.Dropout(0.2), nn.Linear(512, 10))
        model     = model.to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=1e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=8
        )
        print("Build dataset first — see pretrained-resnet.ipynb")

    elif args.model == "ast":
        fe    = ASTFeatureExtractor.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593"
        )
        model = ASTForAudioClassification.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593",
            num_labels=10, ignore_mismatched_sizes=True
        ).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=2e-5, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=8
        )
        print("Build dataset first — see ast-pretrained.ipynb")

    wandb.finish()
