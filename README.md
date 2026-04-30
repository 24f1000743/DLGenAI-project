# DLGenAI Project — Music Genre Classification

**Student:** Vrindesh Pareek
**Roll No:** 24f1000743
**Term:** Jan 2026

## About
Music genre classification on the Messy Mashup Kaggle competition.
The task is to classify noisy audio mashups into 10 genres using deep learning.

## Models
- CNN from Scratch
- ResNet18 (Pretrained)
- AST — Audio Spectrogram Transformer (Pretrained)

## Setup
```bash
pip install -r requirements.txt
Links
HuggingFace: https://huggingface.co/24f1000743/messy-mashup-resnet-genre
Demo: https://huggingface.co/spaces/24f1000743/messy-mashup-genre-classifier
---

# Notebooks

| Notebook | Description |
|----------|-------------|
| `scratch-cnn.ipynb` | CNN built from scratch on log-mel spectrograms |
| `pretrained-resnet.ipynb` | ResNet18 fine-tuned on mel spectrograms |
| `ast-pretrained.ipynb` | AST transformer fine-tuned for genre classification |
| `final_notebook.ipynb` | EDA + inference + W&B comparison + HuggingFace push |

W&B runs: https://wandb.ai/24f1000743-iit-madras/dlgenai-project
