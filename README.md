# 🎵 Messy Mashup — Music Genre Classification
### DL & GenAI Project | Jan 2026 Term

**Student:** Vrindesh Pareek  
**Roll No:** 24f1000743  
**Competition:** [Kaggle — Messy Mashup](https://www.kaggle.com/competitions/jan-2026-dl-gen-ai-project)  
**Best Score:** 0.81+ Macro F1  

---

## 📌 Project Overview
End-to-end music genre classification system trained on noisy audio mashups.
Classifies audio into 10 genres: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock.

---

## 🏗️ Repository Structure
DLGenAI-project/
├── notebooks/         ← Training & inference notebooks
├── src/               ← Modular Python scripts
│   ├── utils.py       ← Audio loading & augmentation utilities
│   ├── train.py       ← Training logic for all 3 models
│   └── inference.py   ← Inference pipeline
├── models/            ← Model weights (hosted externally)
├── reports/           ← Project report
├── requirements.txt   ← Dependencies
└── README.md
---

## 🤖 Models
| Model | Type | Macro F1 |
|-------|------|----------|
| CNNGenre | From Scratch | ~0.56|
| AST | Pretrained Transformer | 0.78+ |
| ResNet18 | Pretrained (ImageNet) | 0.81+ ✅ |

---

## 📦 Model Weights
Weights hosted externally due to file size limits:
- **ResNet18 (best):** https://huggingface.co/24f1000743/messy-mashup-resnet-genre
- **CNN & AST:** Kaggle Datasets

---

## 🚀 Deployment
Live demo on HuggingFace Spaces:  
👉 https://huggingface.co/spaces/24f1000743/messy-mashup-genre-classifier

---

## 🔗 Links
- **Kaggle Notebook:** https://www.kaggle.com/code/ds24f1000743/dl-24f1000743-notebook-t12026
- **W&B Project:** https://wandb.ai/24f1000743-iit-madras/dlgenai-project
- **HuggingFace Model:** https://huggingface.co/24f1000743/messy-mashup-resnet-genre
- **HuggingFace Space:** https://huggingface.co/spaces/24f1000743/messy-mashup-genre-classifier

---

## ⚙️ Setup
```bash
pip install -r requirements.txt
📊 Results
Final submission achieved 0.81+ Macro F1 on the Kaggle leaderboard
using ResNet18 fine-tuned on mel spectrograms with TTA inference.
