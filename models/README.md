# 📦 Models

CNN and ResNet18 weights are stored in this folder.
AST weights are hosted externally due to file size (320MB).

| Model | Location | Size |
|-------|----------|------|
| ResNet18 (best) | `resnet18_model.pth` (this folder) + [HuggingFace Hub](https://huggingface.co/24f1000743/messy-mashup-resnet-genre) | 44MB |
| CNN Scratch | `cnn_scratch_model.pth` (this folder) | 1.58MB |
| AST | Kaggle Dataset: `ast-weights` | 320MB |

## Loading Weights
```python
# ResNet18 (from this repo)
import torch
from torchvision import models
import torch.nn as nn

model = models.resnet18(weights=None)
model.fc = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(512, 10)
)
model.load_state_dict(torch.load("models/resnet18_model.pth", map_location="cpu"))

# AST (from HuggingFace)
from huggingface_hub import hf_hub_download
weights_path = hf_hub_download(
    repo_id="24f1000743/messy-mashup-resnet-genre",
    filename="resnet_genre_model.pth"
)
