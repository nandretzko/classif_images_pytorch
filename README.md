# image-classifier (PyTorch)

Image classification module using **Transfer Learning with MobileNetV2** (PyTorch / torchvision).  
Drop your dataset in, run one command, get a trained `.pth` model — that's it.

---

# Project Structure

```
image-classifier/
├── dataset/                   ← Not versioned (see .gitignore)
│   ├── train/
│   │   ├── class_A/
│   │   └── class_B/
│   ├── val/
│   │   ├── class_A/
│   │   └── class_B/
│   └── test/
│       ├── class_A/
│       └── class_B/
├── train_classifier.py        ← Main training script
├── pyproject.toml             ← Project metadata & dependencies
├── requirements.txt           ← Pip dependencies
├── Makefile                   ← Shortcut commands
├── .gitignore
├── CHANGELOG.md
└── README.md
```

>  The `dataset/` folder is excluded from Git. Each contributor brings their own data locally.

---

#  Requirements

- Python **3.9+**
- pip or a virtual environment manager (venv, conda)
- CUDA (optional) — detected automatically, falls back to CPU

---

#  Quick Start

**1. Clone the repository**
```bash
git clone https://github.com/your-user/image-classifier.git
cd image-classifier
```

**2. Create a virtual environment**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

>  For GPU (CUDA 11.8) support, replace the torch line in requirements.txt with:
> ```
> --index-url https://download.pytorch.org/whl/cu118
> torch>=2.0.0
> torchvision>=0.15.0
> ```

**4. Prepare your dataset**

Place your images following this structure:
```
dataset/
├── train/   ← ~70% of your images
├── val/     ← ~15% of your images
└── test/    ← ~15% of your images
```
Each subfolder must contain one folder per class, with images inside.  
Class names must be **identical** across train, val, and test.

**5. Train the model**
```bash
python train_classifier.py
```

Or using Make:
```bash
make train
```

---

#  Outputs

After training, the following files are generated inside `dataset/`:

| File | Description |
|------|-------------|
| `model.pth` | Final trained model (state dict) |
| `best_model_phase1.pth` | Best checkpoint — Phase 1 (head only) |
| `best_model_phase2.pth` | Best checkpoint — Phase 2 (fine-tuning) |
| `training_curves.png` | Loss & accuracy curves |
| `confusion_matrix.png` | Confusion matrix heatmap |
| `classification_report.json` | Per-class precision, recall, F1 |

---

#  Model Architecture

| Parameter | Value |
|-----------|-------|
| Base model | MobileNetV2 (ImageNet weights via torchvision) |
| Input size | 224 × 224 × 3 |
| Head | Dropout(0.3) → Linear(256) → ReLU → Dropout(0.3) → Linear(num_classes) |
| Training phases | 2 (frozen backbone → partial fine-tuning) |
| Optimizer | Adam |
| Loss | CrossEntropyLoss |
| Device | Auto-detected (CUDA if available, else CPU) |

**Phase 1** — Only the classification head is trained (backbone fully frozen).  
**Phase 2** — Fine-tuning: MobileNetV2 blocks from index `FINE_TUNE_AT=14` onwards are unfrozen with a 10× lower learning rate.

---

##  Callbacks / Training Features

- **Early Stopping** — stops if `val_loss` doesn't improve for 7 epochs, restores best weights
- **ModelCheckpoint** — saves the best `.pth` per phase
- **ReduceLROnPlateau** — halves LR if `val_loss` plateaus for 3 epochs

---

#  Loading the Model for Inference

```python
import torch
from torchvision import models
import torch.nn as nn

# Rebuild architecture
model = models.mobilenet_v2(weights=None)
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(in_features, 256),
    nn.ReLU(),
    nn.Dropout(p=0.3),
    nn.Linear(256, NUM_CLASSES),  # <- replace with your number of classes
)

# Load weights
model.load_state_dict(torch.load("dataset/model.pth", map_location="cpu"))
model.eval()
```

---

#  Makefile Commands

```bash
make install     # Install dependencies
make train       # Run training
make clean       # Remove generated outputs (*.pth, *.png, *.json)
make clean-all   # Remove outputs AND virtual env
make check       # Lint with flake8
make format      # Auto-format with black + isort
make help        # List all commands
```

---

#  Configuration

Key parameters can be adjusted at the top of `train_classifier.py`:

```python
BATCH_SIZE   = 32
EPOCHS       = 50
LR           = 1e-4     # Phase 1 learning rate
LR_FINETUNE  = 1e-5     # Phase 2 learning rate
FINE_TUNE    = True
FINE_TUNE_AT = 14       # MobileNetV2 block index to unfreeze from (0–19)
ES_PATIENCE  = 7        # Early stopping patience
```

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.
