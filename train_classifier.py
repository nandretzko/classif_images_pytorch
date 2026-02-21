"""
=============================================================
  Module de Classification d'Images — Transfer Learning
  Architecture : MobileNetV2 (torchvision) | Images : 224x224
  Framework    : PyTorch (pur)
=============================================================
Structure attendue du dossier dataset/ :
    dataset/
    ├── train/
    │   ├── classe_A/
    │   └── classe_B/
    ├── val/
    │   ├── classe_A/
    │   └── classe_B/
    └── test/
        ├── classe_A/
        └── classe_B/
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
DATASET_DIR   = "dataset"
OUTPUT_MODEL  = os.path.join(DATASET_DIR, "model.pth")
OUTPUT_PLOT   = os.path.join(DATASET_DIR, "training_curves.png")
OUTPUT_CM     = os.path.join(DATASET_DIR, "confusion_matrix.png")
OUTPUT_REPORT = os.path.join(DATASET_DIR, "classification_report.json")

IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
EPOCHS      = 50          # Early stopping arrêtera avant si nécessaire
LR          = 1e-4        # Learning rate Phase 1
LR_FINETUNE = 1e-5        # Learning rate Phase 2 (fine-tuning)
FINE_TUNE   = True        # Dégeler les dernières couches après phase 1
FINE_TUNE_AT = 14         # Index de bloc MobileNetV2 à partir duquel on dégèle (sur 19)

# Early stopping
ES_PATIENCE  = 7          # Epochs sans amélioration avant arrêt
LR_PATIENCE  = 3          # Epochs sans amélioration avant réduction LR
LR_FACTOR    = 0.5        # Facteur de réduction du LR
LR_MIN       = 1e-7       # LR minimum

TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR   = os.path.join(DATASET_DIR, "val")
TEST_DIR  = os.path.join(DATASET_DIR, "test")

# Détection automatique GPU/CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────
#  1. VÉRIFICATION DE LA STRUCTURE
# ─────────────────────────────────────────────
def verify_dataset_structure():
    """Vérifie que train/val/test existent et ont les mêmes classes."""
    for split in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        if not os.path.isdir(split):
            raise FileNotFoundError(f"Dossier manquant : {split}")

    train_classes = sorted(os.listdir(TRAIN_DIR))
    val_classes   = sorted(os.listdir(VAL_DIR))
    test_classes  = sorted(os.listdir(TEST_DIR))

    if train_classes != val_classes or train_classes != test_classes:
        raise ValueError(
            f"Les classes ne correspondent pas entre les splits.\n"
            f"  Train : {train_classes}\n"
            f"  Val   : {val_classes}\n"
            f"  Test  : {test_classes}"
        )

    print(f"✅ Structure validée — {len(train_classes)} classes : {train_classes}")
    return train_classes


# ─────────────────────────────────────────────
#  2. CHARGEMENT DES DONNÉES
# ─────────────────────────────────────────────
def build_dataloaders():
    """Crée les DataLoaders train / val / test avec les transforms MobileNetV2."""

    # Normalisation officielle ImageNet (identique à MobileNetV2 Keras)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    train_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        normalize,
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    val_dataset   = datasets.ImageFolder(VAL_DIR,   transform=eval_transform)
    test_dataset  = datasets.ImageFolder(TEST_DIR,  transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    print(f"\n📦 Données chargées :")
    print(f"   Train : {len(train_dataset)} images")
    print(f"   Val   : {len(val_dataset)} images")
    print(f"   Test  : {len(test_dataset)} images")
    print(f"   Classes : {train_dataset.classes}")

    return train_loader, val_loader, test_loader, train_dataset.classes


# ─────────────────────────────────────────────
#  3. CONSTRUCTION DU MODÈLE
# ─────────────────────────────────────────────
def build_model(num_classes: int):
    """MobileNetV2 pré-entraîné sur ImageNet + tête de classification."""

    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    # Geler tout le backbone en phase 1
    for param in model.parameters():
        param.requires_grad = False

    # Remplacer la tête de classification
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes),
    )

    model = model.to(DEVICE)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🧠 Modèle construit sur : {DEVICE}")
    print(f"   Paramètres total        : {total_params:,}")
    print(f"   Paramètres entraînables : {trainable_params:,}")

    return model


def unfreeze_for_finetuning(model):
    """Dégèle les blocs MobileNetV2 à partir de FINE_TUNE_AT."""
    for param in model.features[FINE_TUNE_AT:].parameters():
        param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Paramètres entraînables après dégel : {trainable_params:,}")
    return model


# ─────────────────────────────────────────────
#  4. EARLY STOPPING
# ─────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience=7, checkpoint_path="best.pth"):
        self.patience        = patience
        self.checkpoint_path = checkpoint_path
        self.best_loss       = float("inf")
        self.counter         = 0
        self.best_epoch      = 0

    def step(self, val_loss, model, epoch):
        if val_loss < self.best_loss:
            self.best_loss  = val_loss
            self.counter    = 0
            self.best_epoch = epoch
            torch.save(model.state_dict(), self.checkpoint_path)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            print(f"\n⏹️  Early stopping déclenché (meilleure epoch : {self.best_epoch + 1})")
            return True
        return False

    def load_best(self, model):
        model.load_state_dict(torch.load(self.checkpoint_path, map_location=DEVICE))
        return model


# ─────────────────────────────────────────────
#  5. BOUCLE D'ENTRAÎNEMENT
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted  = outputs.max(1)
        correct      += predicted.eq(labels).sum().item()
        total        += inputs.size(0)

    return running_loss / total, correct / total


def evaluate(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs        = model(inputs)
            loss           = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted  = outputs.max(1)
            correct      += predicted.eq(labels).sum().item()
            total        += inputs.size(0)

    return running_loss / total, correct / total


def run_phase(model, train_loader, val_loader, phase, lr):
    """Lance une phase d'entraînement complète avec early stopping."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_FACTOR,
        patience=LR_PATIENCE, min_lr=LR_MIN, verbose=True
    )

    checkpoint = os.path.join(DATASET_DIR, f"best_model_phase{phase}.pth")
    early_stop = EarlyStopping(patience=ES_PATIENCE, checkpoint_path=checkpoint)

    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}

    print(f"\n{'─'*50}")
    print(f" Phase {phase} — LR={lr:.0e} | Device={DEVICE}")
    print(f"{'─'*50}")

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion)
        scheduler.step(val_loss)

        history["loss"].append(train_loss)
        history["accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        print(
            f"  Epoch {epoch+1:3d}/{EPOCHS} | "
            f"Loss {train_loss:.4f} | Acc {train_acc*100:.2f}% | "
            f"Val Loss {val_loss:.4f} | Val Acc {val_acc*100:.2f}%"
        )

        if early_stop.step(val_loss, model, epoch):
            break

    # Restaurer les meilleurs poids
    model = early_stop.load_best(model)
    print(f"✅ Phase {phase} terminée — meilleur Val Loss : {early_stop.best_loss:.4f}")
    return model, history


# ─────────────────────────────────────────────
#  6. COURBES D'APPRENTISSAGE
# ─────────────────────────────────────────────
def plot_training_curves(history1, history2=None):
    acc      = history1["accuracy"]
    val_acc  = history1["val_accuracy"]
    loss     = history1["loss"]
    val_loss = history1["val_loss"]
    phase1_end = len(acc)

    if history2:
        acc      += history2["accuracy"]
        val_acc  += history2["val_accuracy"]
        loss     += history2["loss"]
        val_loss += history2["val_loss"]

    epochs_range = range(len(acc))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Courbes d'apprentissage — MobileNetV2 (PyTorch)", fontsize=14, fontweight="bold")

    axes[0].plot(epochs_range, acc,     label="Train Accuracy")
    axes[0].plot(epochs_range, val_acc, label="Val Accuracy")
    if history2:
        axes[0].axvline(x=phase1_end - 1, color="gray", linestyle="--", label="Début Fine-tuning")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Époque")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs_range, loss,     label="Train Loss")
    axes[1].plot(epochs_range, val_loss, label="Val Loss")
    if history2:
        axes[1].axvline(x=phase1_end - 1, color="gray", linestyle="--", label="Début Fine-tuning")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Époque")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=150)
    plt.close()
    print(f"\n📈 Courbes sauvegardées → {OUTPUT_PLOT}")


# ─────────────────────────────────────────────
#  7. ÉVALUATION SUR LE TEST SET
# ─────────────────────────────────────────────
def evaluate_model(model, test_loader, class_names):
    """Génère le rapport de classification et la matrice de confusion."""
    print("\n🔍 Évaluation sur le test set...")
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    y_pred = np.array(all_preds)
    y_true = np.array(all_labels)

    # Rapport de classification
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    print("\n📋 Rapport de classification :")
    print(classification_report(y_true, y_pred, target_names=class_names))

    with open(OUTPUT_REPORT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"💾 Rapport sauvegardé → {OUTPUT_REPORT}")

    # Matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(max(6, len(class_names)), max(5, len(class_names) - 1)))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax
    )
    ax.set_title("Matrice de Confusion", fontsize=13, fontweight="bold")
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")
    plt.tight_layout()
    plt.savefig(OUTPUT_CM, dpi=150)
    plt.close()
    print(f"🔢 Matrice de confusion sauvegardée → {OUTPUT_CM}")

    # Accuracy globale
    test_acc = (y_pred == y_true).mean()
    print(f"\n✅ Test Accuracy : {test_acc * 100:.2f}%")

    return report


# ─────────────────────────────────────────────
#  8. PIPELINE PRINCIPAL
# ─────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  Classification d'Images — MobileNetV2 (PyTorch)")
    print("=" * 55)
    print(f"  Device détecté : {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"  GPU : {torch.cuda.get_device_name(0)}")

    # Vérification
    class_names = verify_dataset_structure()
    num_classes = len(class_names)

    # Données
    train_loader, val_loader, test_loader, _ = build_dataloaders()

    # Modèle
    model = build_model(num_classes)

    # ── Phase 1 : Tête seulement ──
    print("\n📌 Phase 1 : Entraînement de la tête de classification...")
    model, history1 = run_phase(model, train_loader, val_loader, phase=1, lr=LR)

    history2 = None

    # ── Phase 2 : Fine-tuning ──
    if FINE_TUNE:
        print(f"\n📌 Phase 2 : Fine-tuning à partir du bloc {FINE_TUNE_AT}...")
        model = unfreeze_for_finetuning(model)
        model, history2 = run_phase(model, train_loader, val_loader, phase=2, lr=LR_FINETUNE)

    # Courbes
    plot_training_curves(history1, history2)

    # Évaluation
    evaluate_model(model, test_loader, class_names)

    # Sauvegarde finale
    torch.save(model.state_dict(), OUTPUT_MODEL)
    print(f"\n💾 Modèle final sauvegardé → {OUTPUT_MODEL}")
    print("\n🎉 Entraînement terminé avec succès !")


if __name__ == "__main__":
    main()
