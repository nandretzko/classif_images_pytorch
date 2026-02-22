"""
Image Classification — Transfer Learning with MobileNetV2.

Architecture : MobileNetV2 (torchvision) | Input size : 224 × 224
Framework    : PyTorch

Expected dataset layout
-----------------------
::

    dataset/
    ├── train/
    │   ├── class_A/
    │   └── class_B/
    ├── val/
    │   ├── class_A/
    │   └── class_B/
    └── test/
        ├── class_A/
        └── class_B/

Typical usage
-------------
    python train.py
"""

# ── Standard library ──────────────────────────────────────────────────────────
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Third-party ───────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# ── Constants — paths ─────────────────────────────────────────────────────────
DATASET_DIR: Path = Path("dataset")
OUTPUT_MODEL: Path = DATASET_DIR / "model.pth"
OUTPUT_PLOT: Path = DATASET_DIR / "training_curves.png"
OUTPUT_CM: Path = DATASET_DIR / "confusion_matrix.png"
OUTPUT_REPORT: Path = DATASET_DIR / "classification_report.json"

TRAIN_DIR: Path = DATASET_DIR / "train"
VAL_DIR: Path = DATASET_DIR / "val"
TEST_DIR: Path = DATASET_DIR / "test"

# ── Constants — image / data ──────────────────────────────────────────────────
IMG_SIZE: Tuple[int, int] = (224, 224)
BATCH_SIZE: int = 32
NUM_WORKERS: int = 2

# ImageNet normalization statistics (required by MobileNetV2 pre-trained weights)
IMAGENET_MEAN: List[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: List[float] = [0.229, 0.224, 0.225]

# ── Constants — training ──────────────────────────────────────────────────────
EPOCHS: int = 50          # Upper bound; early stopping may trigger earlier
LR: float = 1e-4          # Phase 1 learning rate (classification head only)
LR_FINETUNE: float = 1e-5 # Phase 2 learning rate (backbone fine-tuning)

FINE_TUNE: bool = True    # Whether to run Phase 2 fine-tuning
FINE_TUNE_AT: int = 14    # MobileNetV2 block index from which to unfreeze (0–18)

# ── Constants — schedulers & early stopping ───────────────────────────────────
ES_PATIENCE: int = 7      # Epochs without improvement before early stopping
LR_PATIENCE: int = 3      # Epochs without improvement before LR reduction
LR_FACTOR: float = 0.5    # Multiplicative LR reduction factor
LR_MIN: float = 1e-7      # Minimum learning rate floor

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── 1. Dataset validation ─────────────────────────────────────────────────────

def verify_dataset_structure() -> List[str]:
    """Verify that train / val / test directories exist and share identical classes.

    Raises
    ------
    FileNotFoundError
        If any of the three split directories is missing.
    ValueError
        If the class lists differ across splits.

    Returns
    -------
    List[str]
        Sorted list of class names found in the training split.
    """
    for split_dir in (TRAIN_DIR, VAL_DIR, TEST_DIR):
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing directory: {split_dir}")

    train_classes = sorted(os.listdir(TRAIN_DIR))
    val_classes = sorted(os.listdir(VAL_DIR))
    test_classes = sorted(os.listdir(TEST_DIR))

    if train_classes != val_classes or train_classes != test_classes:
        raise ValueError(
            "Class lists differ across splits.\n"
            f"  Train : {train_classes}\n"
            f"  Val   : {val_classes}\n"
            f"  Test  : {test_classes}"
        )

    print(f"Dataset validated — {len(train_classes)} classes: {train_classes}")
    return train_classes


# ── 2. Data loading ───────────────────────────────────────────────────────────

def build_dataloaders() -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """Build train, validation, and test :class:`~torch.utils.data.DataLoader` objects.

    Applies standard MobileNetV2 / ImageNet preprocessing:

    * Resize to :data:`IMG_SIZE`.
    * Convert to tensor.
    * Normalize with ImageNet mean and standard deviation.

    Training data uses random horizontal flipping for basic augmentation.

    Returns
    -------
    train_loader:
        Shuffled DataLoader for the training split.
    val_loader:
        DataLoader for the validation split (no shuffle).
    test_loader:
        DataLoader for the test split (no shuffle).
    class_names:
        Ordered list of class labels derived from the training folder.
    """
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    train_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = datasets.ImageFolder(str(TRAIN_DIR), transform=train_transform)
    val_dataset = datasets.ImageFolder(str(VAL_DIR), transform=eval_transform)
    test_dataset = datasets.ImageFolder(str(TEST_DIR), transform=eval_transform)

    loader_kwargs = {"batch_size": BATCH_SIZE, "num_workers": NUM_WORKERS, "pin_memory": True}
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    print(
        f"\nData loaded:\n"
        f"  Train : {len(train_dataset)} images\n"
        f"  Val   : {len(val_dataset)} images\n"
        f"  Test  : {len(test_dataset)} images\n"
        f"  Classes : {train_dataset.classes}"
    )
    return train_loader, val_loader, test_loader, train_dataset.classes


# ── 3. Model construction ─────────────────────────────────────────────────────

def build_model(num_classes: int) -> nn.Module:
    """Instantiate MobileNetV2 pre-trained on ImageNet with a custom classification head.

    The entire backbone is frozen in Phase 1 (transfer learning).
    Only the new classification head is trainable at this stage.

    The replacement head architecture::

        Dropout(0.3) → Linear(1280, 256) → ReLU → Dropout(0.3) → Linear(256, num_classes)

    Parameters
    ----------
    num_classes:
        Number of output classes (must match the dataset structure).

    Returns
    -------
    nn.Module
        MobileNetV2 model moved to :data:`DEVICE` with a custom head.
    """
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    for param in model.parameters():
        param.requires_grad = False

    in_features: int = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes),
    )

    model = model.to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"\nModel built on: {DEVICE}\n"
        f"  Total parameters     : {total_params:,}\n"
        f"  Trainable parameters : {trainable_params:,}"
    )
    return model


def unfreeze_for_finetuning(model: nn.Module) -> nn.Module:
    """Unfreeze MobileNetV2 backbone blocks from index :data:`FINE_TUNE_AT` onward.

    Called at the start of Phase 2 to enable gradual fine-tuning of the
    upper convolutional blocks while keeping lower-level feature extractors
    frozen.

    Parameters
    ----------
    model:
        MobileNetV2 model whose ``features`` attribute is partially frozen.

    Returns
    -------
    nn.Module
        The same model with selected layers set to ``requires_grad=True``.
    """
    for param in model.features[FINE_TUNE_AT:].parameters():
        param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters after unfreeze: {trainable_params:,}")
    return model


# ── 4. Early stopping ─────────────────────────────────────────────────────────

class EarlyStopping:
    """Monitor validation loss and stop training when improvement stalls.

    The best model weights are checkpointed to disk whenever a new minimum
    validation loss is reached.  Training should be halted when
    :meth:`step` returns ``True``.

    Parameters
    ----------
    patience:
        Number of epochs with no improvement before triggering a stop.
    checkpoint_path:
        File path where the best model state dict will be saved.

    Attributes
    ----------
    best_loss:
        Lowest validation loss observed so far.
    counter:
        Number of consecutive epochs without improvement.
    best_epoch:
        Zero-based index of the epoch that achieved the best loss.
    """

    def __init__(self, patience: int = 7, checkpoint_path: str = "best.pth") -> None:
        self.patience: int = patience
        self.checkpoint_path: str = checkpoint_path
        self.best_loss: float = float("inf")
        self.counter: int = 0
        self.best_epoch: int = 0

    def step(self, val_loss: float, model: nn.Module, epoch: int) -> bool:
        """Update state with the latest validation loss and checkpoint if improved.

        Parameters
        ----------
        val_loss:
            Validation loss for the current epoch.
        model:
            Model whose state dict will be saved on improvement.
        epoch:
            Zero-based epoch index (used for logging).

        Returns
        -------
        bool
            ``True`` if training should stop, ``False`` otherwise.
        """
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
            torch.save(model.state_dict(), self.checkpoint_path)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            print(f"\nEarly stopping triggered (best epoch: {self.best_epoch + 1})")
            return True
        return False

    def load_best(self, model: nn.Module) -> nn.Module:
        """Restore model weights from the best checkpoint saved to disk.

        Parameters
        ----------
        model:
            Model instance whose weights will be overwritten.

        Returns
        -------
        nn.Module
            The same model with best-checkpoint weights loaded.
        """
        model.load_state_dict(torch.load(self.checkpoint_path, map_location=DEVICE))
        return model


# ── 5. Training loop ──────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
) -> Tuple[float, float]:
    """Run a single training epoch over *loader*.

    Parameters
    ----------
    model:
        Model to train (set to ``train()`` mode internally).
    loader:
        DataLoader yielding ``(inputs, labels)`` batches.
    criterion:
        Loss function (e.g. :class:`~torch.nn.CrossEntropyLoss`).
    optimizer:
        Gradient-based optimizer (e.g. Adam).

    Returns
    -------
    avg_loss:
        Mean loss per sample over the full epoch.
    accuracy:
        Fraction of correctly classified samples.
    """
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += inputs.size(0)

    return running_loss / total, correct / total


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> Tuple[float, float]:
    """Evaluate *model* on *loader* without updating weights.

    Parameters
    ----------
    model:
        Model to evaluate (set to ``eval()`` mode internally).
    loader:
        DataLoader yielding ``(inputs, labels)`` batches.
    criterion:
        Loss function used to compute the reported loss.

    Returns
    -------
    avg_loss:
        Mean loss per sample over the full split.
    accuracy:
        Fraction of correctly classified samples.
    """
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += inputs.size(0)

    return running_loss / total, correct / total


def run_phase(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    phase: int,
    lr: float,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """Execute a complete training phase with early stopping and LR scheduling.

    Each phase runs for up to :data:`EPOCHS` epochs.  A
    :class:`~torch.optim.lr_scheduler.ReduceLROnPlateau` scheduler
    reduces the learning rate when validation loss plateaus, and an
    :class:`EarlyStopping` instance halts training when no improvement is
    observed for :data:`ES_PATIENCE` consecutive epochs.

    Parameters
    ----------
    model:
        Model to train.  Only parameters with ``requires_grad=True`` are
        passed to the optimizer.
    train_loader:
        DataLoader for the training split.
    val_loader:
        DataLoader for the validation split.
    phase:
        Phase identifier (``1`` or ``2``) used for logging and checkpoint
        naming.
    lr:
        Initial learning rate for this phase.

    Returns
    -------
    model:
        Model restored to its best-checkpoint weights after the phase.
    history:
        Dictionary with keys ``loss``, ``accuracy``, ``val_loss``,
        ``val_accuracy``, each mapping to a list of per-epoch values.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_FACTOR,
        patience=LR_PATIENCE,
        min_lr=LR_MIN,
    )

    checkpoint_path = str(DATASET_DIR / f"best_model_phase{phase}.pth")
    early_stop = EarlyStopping(patience=ES_PATIENCE, checkpoint_path=checkpoint_path)

    history: Dict[str, List[float]] = {
        "loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []
    }

    print(f"\n{'─' * 50}")
    print(f" Phase {phase} — LR={lr:.0e} | Device={DEVICE}")
    print(f"{'─' * 50}")

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        scheduler.step(val_loss)

        history["loss"].append(train_loss)
        history["accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        print(
            f"  Epoch {epoch + 1:3d}/{EPOCHS} | "
            f"Loss {train_loss:.4f} | Acc {train_acc * 100:.2f}% | "
            f"Val Loss {val_loss:.4f} | Val Acc {val_acc * 100:.2f}%"
        )

        if early_stop.step(val_loss, model, epoch):
            break

    model = early_stop.load_best(model)
    print(f"Phase {phase} complete — best Val Loss: {early_stop.best_loss:.4f}")
    return model, history


# ── 6. Visualization ──────────────────────────────────────────────────────────

def plot_training_curves(
    history1: Dict[str, List[float]],
    history2: Optional[Dict[str, List[float]]] = None,
) -> None:
    """Plot and save accuracy and loss learning curves.

    If *history2* is provided (Phase 2 fine-tuning), the two phases are
    concatenated and a vertical dashed line marks the phase boundary.

    Parameters
    ----------
    history1:
        Training history from Phase 1.  Must contain keys ``accuracy``,
        ``val_accuracy``, ``loss``, ``val_loss``.
    history2:
        Optional training history from Phase 2 fine-tuning.

    Side effects
    ------------
    Saves a PNG figure to :data:`OUTPUT_PLOT` and closes the figure.
    """
    acc = list(history1["accuracy"])
    val_acc = list(history1["val_accuracy"])
    loss = list(history1["loss"])
    val_loss = list(history1["val_loss"])
    phase1_end = len(acc)

    if history2:
        acc += history2["accuracy"]
        val_acc += history2["val_accuracy"]
        loss += history2["loss"]
        val_loss += history2["val_loss"]

    epochs_range = range(len(acc))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Learning Curves — MobileNetV2 (PyTorch)", fontsize=14, fontweight="bold"
    )

    for ax, (train_values, val_values, title, ylabel) in zip(
        axes,
        [
            (acc, val_acc, "Accuracy", "Accuracy"),
            (loss, val_loss, "Loss", "Loss"),
        ],
    ):
        ax.plot(epochs_range, train_values, label="Train")
        ax.plot(epochs_range, val_values, label="Val")
        if history2:
            ax.axvline(
                x=phase1_end - 1,
                color="gray",
                linestyle="--",
                label="Fine-tuning start",
            )
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=150)
    plt.close()
    print(f"\nLearning curves saved → {OUTPUT_PLOT}")


# ── 7. Test-set evaluation ────────────────────────────────────────────────────

def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    class_names: List[str],
) -> Dict:
    """Evaluate the trained model on the held-out test set.

    Computes and saves:

    * A full :func:`~sklearn.metrics.classification_report` (JSON + stdout).
    * A :func:`~sklearn.metrics.confusion_matrix` heatmap (PNG).

    Parameters
    ----------
    model:
        Trained model to evaluate (set to ``eval()`` mode internally).
    test_loader:
        DataLoader for the test split.
    class_names:
        Ordered list of class labels matching the dataset folder names.

    Returns
    -------
    Dict
        Classification report as a nested dictionary (as returned by
        :func:`~sklearn.metrics.classification_report` with
        ``output_dict=True``).

    Side effects
    ------------
    Saves ``classification_report.json`` to :data:`OUTPUT_REPORT` and a
    confusion matrix PNG to :data:`OUTPUT_CM`.
    """
    print("\nEvaluating on test set...")
    model.eval()

    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    y_pred = np.array(all_preds)
    y_true = np.array(all_labels)

    # Classification report
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True
    )
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2)
    print(f"Report saved → {OUTPUT_REPORT}")

    # Confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    fig_size = max(6, len(class_names))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size - 1))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(OUTPUT_CM, dpi=150)
    plt.close()
    print(f"Confusion matrix saved → {OUTPUT_CM}")

    test_acc = (y_pred == y_true).mean()
    print(f"\nTest Accuracy: {test_acc * 100:.2f}%")

    return report


# ── 8. Main pipeline ──────────────────────────────────────────────────────────

def main() -> None:
    """Orchestrate the full two-phase training and evaluation pipeline.

    Pipeline steps
    --------------
    1. Validate dataset directory structure.
    2. Build DataLoaders for all three splits.
    3. Instantiate the MobileNetV2 model with a custom head.
    4. **Phase 1** — Train the classification head only (backbone frozen).
    5. **Phase 2** — Fine-tune upper backbone blocks (if :data:`FINE_TUNE`).
    6. Plot and save learning curves.
    7. Evaluate the final model on the test set.
    8. Serialize the final model weights to :data:`OUTPUT_MODEL`.
    """
    print("=" * 55)
    print("  Image Classification — MobileNetV2 (PyTorch)")
    print("=" * 55)
    print(f"  Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"  GPU   : {torch.cuda.get_device_name(0)}")

    class_names = verify_dataset_structure()
    num_classes = len(class_names)

    train_loader, val_loader, test_loader, _ = build_dataloaders()

    model = build_model(num_classes)

    # Phase 1 — classification head only
    print("\nPhase 1: Training classification head...")
    model, history1 = run_phase(model, train_loader, val_loader, phase=1, lr=LR)

    history2: Optional[Dict[str, List[float]]] = None

    # Phase 2 — fine-tuning
    if FINE_TUNE:
        print(f"\nPhase 2: Fine-tuning from block {FINE_TUNE_AT}...")
        model = unfreeze_for_finetuning(model)
        model, history2 = run_phase(model, train_loader, val_loader, phase=2, lr=LR_FINETUNE)

    plot_training_curves(history1, history2)

    evaluate_model(model, test_loader, class_names)

    torch.save(model.state_dict(), OUTPUT_MODEL)
    print(f"\nFinal model saved → {OUTPUT_MODEL}")
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
