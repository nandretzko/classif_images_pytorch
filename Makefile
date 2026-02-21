# ─────────────────────────────────────────────────────────────
#  Makefile — image-classifier (PyTorch)
#  Compatible Windows (via Git Bash ou PowerShell + make)
#  Installer make sur Windows : winget install GnuWin32.Make
# ─────────────────────────────────────────────────────────────

PYTHON     = python
VENV       = venv
PIP        = $(VENV)/Scripts/pip
SCRIPT     = train_classifier.py
DATASET    = dataset

.DEFAULT_GOAL := help

# ── Aide ──────────────────────────────────────────────────────
.PHONY: help
help:
	@echo.
	@echo  ╔══════════════════════════════════════════╗
	@echo  ║   image-classifier (PyTorch) — Makefile  ║
	@echo  ╚══════════════════════════════════════════╝
	@echo.
	@echo  Commandes disponibles :
	@echo.
	@echo    make install        Install CPU dependencies in venv
	@echo    make install-cuda   Install GPU (CUDA 11.8) dependencies
	@echo    make train          Run the training pipeline
	@echo    make clean          Remove generated outputs
	@echo    make clean-all      Remove outputs AND virtual env
	@echo    make check          Check code style (flake8)
	@echo    make format         Auto-format code (black + isort)
	@echo    make gpu-check      Verify CUDA is detected by PyTorch
	@echo    make help           Show this message
	@echo.

# ── Environnement CPU ─────────────────────────────────────────
.PHONY: install
install:
	@echo [1/3] Creating virtual environment...
	$(PYTHON) -m venv $(VENV)
	@echo [2/3] Upgrading pip...
	$(PIP) install --upgrade pip
	@echo [3/3] Installing CPU dependencies...
	$(PIP) install -r requirements.txt
	@echo Done! Activate with: venv\Scripts\activate

# ── Environnement GPU (CUDA 11.8) ─────────────────────────────
.PHONY: install-cuda
install-cuda:
	@echo [1/3] Creating virtual environment...
	$(PYTHON) -m venv $(VENV)
	@echo [2/3] Upgrading pip...
	$(PIP) install --upgrade pip
	@echo [3/3] Installing GPU dependencies (CUDA 11.8)...
	$(PIP) install torch torchvision --index-url https://download.pytorch.org/whl/cu118
	$(PIP) install scikit-learn seaborn matplotlib numpy
	@echo Done! Activate with: venv\Scripts\activate

# ── Vérification GPU ──────────────────────────────────────────
.PHONY: gpu-check
gpu-check:
	$(VENV)/Scripts/python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# ── Entraînement ──────────────────────────────────────────────
.PHONY: train
train:
	@echo Starting training pipeline...
	$(VENV)/Scripts/python $(SCRIPT)

# ── Nettoyage ─────────────────────────────────────────────────
.PHONY: clean
clean:
	@echo Removing generated outputs...
	-del /Q $(DATASET)\*.pth 2>nul
	-del /Q $(DATASET)\*.pt 2>nul
	-del /Q $(DATASET)\*.png 2>nul
	-del /Q $(DATASET)\*.json 2>nul
	@echo Clean done.

.PHONY: clean-all
clean-all: clean
	@echo Removing virtual environment...
	-rmdir /S /Q $(VENV)
	@echo Full clean done.

# ── Qualité de code ───────────────────────────────────────────
.PHONY: check
check:
	$(VENV)/Scripts/flake8 $(SCRIPT) --max-line-length=100

.PHONY: format
format:
	$(VENV)/Scripts/isort $(SCRIPT)
	$(VENV)/Scripts/black $(SCRIPT) --line-length=100
