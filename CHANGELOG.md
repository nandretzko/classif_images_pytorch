# Changelog

All notable changes to this project will be documented in this file.  
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),  
versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Planned
- CLI arguments support (batch size, epochs, learning rate)
- ONNX export for production deployment
- Support for custom backbone selection at runtime
- DVC integration for dataset versioning
- Docker support with CUDA base image

---

## [0.1.0] — 2025-02-21

### Added
- Initial release of `train_classifier.py` (PyTorch)
- MobileNetV2 Transfer Learning pipeline via torchvision (2-phase training)
- Automatic GPU/CPU detection via `torch.cuda.is_available()`
- Automatic dataset structure validation
- Custom `EarlyStopping` class with best-weight restoration
- `ReduceLROnPlateau` scheduler
- Per-phase `ModelCheckpoint` saving (`.pth`)
- Training curves export (`training_curves.png`)
- Confusion matrix export (`confusion_matrix.png`)
- Classification report export (`classification_report.json`)
- Final model saved as `dataset/model.pth`
- `pyproject.toml` with project metadata and tool configuration
- `Makefile` with Windows-compatible shortcuts
- `.gitignore` tailored for PyTorch / Python / Windows projects
- `README.md` with full setup, usage, and inference documentation

---

<!-- Links -->
[Unreleased]: https://github.com/your-user/image-classifier/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/your-user/image-classifier/releases/tag/v0.1.0
