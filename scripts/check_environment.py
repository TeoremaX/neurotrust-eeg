from __future__ import annotations

import importlib
import platform
import sys

import torch


REQUIRED_PACKAGES = [
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "matplotlib",
    "yaml",
    "tqdm",
    "mne",
    "wfdb",
    "torch",
    "captum",
    "pytest",
]


def main() -> None:
    print("NeuroTrust-EEG environment check")
    print("=" * 40)
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    missing_packages = []

    for package in REQUIRED_PACKAGES:
        try:
            importlib.import_module(package)
            print(f"OK: {package}")
        except ImportError:
            print(f"MISSING: {package}")
            missing_packages.append(package)

    if missing_packages:
        packages = ", ".join(missing_packages)
        raise SystemExit(f"Missing packages: {packages}")

    print("=" * 40)
    print("Environment looks good.")


if __name__ == "__main__":
    main()