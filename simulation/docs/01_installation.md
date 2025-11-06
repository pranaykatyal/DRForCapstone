# 01 - Isaac Sim Installation

## Overview
Installing NVIDIA Isaac Sim for multi-drone asynchronous communication research.

## System Requirements

### Hardware
- **GPU**: NVIDIA RTX 4080 Laptop (12GB VRAM) ✓
- **RAM**: 32GB recommended (minimum 16GB)
- **Storage**: ~50GB free space for Isaac Sim + assets
- **OS**: Ubuntu 24.04 LTS ✓

### Software Prerequisites
- NVIDIA Driver: 580.65.06 ✓
- CUDA: 13.0 ✓
- Python: 3.10 or 3.11

## Installation Methods

Isaac Sim can be installed via:
1. **Omniverse Launcher** (Recommended for beginners)
2. **pip install** (Lighter, CLI-focused)
3. **Docker** (Reproducible, isolated)

We'll use **pip install** method for better control and documentation.

---

## Step 1: Download Omniverse Launcher

### Download
```bash
# Navigate to downloads
cd ~/Downloads

# Download Omniverse Launcher
wget https://install.launcher.omniverse.nvidia.com/installers/omniverse-launcher-linux.AppImage

# Make executable
chmod +x omniverse-launcher-linux.AppImage
```
