RMBG-2.0 Background Removal Node for ComfyUI
This custom node for ComfyUI provides background removal functionality using the briaai/RMBG-2.0 model.
Node name: GFrbmg2
Author: GorillaFrame


Features
High-quality background removal
Support for batch processing
Options for inverted masks
Three output formats: RGBA, mask, and black background
Installation
Required Directory Structure

ComfyUI/
├── models/
│   └── RMBG-2.0/ # Model files
│   	├── config.json
│   	├── pytorch_model.bin (≈1.5GB)
│   	└── other files...
│└── GF_nodes/
		└── gfrbmg2.py # Node implementation
		└── init.py # Node implementation
		
Installation Steps
Navigate to your ComfyUI/models directory
Create RMBG-2.0 folder and download model:
cd ComfyUI/models
mkdir RMBG-2.0
cd RMBG-2.0
git clone https://huggingface.co/briaai/RMBG-2.0 .
If Git LFS is not installed:
git lfs install
git lfs pull
Requirements
Git LFS (for model download)
Sufficient disk space (≈2GB)
CUDA-compatible GPU (recommended)
Usage
Restart ComfyUI after installation
Find the node in "🐵 GorillaFrame/Image" category
Connect an image input
Get outputs: image_rgba (transparent background), mask, and image_black (black background)
Credits
Original model: briaai/RMBG-2.0

Node implementation: @GorillaFrame 🐵