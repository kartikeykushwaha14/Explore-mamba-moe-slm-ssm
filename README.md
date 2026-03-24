# Mamba-MoE Small Language Model

A small language model built with Mamba SSM + Mixture of Experts.

## Architecture
- Mamba SSM blocks with learnable A, B, C, Delta matrices
- Mixture of Experts (4 experts, top-2 routing)
- SwiGLU FFN per expert
- RMSNorm throughout
- ~35M parameters

## Datasets
- Dolly-15k
- GSM8K
- Stanford Alpaca
- OpenAssistant (oasst1)
- ELI5
- SciQ
- HuggingFace Everyday Conversations

## Requirements
- Python 3.8+
- CUDA GPU (4GB+ VRAM recommended)
- PyTorch 2.0+

## Usage
pip install -r requirements.txt
python mamba_model.py
