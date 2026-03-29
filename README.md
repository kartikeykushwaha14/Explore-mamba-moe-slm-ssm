# 🚀 Mamba-MoE Small Language Model (SLM)

A **Small Language Model (SLM)** built from scratch using **Mamba State Space Models (SSM)** combined with a **Mixture of Experts (MoE)** architecture.

This project explores an alternative to Transformers for sequence modeling, focusing on:
- ⚡ Faster inference
- 📉 Linear scaling with sequence length
- 🧠 Efficient memory usage
- 🔬 Research-oriented implementation

---

# 🧠 Why This Project?

Transformers dominate modern NLP, but they have key limitations:

- ❌ Quadratic complexity with sequence length  
- ❌ High memory usage  
- ❌ Expensive inference for long contexts  

---

## 💡 Motivation

This project is inspired by recent research on **State Space Models (SSMs)** and specifically **Mamba**, which introduces:

- Linear-time sequence processing  
- Constant memory usage  
- Better scalability for long sequences  

👉 The goal of this repository is:

> To build a **small, efficient, research-oriented language model** that demonstrates how Mamba + MoE can be used to create scalable AI systems.

---

# ⚙️ Architecture Overview

## 🔹 Core Components

### 1. Mamba SSM Blocks
- Uses learnable matrices:
  - **A** → controls memory decay  
  - **B** → controls input injection  
  - **C** → controls output projection  
  - **Δ (Delta)** → controls time step dynamics  

These allow the model to maintain and update a hidden state over time.

---

### 2. Mixture of Experts (MoE)
- **4 experts**
- **Top-2 routing per token**
- Dynamic expert selection per token

**Benefits:**
- Better parameter efficiency  
- Increased capacity without full compute cost  

---

### 3. SwiGLU Feedforward Networks
- Used inside each expert
- Improves expressiveness compared to standard FFN

---

### 4. RMSNorm
- Stabilizes training
- Efficient normalization technique

---

### 5. Depthwise Convolution
- Captures local context
- Complements SSM global memory

---

## 📊 Model Specs

| Component     | Value            |
|--------------|------------------|
| Parameters   | ~35M             |
| Experts      | 4                |
| Routing      | Top-2            |
| Architecture | Mamba SSM + MoE  |
| Precision    | Mixed (FP16)     |

---

# 📚 Datasets Used

The model is trained on a mixture of instruction, reasoning, and conversational datasets:

- Dolly-15k  
- GSM8K  
- Stanford Alpaca  
- OpenAssistant (oasst1)  
- ELI5  
- SciQ  
- HuggingFace Everyday Conversations  

---

## 🧠 Dataset Strategy

The dataset mix helps the model learn:

- 💬 Conversation  
- 🧠 Reasoning  
- 📖 Explanation  
- ✍️ Instruction following  

---

# 🛠️ Setup Guide

## 🔹 Requirements

- Python **3.8+**
- PyTorch **2.0+**
- CUDA GPU (**4GB+ VRAM recommended**)

---

## 🔹 Installation

```bash
git clone https://github.com/YOURNAME/mamba-moe-slm.git
cd mamba-moe-slm
