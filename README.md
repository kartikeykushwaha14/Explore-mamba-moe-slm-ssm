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
Install Dependencies
pip install -r requirements.txt
Check GPU
python -c "import torch; print(torch.cuda.is_available())"
```

## 🔹 Usage
```bash
Run Model and Train
python mamba_modelv3.py
```

# ⚡ Key Learnings

## ❌ Challenges

- Wikipedia-style data caused overly formal outputs  
- No stop tokens caused endless generation  
- Small dataset (~10k samples) limited generalization  

---

## ✅ Improvements

- Structured chat templates improved responses  
- Data quality matters more than model size  
- Cleaning data improved performance significantly  

---

# 📈 Performance Insights

| Feature       | Transformers | Mamba SSM |
|--------------|-------------|----------|
| Complexity   | O(n²)       | O(n)     |
| Memory       | Grows       | Constant |
| Long Context | Expensive   | Efficient |
| Inference    | Slower      | Faster   |

---

# 🧩 How It Works (Simple)

### Transformer:
- Processes all tokens together  
- High compute cost  

### Mamba:
- Processes sequentially  
- Maintains internal state  
- Efficient for long sequences  

---

# 🛠️ Customization

You can modify:

- Model size (layers, hidden size)  
- Number of experts  
- Dataset composition  
- Training hyperparameters  

---

# 💻 Hardware Notes

- Works on 4GB VRAM GPUs (tested on RTX 3050)  
- CPU supported (slow)  
- Mixed precision recommended  

---

# 🚀 Future Improvements

- Better tokenizer  
- Improved MoE routing  
- Larger datasets  
- LoRA / PEFT integration  
- Web or API interface  

---

# 🤝 Contributing

Contributions are welcome:

- Open issues  
- Suggest features  
- Submit pull requests  

---

# ⭐ Support

If you like this project:

- ⭐ Star the repo  
- 🍴 Fork it  
- 🧠 Experiment with it  

---

# 🔗 License

MIT License 
