import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
import sentencepiece as spm
import os
import random
from torch.cuda.amp import autocast, GradScaler

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)



# =========================
# CONFIG
# =========================
# CONFIG = {
#     "max_texts": 800,
#     "seq_len": 32,
#     "batch_size": 4,
#     "epochs": 4,
#     "d_model": 192,
#     "n_layers": 4,
#     "num_experts": 3,
#     "lr": 3e-4
# }

CONFIG = {
    "max_texts": 8000,     # BIG dataset
    "seq_len": 32,
    "batch_size": 4,
    "epochs": 5,

    "d_model": 192,
    "n_layers": 4,
    "num_experts": 3,
    "lr": 3e-4
}

CHECKPOINT = "model.pt"
SP_MODEL = "sp.model"

def load_mixed_data(max_total=5000):
    from datasets import load_dataset
    import requests
    import random

    texts = []

    # ======================
    # 1. WikiText
    # ======================
    print("Loading WikiText...")
    wiki = load_dataset("wikitext", "wikitext-2-raw-v1")["train"]

    for x in wiki[:1500]:
        if isinstance(x, dict):
            txt = x.get("text", "")
        else:
            txt = x

        if isinstance(txt, str) and txt.strip():
            texts.append(txt)

    # ======================
    # 2. Dolly
    # ======================
    print("Loading Dolly...")
    dolly = load_dataset("databricks/databricks-dolly-15k")["train"]

    for x in dolly[:1500]:
        if isinstance(x, dict):
            instruction = x.get("instruction", "")
            response = x.get("response", "")
            texts.append(f"User: {instruction} Assistant: {response}")
        else:
            # fallback if string
            if isinstance(x, str) and x.strip():
                texts.append(x)

    # ======================
    # 3. GSM8K
    # ======================
    print("Loading GSM8K...")
    gsm = load_dataset("gsm8k", "main")["train"]

    for x in gsm[:800]:
        if isinstance(x, dict):
            q = x.get("question", "")
            a = x.get("answer", "")
            texts.append(f"User: {q} Assistant: Let's think step by step. {a}")
        else:
            if isinstance(x, str):
                texts.append(x)

    # ======================
    # 4. Alpaca
    # ======================
    print("Loading Alpaca...")
    url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"

    try:
        data = requests.get(url).json()
    except Exception as e:
        print("Alpaca load failed:", e)
        data = []

    for item in data[:1200]:
        if isinstance(item, dict):
            prompt = item.get("instruction", "")
            if item.get("input"):
                prompt += " " + item["input"]

            response = item.get("output", "")
            texts.append(f"User: {prompt} Assistant: {response}")

    # ======================
    # FINAL
    # ======================
    random.shuffle(texts)

    print("Total samples:", len(texts))
    return texts[:max_total]

# =========================
# TOKENIZER
# =========================
def train_tokenizer(texts):
    with open("temp.txt", "w") as f:
        f.write("\n".join(texts))

    spm.SentencePieceTrainer.train(
        input="temp.txt",
        model_prefix="sp",
        vocab_size=2000
    )

def load_tokenizer():
    sp = spm.SentencePieceProcessor()
    sp.load(SP_MODEL)
    return sp

# =========================
# MAMBA BLOCK
# =========================
class MambaBlock(nn.Module):
    def __init__(self, d_model, state_dim=64):
        super().__init__()
        self.A = nn.Linear(d_model, state_dim)
        self.B = nn.Linear(d_model, state_dim)
        self.out = nn.Linear(state_dim, d_model)

    def forward(self, x):
        B, T, D = x.shape
        h = torch.zeros(B, self.A.out_features, device=x.device)

        outputs = []
        for t in range(T):
            xt = x[:, t]
            h = torch.tanh(self.A(xt)) * h + torch.tanh(self.B(xt))
            outputs.append(self.out(h).unsqueeze(1))

        return x + torch.cat(outputs, dim=1)

# =========================
# MoE BLOCK
# =========================
class MoEMamba(nn.Module):
    def __init__(self, d_model, num_experts):
        super().__init__()
        self.router = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList([
            MambaBlock(d_model) for _ in range(num_experts)
        ])

    def forward(self, x):
        weights = F.softmax(self.router(x), dim=-1)

        out = 0
        for i, expert in enumerate(self.experts):
            out += weights[..., i:i+1] * expert(x)

        return out

# =========================
# MODEL
# =========================
class Model(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.embed = nn.Embedding(vocab, CONFIG["d_model"])

        self.layers = nn.ModuleList([
            MoEMamba(CONFIG["d_model"], CONFIG["num_experts"])
            for _ in range(CONFIG["n_layers"])
        ])

        self.norm = nn.LayerNorm(CONFIG["d_model"])
        self.head = nn.Linear(CONFIG["d_model"], vocab)

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        return self.head(self.norm(x))

def collate_fn(batch):
    x = torch.tensor([item[0] for item in batch], dtype=torch.long)
    y = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return x, y

# =========================
# DATA PREP
# =========================
print("Preparing dataset...")
texts = load_mixed_data()[:CONFIG["max_texts"]]

if not os.path.exists(SP_MODEL):
    train_tokenizer(texts)

sp = load_tokenizer()
vocab_size = sp.get_piece_size()

def encode(texts):
    data = []
    for t in texts:
        ids = sp.encode(t)
        for i in range(len(ids) - CONFIG["seq_len"]):
            x = ids[i:i+CONFIG["seq_len"]]
            y = ids[i+1:i+CONFIG["seq_len"]+1]
            data.append((x, y))
    return data

train_data = encode(texts)
loader = DataLoader(
    train_data,
    batch_size=CONFIG["batch_size"],
    shuffle=True,
    collate_fn=collate_fn
)

# =========================
# INIT
# =========================
model = Model(vocab_size).to(device)
opt = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
scaler = GradScaler()

best_loss = float("inf")

if os.path.exists(CHECKPOINT):
    print("Loading checkpoint...")
    ckpt = torch.load(CHECKPOINT)
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["opt"])

# =========================
# TRAIN
# =========================
def train():
    global best_loss

    # 🔥 ensure best_loss exists
    if "best_loss" not in globals():
        best_loss = float("inf")

    model.train()

    for epoch in range(CONFIG["epochs"]):
        print(f"\nEpoch {epoch}")

        for step, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)

            # 🔥 AMP only if GPU
            if device == "cuda":
                with autocast():
                    logits = model(x)
                    loss = F.cross_entropy(
                        logits.reshape(-1, vocab_size),   # 🔥 FIXED (no view)
                        y.reshape(-1)
                    )
            else:
                logits = model(x)
                loss = F.cross_entropy(
                    logits.reshape(-1, vocab_size),
                    y.reshape(-1)
                )

            # 🔥 NaN protection (VERY IMPORTANT)
            if torch.isnan(loss):
                print("⚠️ NaN detected, skipping step")
                continue

            opt.zero_grad()

            if device == "cuda":
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            scheduler.step()

            # 🔥 logging
            if step % 20 == 0:
                print(f"Step {step} | Loss: {loss.item():.4f}")

            # 🔥 save best model
            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save({
                    "model": model.state_dict(),
                    "opt": opt.state_dict()
                }, CHECKPOINT)

# =========================
# FINE-TUNE
# =========================
def fine_tune(path):
    print("Fine-tuning on:", path)

    with open(path) as f:
        new_texts = f.readlines()

    data = encode(new_texts)
    loader = DataLoader(data, batch_size=CONFIG["batch_size"], shuffle=True)

    model.train()
    for epoch in range(2):
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

            opt.zero_grad()
            loss.backward()
            opt.step()

    torch.save({"model": model.state_dict(), "opt": opt.state_dict()}, CHECKPOINT)

# =========================
# GENERATION
# =========================
def generate(prompt, max_len=50, temperature=0.7, top_k=50, top_p=0.9, repetition_penalty=1.2):
    model.eval()

    tokens = torch.tensor(sp.encode(prompt)).unsqueeze(0).to(device)

    for _ in range(max_len):
        with torch.no_grad():
            logits = model(tokens)[:, -1, :]

            for token in set(tokens[0].tolist()):
                logits[0, token] /= repetition_penalty

            logits = logits / temperature

            topk_vals, topk_idx = torch.topk(logits, top_k)
            probs = F.softmax(topk_vals, dim=-1)

            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            mask = cumulative_probs > top_p
            sorted_probs[mask] = 0
            sorted_probs /= sorted_probs.sum()

            next_token = sorted_idx.gather(-1, torch.multinomial(sorted_probs, 1))

        tokens = torch.cat([tokens, next_token], dim=1)

    return sp.decode(tokens[0].tolist())

# =========================
# RUN
# =========================
train()

print("\n=== TEST ===")
print(generate("User: What is AI?\nAssistant:", temperature=0.5))
print(generate("The future of AI is", temperature=1.0))