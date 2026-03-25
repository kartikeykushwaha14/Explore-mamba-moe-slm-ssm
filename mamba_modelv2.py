import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
import sentencepiece as spm
import os, random
from torch.cuda.amp import autocast, GradScaler

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# =========================
# CONFIG
# =========================
CONFIG = {
    "max_texts": 8000,
    "seq_len": 64,
    "batch_size": 4,
    "epochs": 3,

    "d_model": 256,
    "n_layers": 6,
    "num_experts": 4,
    "top_k": 2,

    "lr": 3e-4
}

CHECKPOINT = "model.pt"
SP_MODEL = "sp.model"

# =========================
# DATASET (IMPROVED FORMAT)
# =========================
SYSTEM_PROMPT = """You are a helpful AI assistant.
- Give clear answers
- Use step-by-step reasoning when needed
- Be accurate and concise
"""

def format_sample(user, assistant):
    return f"{SYSTEM_PROMPT}\nUser: {user}\nAssistant: {assistant}"

def load_mixed_data(max_total=8000):
    from datasets import load_dataset
    import requests

    texts = []

    # Wiki (language)
    wiki = load_dataset("wikitext", "wikitext-2-raw-v1")["train"]
    for x in wiki[:2000]:
        txt = x["text"] if isinstance(x, dict) else x
        if txt.strip():
            texts.append(txt)

    # Dolly (chat)
    dolly = load_dataset("databricks/databricks-dolly-15k")["train"]
    for x in dolly[:2000]:
        texts.append(format_sample(x["instruction"], x["response"]))

    # GSM8K (reasoning)
    gsm = load_dataset("gsm8k", "main")["train"]
    for x in gsm[:1000]:
        texts.append(format_sample(
            x["question"],
            "Let's think step by step. " + x["answer"]
        ))

    # Alpaca
    url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
    data = requests.get(url).json()
    for item in data[:2000]:
        prompt = item["instruction"]
        if item["input"]:
            prompt += " " + item["input"]
        texts.append(format_sample(prompt, item["output"]))

    random.shuffle(texts)
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
        vocab_size=3000
    )

def load_tokenizer():
    sp = spm.SentencePieceProcessor()
    sp.load(SP_MODEL)
    return sp

# =========================
# GATED MAMBA BLOCK (IMPROVED)
# =========================
class GatedMamba(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.in_proj = nn.Linear(d_model, d_model * 2)
        self.state = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, D = x.shape
        h = torch.zeros(B, D, device=x.device)

        out = []
        for t in range(T):
            xt = x[:, t]
            gate, val = self.in_proj(xt).chunk(2, dim=-1)
            gate = torch.sigmoid(gate)

            h = gate * torch.tanh(self.state(val)) + (1 - gate) * h
            out.append(self.out(h).unsqueeze(1))

        return x + torch.cat(out, dim=1)

# =========================
# TOP-K MoE (REAL)
# =========================
class MoE(nn.Module):
    def __init__(self, d_model, num_experts, top_k=2):
        super().__init__()
        self.router = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList([GatedMamba(d_model) for _ in range(num_experts)])
        self.top_k = top_k

    def forward(self, x):
        scores = self.router(x)
        topk = torch.topk(scores, self.top_k, dim=-1)

        weights = F.softmax(topk.values, dim=-1)

        output = 0
        for i in range(self.top_k):
            expert_idx = topk.indices[..., i]
            expert_out = torch.stack([
                self.experts[idx](x[j:j+1])
                for j, idx in enumerate(expert_idx[:, 0])
            ])
            output += weights[..., i:i+1] * expert_out.squeeze(1)

        return output

# =========================
# MODEL
# =========================
class Model(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.embed = nn.Embedding(vocab, CONFIG["d_model"])

        self.layers = nn.ModuleList([
            MoE(CONFIG["d_model"], CONFIG["num_experts"], CONFIG["top_k"])
            for _ in range(CONFIG["n_layers"])
        ])

        self.norm = nn.LayerNorm(CONFIG["d_model"])
        self.head = nn.Linear(CONFIG["d_model"], vocab)

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        return self.head(self.norm(x))

# =========================
# DATA PREP
# =========================
texts = load_mixed_data(CONFIG["max_texts"])

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

def collate_fn(batch):
    x = torch.tensor([i[0] for i in batch])
    y = torch.tensor([i[1] for i in batch])
    return x, y

train_data = encode(texts)
loader = DataLoader(train_data, batch_size=CONFIG["batch_size"], shuffle=True, collate_fn=collate_fn)

# =========================
# INIT
# =========================
model = Model(vocab_size).to(device)
opt = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
scaler = GradScaler()

# =========================
# TRAIN
# =========================
def train():
    model.train()
    for epoch in range(CONFIG["epochs"]):
        print("\nEpoch", epoch)

        for step, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)

            with autocast():
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            if step % 50 == 0:
                print("Step", step, "Loss", loss.item())

# =========================
# GENERATION
# =========================
def generate(prompt, max_len=80, temperature=0.7):
    model.eval()

    tokens = torch.tensor(sp.encode(prompt)).unsqueeze(0).to(device)

    for _ in range(max_len):
        with torch.no_grad():
            logits = model(tokens)[:, -1] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)

        tokens = torch.cat([tokens, next_token], dim=1)

    return sp.decode(tokens[0].tolist())

# =========================
# RUN
# =========================
train()

print("\n=== TEST ===")
print(generate("User: What is AI?\nAssistant:"))
print(generate("User: Solve 12+45\nAssistant:"))