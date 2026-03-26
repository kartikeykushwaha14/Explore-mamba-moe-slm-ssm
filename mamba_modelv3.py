"""
Mamba-MoE Language Model  —  Complete Final Version
=====================================================
Architecture:
  - Real SSM (Mamba block): learnable A, B, C, Delta + ZOH discretization
  - Depthwise Conv1d for local context mixing
  - SwiGLU FFN experts inside MoE
  - Per-token MoE routing with load-balance auxiliary loss
  - RMSNorm throughout
  - Weight-tied embedding + LM head
  - Top-p nucleus sampling for generation

Training:
  - Mixed precision AMP (GradScaler)
  - AdamW + cosine LR with linear warmup
  - Gradient clipping
  - Checkpoint save/resume

Data (based on actual samples):
  - WikiText-2  : raw Wikipedia prose  (cleaned of = headings = and @-@ artifacts)
  - Dolly-15k   : "User: ... Assistant: ..." instruction format
  - GSM8K       : "User: ... Assistant: Let's think step by step. ..." math reasoning
  - Alpaca      : "User: ... Assistant: ..." instruction format
  - All unified under a single SYSTEM_PROMPT prefix

Datasets:
  1. Dolly-15k     : open_qa, general_qa, brainstorming, creative_writing, classification
  2. GSM8K         : math reasoning, <<calc>> cleaned, #### replaced
  3. Alpaca        : instruction following, instruction+input merged
  4. OpenAssistant : quality filtered (rank=0, quality>=0.5, toxicity<0.3)
  5. ELI5          : Pavithree/eli5, best scored answer, len 100-1000
  6. SciQ          : science Q&A, support+correct_answer combined
  7. Everyday Conv : all 2260 samples, all turn pairs extracted (8625 pairs)

Fixes over v4:
  - load_all_data fully rewritten with proper per-dataset processing
  - OASST labels parsed as parallel lists (name/value zip)
  - ELI5 uses title + answers.text[best_score]
  - FLAN removed (useless tasks)
  - Camel removed (gone from Hub)
  - clean_gsm() removes <<>> and #### artifacts
  - Dolly category filtering (drop closed_qa, info_extraction, summarization)
  - Alpaca output filter > 30 chars, instruction+input merged
  - LMDataset: no cross-boundary mixing (truncate per sample)
  - SYSTEM_MSG: friendly + conversational tone added
  - Total cap removed (each dataset self-caps during loading)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import math, os, random, re

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ══════════════════════════════════════════════════════
#  DEVICE
# ══════════════════════════════════════════════════════
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device : {device}")
if device == "cuda":
    props = torch.cuda.get_device_properties(0)
    print(f"GPU    : {props.name}")
    print(f"VRAM   : {props.total_memory / 1e9:.1f} GB")

# ══════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════
# use this if you have Good vram
CONFIG = {
    # data — balanced for good factual + conversational learning
    "max_dolly"    : 3000,   # 1500 → 3000  : more factual Q&A
    "max_gsm"      : 1500,   # keep          : math is already good
    "max_alpaca"   : 3000,   # 1500 → 3000  : more instructions
    "max_oasst"    : 2000,   # 1000 → 2000  : more quality conversations
    "max_eli5"     : 2000,   # 1000 → 2000  : more explanations
    "max_sciq"     : 2000,   # 1000 → 2000  : more science facts
    "max_everyday" : 9000,   # 2500 → 9000  : captures all 8625 pairs

    # model — unchanged, already good
    "seq_len"      : 256,    # 256 → 128    : halves training time, same quality
    "batch_size"   : 4,
    "d_model"      : 320,
    "d_state"      : 32,
    "d_conv"       : 4,
    "expand"       : 2,
    "n_layers"     : 6,
    "num_experts"  : 4,
    "top_k"        : 2,
    "moe_aux_w"    : 0.02,

    # training — tuned for larger dataset
    "epochs"       : 4,
    "lr"           : 2e-4,
    "warmup_steps" : 500,
    "grad_clip"    : 1.0,
    "weight_decay" : 0.05,
    "vocab_size"   : 6000,
}

CHECKPOINT = "mamba_checkpoint.pt"
SP_MODEL   = "sp.model"

# ══════════════════════════════════════════════════════
#  SYSTEM MESSAGE — friendly + conversational
# ══════════════════════════════════════════════════════
SYSTEM_MSG = (
    "You are a helpful and friendly AI assistant. "
    "You greet users warmly, engage in casual conversation, "
    "and give clear accurate answers when asked questions. "
    "For math, show step-by-step working."
)

# ══════════════════════════════════════════════════════
#  FORMAT HELPERS
# ══════════════════════════════════════════════════════
def fmt_chat(user: str, assistant: str) -> str:
    return (
        f"<|system|>{SYSTEM_MSG}<|end|>"
        f"<|user|>{user.strip()}<|end|>"
        f"<|assistant|>{assistant.strip()}<|end|>"
    )

def fmt_math(question: str, answer: str) -> str:
    return (
        f"<|system|>{SYSTEM_MSG}<|end|>"
        f"<|user|>{question.strip()}<|end|>"
        f"<|assistant|>Let's think step by step.\n{answer.strip()}<|end|>"
    )

# ══════════════════════════════════════════════════════
#  GSM8K CLEANER
#  removes <<48/2=24>> calculator annotations
#  replaces #### 72 with "The answer is 72."
# ══════════════════════════════════════════════════════
def clean_gsm(answer: str) -> str:
    answer = re.sub(r"<<.*?>>", "", answer)
    answer = re.sub(r"####\s*(\S+)", r"The answer is \1.", answer)
    answer = re.sub(r" +", " ", answer).strip()
    return answer

# ══════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════
def load_all_data(cfg: dict) -> list:
    from datasets import load_dataset
    import requests

    texts = []

    # ── 1. DOLLY-15K ──────────────────────────────────
    # Keep: open_qa, general_qa, brainstorming,
    #       creative_writing, classification (len>20)
    # Drop: closed_qa, information_extraction, summarization
    #       (require context paragraph — useless without it)
    DOLLY_KEEP = {
        "open_qa", "general_qa", "brainstorming", "creative_writing"
    }
    print("Loading Dolly-15k ...")
    try:
        dolly = load_dataset("databricks/databricks-dolly-15k", split="train")
        count = 0
        for row in dolly:
            cat      = row.get("category", "")
            instr    = row.get("instruction", "").strip()
            response = row.get("response",    "").strip()
            if not instr or not response:
                continue
            if cat == "classification":
                if len(response) < 20:
                    continue
                texts.append(fmt_chat(instr, response))
                count += 1
            elif cat in DOLLY_KEEP:
                if len(response) < 20 or len(response) > 1200:
                    continue
                texts.append(fmt_chat(instr, response))
                count += 1
            if count >= cfg["max_dolly"]:
                break
        print(f"  dolly    : {count:,} samples")
    except Exception as e:
        print(f"  dolly failed: {e}")

    # ── 2. GSM8K ──────────────────────────────────────
    # clean_gsm() removes <<calc>> and #### markers
    print("Loading GSM8K ...")
    try:
        gsm   = load_dataset("gsm8k", "main", split="train")
        count = 0
        for row in gsm:
            q = row.get("question", "").strip()
            a = row.get("answer",   "").strip()
            if not q or not a:
                continue
            texts.append(fmt_math(q, clean_gsm(a)))
            count += 1
            if count >= cfg["max_gsm"]:
                break
        print(f"  gsm8k    : {count:,} samples")
    except Exception as e:
        print(f"  gsm8k failed: {e}")

    # ── 3. ALPACA ─────────────────────────────────────
    # Merge instruction + input into one question
    # Drop outputs < 30 chars (trivial one-word answers)
    print("Loading Alpaca ...")
    try:
        url  = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
        data = requests.get(url, timeout=30).json()
        count = 0
        for item in data:
            instr  = item.get("instruction", "").strip()
            inp    = item.get("input",       "").strip()
            output = item.get("output",      "").strip()
            if not instr or not output:
                continue
            if len(output) < 30 or len(output) > 1200:
                continue
            question = f"{instr}\n{inp}" if inp else instr
            texts.append(fmt_chat(question, output))
            count += 1
            if count >= cfg["max_alpaca"]:
                break
        print(f"  alpaca   : {count:,} samples")
    except Exception as e:
        print(f"  alpaca failed: {e}")

    # ── 4. OPENASSISTANT ──────────────────────────────
    # labels stored as parallel lists {name:[...], value:[...]}
    # filters: lang=en, not deleted, rank=0,
    #          quality>=0.5, toxicity<0.3, spam=0
    print("Loading OpenAssistant ...")
    try:
        oasst = load_dataset("OpenAssistant/oasst1", split="train")
        id_to_row = {}
        for row in oasst:
            if row.get("message_id"):
                id_to_row[row["message_id"]] = row
        count = 0
        for row in oasst:
            if row.get("role")    != "assistant": continue
            if row.get("lang")    != "en":        continue
            if row.get("deleted") == True:        continue
            if row.get("rank", 1) != 0:           continue
            text = row.get("text", "").strip()
            if not text:                            continue
            if len(text) < 20 or len(text) > 1200: continue

            # parse parallel list label structure
            labels    = row.get("labels", {})
            names     = labels.get("name",  [])
            values    = labels.get("value", [])
            label_map = dict(zip(names, values))

            if label_map.get("quality",         0)  < 0.5: continue
            if label_map.get("toxicity",        1) >= 0.3: continue
            if label_map.get("spam",            1)  > 0.0: continue
            if label_map.get("not_appropriate", 1)  > 0.0: continue

            parent      = id_to_row.get(row.get("parent_id", ""), {})
            parent_text = parent.get("text", "").strip()
            if not parent_text: continue

            texts.append(fmt_chat(parent_text, text))
            count += 1
            if count >= cfg["max_oasst"]:
                break
        print(f"  oasst    : {count:,} samples")
    except Exception as e:
        print(f"  oasst failed: {e}")

    # ── 5. ELI5 ───────────────────────────────────────
    # Pavithree/eli5 — modern Parquet format
    # fields: title, answers.text[], answers.score[]
    # Keep: len 100-1000, not starting with I/Me/My
    print("Loading ELI5 ...")
    try:
        eli5  = load_dataset("Pavithree/eli5", split="train")
        count = 0
        for row in eli5:
            question   = row.get("title", "").strip()
            answers    = row.get("answers", {})
            texts_list = answers.get("text",  [])
            scores     = answers.get("score", [])
            if not question or not texts_list:
                continue
            best = texts_list[scores.index(max(scores))] if scores else texts_list[0]
            best = best.strip()
            if len(best) < 100 or len(best) > 1000:
                continue
            if best.startswith(("I ", "Me ", "My ", "I'm ", "I've ")):
                continue
            texts.append(fmt_chat(question, best))
            count += 1
            if count >= cfg["max_eli5"]:
                break
        print(f"  eli5     : {count:,} samples")
    except Exception as e:
        print(f"  eli5 failed: {e}")

    # ── 6. SCIQ ───────────────────────────────────────
    # Science Q&A — combines support + correct_answer
    # into a proper explanation (not just a one-word answer)
    print("Loading SciQ ...")
    try:
        sciq  = load_dataset("allenai/sciq", split="train")
        count = 0
        for row in sciq:
            question = row.get("question",       "").strip()
            correct  = row.get("correct_answer", "").strip()
            support  = row.get("support",        "").strip()
            if not question or not correct:
                continue
            answer = f"{correct.capitalize()}. {support}" if support else correct
            if len(answer) > 1200:
                answer = answer[:1200]
            texts.append(fmt_chat(question, answer))
            count += 1
            if count >= cfg["max_sciq"]:
                break
        print(f"  sciq     : {count:,} samples")
    except Exception as e:
        print(f"  sciq failed: {e}")

    # ── 7. EVERYDAY CONVERSATIONS ─────────────────────
    # ALL 2260 conversations, every user→assistant turn extracted
    # built by HuggingFace specifically for small LLMs
    # fixes: "hii", "how are you", "who are you", casual greetings
    print("Loading Everyday Conversations ...")
    try:
        everyday = load_dataset(
            "HuggingFaceTB/everyday-conversations-llama3.1-2k",
            split="train_sft"
        )
        count = 0
        for row in everyday:
            messages = row.get("messages", [])
            for i in range(len(messages) - 1):
                if (messages[i]["role"]   == "user" and
                    messages[i+1]["role"] == "assistant"):
                    user = messages[i]["content"].strip()
                    asst = messages[i+1]["content"].strip()
                    if not user or not asst:
                        continue
                    if len(asst) > 800:
                        continue
                    texts.append(fmt_chat(user, asst))
                    count += 1
        print(f"  everyday : {count:,} samples (all turns extracted)")
    except Exception as e:
        print(f"  everyday failed: {e}")

    # ── shuffle — no cap, each dataset self-capped ────
    random.shuffle(texts)
    print(f"\n  TOTAL    : {len(texts):,} samples")
    return texts

# ══════════════════════════════════════════════════════
#  TOKENIZER
# ══════════════════════════════════════════════════════
def train_tokenizer(texts: list, vocab_size: int):
    import sentencepiece as spm
    tmp = "spm_train_tmp.txt"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\n".join(texts))
    spm.SentencePieceTrainer.train(
        input              = tmp,
        model_prefix       = "sp",
        vocab_size         = vocab_size,
        character_coverage = 0.9995,
        model_type         = "bpe",
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        user_defined_symbols = [
            "<|system|>", "<|user|>", "<|assistant|>", "<|end|>"
        ],
    )
    os.remove(tmp)
    print(f"Tokenizer trained  vocab_size={vocab_size}")

def load_tokenizer():
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.load(SP_MODEL)
    return sp

def get_end_token_id(sp) -> int:
    # piece_to_id gives exact match for user_defined_symbols
    eid = sp.piece_to_id("<|end|>")
    if eid != sp.unk_id():
        return eid
    ids = sp.encode("<|end|>")
    return ids[0] if ids else sp.eos_id()

# ══════════════════════════════════════════════════════
#  DATASET
#  one sample per chunk — no cross-boundary mixing
# ══════════════════════════════════════════════════════
class LMDataset(Dataset):
    def __init__(self, texts: list, sp, seq_len: int):
        self.seq_len = seq_len
        self.samples = []
        pad_id = sp.pad_id()
        end_id = get_end_token_id(sp)

        for text in texts:
            ids = sp.encode(text) + [end_id]
            if len(ids) < 4:
                continue
            if len(ids) > seq_len + 1:
                ids = ids[:seq_len + 1]
            if len(ids) < seq_len + 1:
                ids = ids + [pad_id] * (seq_len + 1 - len(ids))
            self.samples.append(ids)

        print(f"LMDataset: {len(self.samples):,} training chunks")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        c = self.samples[idx]
        return (torch.tensor(c[:-1], dtype=torch.long),
                torch.tensor(c[1:],  dtype=torch.long))

# ══════════════════════════════════════════════════════
#  MODEL COMPONENTS
# ══════════════════════════════════════════════════════
class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.g   = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.g


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        d_inner      = d_model * expand
        self.d_inner = d_inner
        self.d_state = d_state
        self.norm     = RMSNorm(d_model)
        self.in_proj  = nn.Linear(d_model, 2 * d_inner, bias=False)
        self.out_proj = nn.Linear(d_inner, d_model,     bias=False)
        self.conv1d   = nn.Conv1d(
            d_inner, d_inner, kernel_size=d_conv,
            padding=d_conv - 1, groups=d_inner, bias=True,
        )
        self.x_proj  = nn.Linear(d_inner, 2 * d_state + 1, bias=False)
        self.dt_proj = nn.Linear(1, d_inner, bias=True)
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(
            torch.log(A).unsqueeze(0).expand(d_inner, -1).clone()
        )
        self.D = nn.Parameter(torch.ones(d_inner))
        nn.init.constant_(self.dt_proj.bias, math.log(0.001))

    def forward(self, x):
        B, T, _ = x.shape
        residual = x
        x        = self.norm(x)
        xz       = self.in_proj(x)
        x_s, z   = xz.chunk(2, dim=-1)
        xc = self.conv1d(x_s.transpose(1, 2))[..., :T]
        xc = F.silu(xc.transpose(1, 2))
        params  = self.x_proj(xc)
        B_coef  = params[..., :self.d_state]
        C_coef  = params[..., self.d_state:2*self.d_state]
        log_dt  = params[..., -1:]
        dt      = F.softplus(self.dt_proj(log_dt))
        A       = -torch.exp(self.A_log.float())
        dt_e    = dt.unsqueeze(-1)
        A_bar   = torch.exp(dt_e * A)
        B_bar   = dt_e * B_coef.unsqueeze(2)
        Bu      = B_bar * xc.unsqueeze(-1)
        h  = torch.zeros(B, self.d_inner, self.d_state,
                         device=x.device, dtype=x.dtype)
        ys = []
        for t in range(T):
            h  = A_bar[:, t] * h + Bu[:, t]
            yt = (h * C_coef[:, t].unsqueeze(1)).sum(-1)
            ys.append(yt)
        y = torch.stack(ys, dim=1)
        y = y + xc * self.D
        y = y * F.silu(z)
        return residual + self.out_proj(y)


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model, expand=4):
        super().__init__()
        d_ff      = d_model * expand
        self.norm = RMSNorm(d_model)
        self.w1   = nn.Linear(d_model, d_ff, bias=False)
        self.w2   = nn.Linear(d_ff, d_model, bias=False)
        self.w3   = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x):
        h = self.norm(x)
        return x + self.w2(F.silu(self.w1(h)) * self.w3(h))


class MoELayer(nn.Module):
    def __init__(self, d_model, num_experts, top_k):
        super().__init__()
        self.num_experts = num_experts
        self.top_k       = top_k
        self.router      = nn.Linear(d_model, num_experts, bias=False)
        self.experts     = nn.ModuleList(
            [SwiGLUFFN(d_model) for _ in range(num_experts)]
        )
        self.norm = RMSNorm(d_model)

    def forward(self, x):
        B, T, D   = x.shape
        h_flat    = self.norm(x).reshape(B * T, D)
        scores    = self.router(h_flat)
        topk_vals, topk_idx = torch.topk(scores, self.top_k, dim=-1)
        weights   = F.softmax(topk_vals, dim=-1)
        gate_probs = F.softmax(scores, dim=-1)
        dispatch   = torch.zeros(B * T, self.num_experts, device=x.device)
        dispatch.scatter_(1, topk_idx, 1.0)
        self.aux_loss = self.num_experts * (
            gate_probs.mean(0) * dispatch.mean(0)
        ).sum()
        out    = torch.zeros(B * T, D, device=x.device, dtype=x.dtype)
        x_flat = x.reshape(B * T, D)
        for e in range(self.num_experts):
            mask    = (topk_idx == e).any(dim=-1)
            tok_ids = mask.nonzero(as_tuple=True)[0]
            if tok_ids.numel() == 0:
                continue
            tok_in  = x_flat[tok_ids].unsqueeze(0)
            tok_out = self.experts[e](tok_in).squeeze(0)
            w = (weights[tok_ids] * (topk_idx[tok_ids] == e).float()).sum(-1)
            out[tok_ids] += w.unsqueeze(-1) * tok_out
        return out.reshape(B, T, D)


class MambaMoELayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mamba = MambaBlock(
            d_model = cfg["d_model"],
            d_state = cfg["d_state"],
            d_conv  = cfg["d_conv"],
            expand  = cfg["expand"],
        )
        self.moe = MoELayer(
            d_model     = cfg["d_model"],
            num_experts = cfg["num_experts"],
            top_k       = cfg["top_k"],
        )

    def forward(self, x):
        x = self.mamba(x)
        x = self.moe(x)
        return x


class MambaMoE(nn.Module):
    def __init__(self, vocab_size, cfg):
        super().__init__()
        D = cfg["d_model"]
        self.embed  = nn.Embedding(vocab_size, D, padding_idx=0)
        self.drop   = nn.Dropout(0.1)
        self.layers = nn.ModuleList(
            [MambaMoELayer(cfg) for _ in range(cfg["n_layers"])]
        )
        self.norm = RMSNorm(D)
        self.head = nn.Linear(D, vocab_size, bias=False)
        self.head.weight = self.embed.weight
        self._init_weights()
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Parameters : {n/1e6:.2f}M")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, x, return_aux=False):
        x   = self.drop(self.embed(x))
        aux = torch.tensor(0.0, device=x.device)
        for layer in self.layers:
            x   = layer(x)
            aux = aux + layer.moe.aux_loss
        logits = self.head(self.norm(x))
        return (logits, aux) if return_aux else logits

# ══════════════════════════════════════════════════════
#  LR SCHEDULE
# ══════════════════════════════════════════════════════
def get_lr(step, warmup, total, base_lr):
    if step < warmup:
        return base_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))

# ══════════════════════════════════════════════════════
#  TRAINING LOOP
# ══════════════════════════════════════════════════════
def train(model, loader, vocab_size, cfg, start_epoch=0, start_step=0):
    total_steps = cfg["epochs"] * len(loader)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr           = cfg["lr"],
        weight_decay = cfg["weight_decay"],
        betas        = (0.9, 0.95),
        eps          = 1e-8,
    )
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None
    step   = start_step

    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        running_ce = 0.0
        print(f"\n{'━'*55}")
        print(f"  Epoch {epoch+1} / {cfg['epochs']}   ({len(loader)} steps/epoch)")
        print(f"{'━'*55}")
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}", ncols=90) if HAS_TQDM else loader

        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            lr   = get_lr(step, cfg["warmup_steps"], total_steps, cfg["lr"])
            for g in opt.param_groups:
                g["lr"] = lr

            with torch.amp.autocast("cuda") if device == "cuda" else open(os.devnull):
                logits, aux = model(x, return_aux=True)
                ce   = F.cross_entropy(
                    logits.reshape(-1, vocab_size),
                    y.reshape(-1),
                    ignore_index=0,
                )
                loss = ce + cfg["moe_aux_w"] * aux

            opt.zero_grad()
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                opt.step()

            running_ce += ce.item()
            step       += 1

            if HAS_TQDM:
                ppl = math.exp(min(ce.item(), 20))
                pbar.set_postfix({
                    "loss": f"{ce.item():.4f}",
                    "ppl" : f"{ppl:.1f}",
                    "lr"  : f"{lr:.1e}",
                })
            else:
                if step % 50 == 0:
                    ppl = math.exp(min(ce.item(), 20))
                    print(f"  step {step:5d}  loss {ce.item():.4f}  ppl {ppl:.1f}  lr {lr:.2e}")

            if step % 500 == 0:
                _save(model, epoch, step, cfg)

        avg = running_ce / len(loader)
        print(f"\n  Epoch {epoch+1} done — avg_loss={avg:.4f}  ppl={math.exp(min(avg,20)):.1f}")
        _save(model, epoch + 1, step, cfg)

# ══════════════════════════════════════════════════════
#  CHECKPOINT
# ══════════════════════════════════════════════════════
def _save(model, epoch, step, cfg):
    torch.save({
        "model" : model.state_dict(),
        "epoch" : epoch,
        "step"  : step,
        "cfg"   : cfg,
    }, CHECKPOINT)
    print(f"  ✓ checkpoint saved  epoch={epoch}  step={step}")

def maybe_resume(model):
    if not os.path.exists(CHECKPOINT):
        return 0, 0
    ck = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(ck["model"])
    print(f"  ✓ resumed  epoch={ck['epoch']}  step={ck['step']}")
    return ck["epoch"], ck["step"]

# ══════════════════════════════════════════════════════
#  GENERATION
# ══════════════════════════════════════════════════════
@torch.no_grad()
def generate(model, sp, prompt: str,
             max_new: int       = 60,
             temperature: float = 0.2,
             top_p: float       = 0.5,
             rep_penalty: float = 2.0) -> str:
    model.eval()
    end_id    = get_end_token_id(sp)
    # also stop at any structural token to prevent runaway generation
    stop_ids  = {
        end_id,
        sp.eos_id(),
        sp.piece_to_id("<|user|>"),
        sp.piece_to_id("<|system|>"),
        sp.piece_to_id("<|assistant|>"),
    }
    ids       = sp.encode(prompt)
    tokens    = torch.tensor(ids, device=device).unsqueeze(0)
    generated = []

    for _ in range(max_new):
        ctx    = tokens[:, -CONFIG["seq_len"]:]
        # squeeze to 1D — shape (vocab_size,)
        logits = model(ctx)[0, -1].float()

        # repetition penalty — logits is 1D here
        if rep_penalty != 1.0 and generated:
            for gid in set(generated):
                if logits[gid] > 0:
                    logits[gid] /= rep_penalty
                else:
                    logits[gid] *= rep_penalty

        logits = logits / max(temperature, 1e-6)
        # probs is 1D shape (vocab_size,)
        probs  = F.softmax(logits, dim=-1)

        # top-p nucleus filter on 1D tensor
        sorted_p, sorted_i = torch.sort(probs, descending=True)
        cum_p  = torch.cumsum(sorted_p, dim=-1)
        remove = (cum_p - sorted_p) > top_p
        sorted_p = sorted_p.clone()
        sorted_p[remove] = 0.0
        if sorted_p.sum() <= 0:
            sorted_p = torch.ones_like(sorted_p)
        sorted_p /= sorted_p.sum()

        # sample — multinomial on 1D returns 1D tensor of length 1
        sample_idx = torch.multinomial(sorted_p, num_samples=1).item()  # plain int
        tok_id     = sorted_i[sample_idx].item()                         # plain int

        if tok_id in stop_ids:
            break

        tokens = torch.cat([tokens, torch.tensor([[tok_id]], device=device)], dim=1)
        generated.append(tok_id)

    text = sp.decode(generated).strip()

    # strip leaked special tokens
    for tok in ["<|end|>", "<|user|>", "<|assistant|>", "<|system|>"]:
        if tok in text:
            text = text[:text.index(tok)].strip()

    # cut at first sentence boundary
    for punct in [". ", "! ", "? "]:
        idx = text.find(punct)
        if 20 < idx < 250:
            text = text[:idx + 1].strip()
            break

    # if still looks like gibberish (no vowels ratio high, numbers mid-sentence)
    import re
    words = text.split()
    if len(words) > 3:
        # remove responses that are mostly numbers/symbols
        num_count = sum(1 for w in words if re.search(r"[0-9]", w))
        if num_count > len(words) * 0.4:
            text = ""

    return text if text else "I'm not sure about that. Could you rephrase your question?"

# ══════════════════════════════════════════════════════
#  CHAT
# ══════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════
#  MATH SOLVER  — handles arithmetic without the model
#  supports: basic ops, percentages, squares, square roots,
#            powers, word-form numbers, unit conversions
# ══════════════════════════════════════════════════════
def try_solve_math(text: str):
    import re, math

    t = text.strip().lower()

    # ── 1. pure expression: 2+2, (3*4)+5, 10/2, 2**8 ──
    # only allow safe characters
    expr = re.sub(r"[^0-9+\-*/().% ]", "", t).strip()
    if expr and re.search(r"[0-9]", expr) and re.search(r"[+\-*/]", expr):
        try:
            result = eval(expr, {"__builtins__": {}})
            if isinstance(result, float) and result.is_integer():
                result = int(result)
            return f"{expr.strip()} = {round(result, 6)}"
        except Exception:
            pass

    # ── 2. what is X + Y / X * Y / X - Y ──────────────
    m = re.match(r"what(?:\s+is)?\s+([0-9.]+)\s*([+\-*/x×÷])\s*([0-9.]+)", t)
    if m:
        a, op, b = float(m.group(1)), m.group(2), float(m.group(3))
        op_map = {"+": a+b, "-": a-b, "*": a*b, "x": a*b, "×": a*b,
                  "/": round(a/b, 6) if b != 0 else None, "÷": round(a/b, 6) if b != 0 else None}
        r = op_map.get(op)
        if r is None:
            return "Cannot divide by zero."
        if isinstance(r, float) and r == int(r):
            r = int(r)
        op_sym = {"x": "×", "×": "×", "÷": "÷"}.get(op, op)
        return f"{int(a) if a==int(a) else a} {op_sym} {int(b) if b==int(b) else b} = {r}"

    # ── 3. calculate / compute / solve prefix ──────────
    m = re.match(r"(?:calculate|compute|solve|find)\s+(.+)", t)
    if m:
        inner = try_solve_math(m.group(1))
        if inner:
            return inner

    # ── 4. X percent of Y ──────────────────────────────
    m = re.match(r"([0-9.]+)\s*(?:percent|%)\s+of\s+([0-9.]+)", t)
    if m:
        pct, total = float(m.group(1)), float(m.group(2))
        result = round(pct / 100 * total, 6)
        if result == int(result):
            result = int(result)
        return f"{pct}% of {total} = {result}"

    # ── 5. square root of X ────────────────────────────
    m = re.match(r"(?:square root|sqrt)\s+(?:of\s+)?([0-9.]+)", t)
    if m:
        n = float(m.group(1))
        result = round(math.sqrt(n), 6)
        if result == int(result):
            result = int(result)
        return f"√{n} = {result}"

    # ── 6. X squared / X cubed ─────────────────────────
    m = re.match(r"([0-9.]+)\s+(squared|cubed|to the power of\s+([0-9]+))", t)
    if m:
        base = float(m.group(1))
        if "squared" in m.group(2):
            exp = 2
        elif "cubed" in m.group(2):
            exp = 3
        else:
            exp = int(m.group(3))
        result = base ** exp
        if isinstance(result, float) and result == int(result):
            result = int(result)
        return f"{base}^{exp} = {result}"

    # ── 7. simple word problems with numbers ───────────
    # "I have 10 apples and eat 3, how many left"
    m = re.search(r"([0-9]+)\s+(?:apples?|items?|things?|coins?|books?|balls?).*?(?:eat|lose|give away|remove|take away)\s+([0-9]+)", t)
    if m:
        total, removed = int(m.group(1)), int(m.group(2))
        return f"{total} - {removed} = {total - removed}"

    # ── 8. temperature conversion ──────────────────────
    m = re.match(r"([0-9.]+)\s*(?:degrees?\s+)?celsius\s+(?:to|in)\s+fahrenheit", t)
    if m:
        c = float(m.group(1))
        f = round(c * 9/5 + 32, 2)
        return f"{c}°C = {f}°F"

    m = re.match(r"([0-9.]+)\s*(?:degrees?\s+)?fahrenheit\s+(?:to|in)\s+celsius", t)
    if m:
        f = float(m.group(1))
        c = round((f - 32) * 5/9, 2)
        return f"{f}°F = {c}°C"

    # ── 9. km to miles / miles to km ───────────────────
    m = re.match(r"([0-9.]+)\s*km\s+(?:to|in)\s+miles", t)
    if m:
        km = float(m.group(1))
        return f"{km} km = {round(km * 0.621371, 4)} miles"

    m = re.match(r"([0-9.]+)\s*miles?\s+(?:to|in)\s+km", t)
    if m:
        mi = float(m.group(1))
        return f"{mi} miles = {round(mi * 1.60934, 4)} km"

    return None  # not a math question — pass to model


def chat(model, sp):
    print("\n" + "═"*55)
    print("  MAMBA-MOE CHAT  |  type 'quit' to exit")
    print("  Commands: 'temp 0.5' | 'rep 1.2' | 'math: <q>'")
    print("═"*55)

    temperature = 0.9
    rep_penalty = 2.0

    while True:
        try:
            user_in = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not user_in:
            continue
        if user_in.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break
        if user_in.lower().startswith("temp "):
            try:
                temperature = float(user_in.split()[1])
                print(f"  temperature → {temperature}")
            except ValueError:
                print("  usage: temp 0.5")
            continue
        if user_in.lower().startswith("rep "):
            try:
                rep_penalty = float(user_in.split()[1])
                print(f"  rep_penalty → {rep_penalty}")
            except ValueError:
                print("  usage: rep 1.3")
            continue

        # ── math interceptor — always correct ────────
        math_answer = try_solve_math(user_in)
        if math_answer is not None:
            print(f"\nAssistant: {math_answer}")
            continue

        if user_in.lower().startswith("math:"):
            question = user_in[5:].strip()
            prompt = (
                f"<|system|>{SYSTEM_MSG}<|end|>"
                f"<|user|>{question}<|end|>"
                f"<|assistant|>Let's think step by step.\n"
            )
        else:
            prompt = (
                f"<|system|>{SYSTEM_MSG}<|end|>"
                f"<|user|>{user_in}<|end|>"
                f"<|assistant|>"
            )

        answer = generate(
            model, sp, prompt,
            temperature = temperature,
            rep_penalty = rep_penalty,
        )
        print(f"\nAssistant: {answer.strip()}")

# ══════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════
if __name__ == "__main__":
    texts = load_all_data(CONFIG)

    if not os.path.exists(SP_MODEL):
        train_tokenizer(texts, CONFIG["vocab_size"])
    sp         = load_tokenizer()
    vocab_size = sp.get_piece_size()
    print(f"Vocab size : {vocab_size}")

    dataset = LMDataset(texts, sp, CONFIG["seq_len"])
    loader  = DataLoader(
        dataset,
        batch_size  = CONFIG["batch_size"],
        shuffle     = True,
        num_workers = 0,
        pin_memory  = (device == "cuda"),
        drop_last   = True,
    )
    print(f"Steps/epoch: {len(loader)}")

    model = MambaMoE(vocab_size, CONFIG).to(device)
    start_epoch, start_step = maybe_resume(model)
    train(model, loader, vocab_size, CONFIG, start_epoch, start_step)
    chat(model, sp)