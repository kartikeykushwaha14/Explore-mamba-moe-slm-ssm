"""
test_model.py  —  Interactive test for your trained Mamba-MoE model
===================================================================
Run AFTER training is done and mamba_checkpoint.pt exists.

Usage:
    python test_model.py                  # interactive chat mode
    python test_model.py --auto           # run built-in test prompts
    python test_model.py --temp 0.5       # lower temp = more focused answers
    python test_model.py --top_p 0.8      # tighter nucleus sampling
"""

import torch
import torch.nn.functional as F
import argparse, os, sys

# ── import model from your main file ──────────────────────────────────────────
# make sure mamba_model.py is in the same folder
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mamba_modelv3 import (
    MambaMoE, load_tokenizer, CHECKPOINT, SP_MODEL, CONFIG, SYSTEM, device
)

# ══════════════════════════════════════════════════════════════════════════════
#  LOAD MODEL
# ══════════════════════════════════════════════════════════════════════════════
def load_model():
    if not os.path.exists(CHECKPOINT):
        print(f"ERROR: No checkpoint found at '{CHECKPOINT}'")
        print("Train first:  python mamba_model.py")
        sys.exit(1)

    if not os.path.exists(SP_MODEL):
        print(f"ERROR: No tokenizer found at '{SP_MODEL}'")
        sys.exit(1)

    sp         = load_tokenizer()
    vocab_size = sp.get_piece_size()

    model = MambaMoE(vocab_size, CONFIG).to(device)
    ck    = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(ck["state"] if "state" in ck else ck["model"])
    model.eval()

    epoch = ck.get("epoch", "?")
    step  = ck.get("step",  "?")
    print(f"Model loaded  —  checkpoint epoch={epoch}, step={step}")
    print(f"Parameters   : {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    print(f"Vocab size   : {vocab_size}")
    print(f"Device       : {device}\n")

    return model, sp

# ══════════════════════════════════════════════════════════════════════════════
#  GENERATION  (same nucleus sampling as mamba_model.py)
# ══════════════════════════════════════════════════════════════════════════════
# @torch.no_grad()
# def generate(model, sp, prompt: str,
#              max_new: int   = 200,
#              temperature: float = 0.7,
#              top_p: float   = 0.9) -> str:
#     ids    = [sp.bos_id()] + sp.encode(prompt)
#     tokens = torch.tensor(ids, device=device).unsqueeze(0)
#
#     for _ in range(max_new):
#         ctx    = tokens[:, -CONFIG["seq_len"]:]
#         logits = model(ctx)[:, -1] / max(temperature, 1e-6)
#         probs  = F.softmax(logits, dim=-1)
#
#         # nucleus filter
#         sorted_p, sorted_i = torch.sort(probs, descending=True)
#         cum_p   = torch.cumsum(sorted_p, dim=-1)
#         remove  = (cum_p - sorted_p) > top_p
#         sorted_p[remove] = 0.0
#         sorted_p /= sorted_p.sum()
#
#         next_tok = sorted_i[torch.multinomial(sorted_p, 1)]
#         tokens   = torch.cat([tokens, next_tok.unsqueeze(0)], dim=1)
#
#         if next_tok.item() == sp.eos_id():
#             break
#
#     return sp.decode(tokens[0].tolist())

@torch.no_grad()
def generate(model, sp, prompt: str,
             max_new: int        = 150,
             temperature: float  = 0.2,    # lowered from 0.35
             top_p: float        = 0.85,
             rep_penalty: float  = 1.3) -> str:  # NEW — penalize repeats
    model.eval()
    ids    = [sp.bos_id()] + sp.encode(prompt)
    tokens = torch.tensor(ids, device=device).unsqueeze(0)

    stop_strings  = ["User:", "Assistant:", "\nUser", "\nAssistant"]
    generated_ids = []

    for _ in range(max_new):
        ctx    = tokens[:, -CONFIG["seq_len"]:]
        logits = model(ctx)[:, -1, :].squeeze(0) / max(temperature, 1e-6)

        # ── repetition penalty ──────────────────────────────────
        # any token already generated gets its logit divided by rep_penalty
        # making it less likely to be picked again
        if generated_ids:
            seen = torch.tensor(list(set(generated_ids)), device=device)
            logits[seen] = logits[seen] / rep_penalty
        # ────────────────────────────────────────────────────────

        probs  = F.softmax(logits, dim=-1)

        # top-p nucleus filter
        sorted_p, sorted_i = torch.sort(probs, descending=True)
        cum_p  = torch.cumsum(sorted_p, dim=-1)
        remove = (cum_p - sorted_p) > top_p
        sorted_p[remove] = 0.0
        sorted_p /= sorted_p.sum()

        sampled_pos     = torch.multinomial(sorted_p, 1)
        next_tok_id     = sorted_i[sampled_pos].item()
        next_tok_tensor = torch.tensor([[next_tok_id]], device=device)
        tokens          = torch.cat([tokens, next_tok_tensor], dim=1)

        if next_tok_id == sp.eos_id():
            break

        generated_ids.append(next_tok_id)

        # stop at new turn
        current_text = sp.decode(generated_ids)
        if any(s in current_text for s in stop_strings):
            for s in stop_strings:
                if s in current_text:
                    current_text = current_text[:current_text.index(s)]
            return current_text.strip()

    return sp.decode(generated_ids).strip()

def build_prompt(user_input: str) -> str:
    """Wrap user input in the exact format the model was trained on."""
    return f"{SYSTEM}User: {user_input.strip()}\nAssistant:"

def build_math_prompt(user_input: str) -> str:
    """For math / reasoning questions, add the CoT trigger."""
    return f"{SYSTEM}User: {user_input.strip()}\nAssistant: Let's think step by step.\n"

# ══════════════════════════════════════════════════════════════════════════════
#  AUTO TEST  —  runs a fixed set of prompts covering all data types
# ══════════════════════════════════════════════════════════════════════════════
AUTO_TESTS = [
    # (label, prompt_type, question)
    ("factual — AI",        "chat",  "What is artificial intelligence?"),
    ("factual — camels",    "chat",  "Why can camels survive for long without water?"),
    ("factual — colors",    "chat",  "What are the three primary colors?"),
    ("factual — atom",      "chat",  "Describe the structure of an atom."),
    ("factual — Virgin AU", "chat",  "When did Virgin Australia start operating?"),
    ("math — clips",        "math",  "Natalia sold clips to 48 of her friends in April, "
                                     "and then she sold half as many clips in May. "
                                     "How many clips did Natalia sell altogether?"),
    ("math — babysit",      "math",  "Weng earns $12 an hour for babysitting. "
                                     "She did 50 minutes of babysitting. How much did she earn?"),
    ("math — wallet",       "math",  "Betty needs $100 for a wallet. She has half the money. "
                                     "Her parents give $15, grandparents give twice that. "
                                     "How much more does she need?"),
    ("instruction — tips",  "chat",  "Give three tips for staying healthy."),
    ("instruction — outer", "chat",  "Convert the phrase 'outer space' into a complete sentence."),
]

def run_auto_tests(model, sp, temperature: float, top_p: float):
    print("=" * 60)
    print("  AUTO TEST SUITE")
    print("=" * 60)
    for label, ptype, question in AUTO_TESTS:
        if ptype == "math":
            prompt = build_math_prompt(question)
        else:
            prompt = build_prompt(question)

        answer = generate(model, sp, prompt,
                          temperature=temperature, top_p=top_p)
        # strip the prompt from the output
        answer = answer.replace(prompt, "").strip()
        if not answer:
            answer = "(empty — model may need more training)"

        print(f"\n[{label}]")
        print(f"  Q : {question[:80]}")
        print(f"  A : {answer[:300]}")
        print("  " + "─" * 55)

    print("\nAuto test complete.")

# ══════════════════════════════════════════════════════════════════════════════
#  INTERACTIVE CHAT MODE
# ══════════════════════════════════════════════════════════════════════════════
def run_interactive(model, sp, temperature: float, top_p: float):
    print("=" * 60)
    print("  MAMBA-MOE INTERACTIVE TEST")
    print("  type 'quit' to exit")
    print("  type 'math: <question>' for step-by-step math mode")
    print("  type 'temp <0.1–1.5>' to change temperature")
    print("  type 'top_p <0.1–1.0>' to change top_p")
    print("=" * 60)

    while True:
        try:
            user_in = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_in:
            continue
        if user_in.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        # live parameter adjustment
        if user_in.lower().startswith("temp "):
            try:
                temperature = float(user_in.split()[1])
                print(f"  temperature set to {temperature}")
            except ValueError:
                print("  invalid value")
            continue

        if user_in.lower().startswith("top_p "):
            try:
                top_p = float(user_in.split()[1])
                print(f"  top_p set to {top_p}")
            except ValueError:
                print("  invalid value")
            continue

        # math mode trigger
        if user_in.lower().startswith("math:"):
            question = user_in[5:].strip()
            prompt   = build_math_prompt(question)
        else:
            prompt = build_prompt(user_in)

        print("\nAssistant: ", end="", flush=True)
        answer = generate(model, sp, prompt,
                          temperature=temperature, top_p=top_p)
        # strip the injected prompt prefix
        answer = answer.replace(prompt, "").strip()
        if not answer:
            answer = "(no output — model may need more training)"
        print(answer)

        # quick stats
        tokens_out = len(sp.encode(answer))
        print(f"\n  [{tokens_out} tokens | temp={temperature} | top_p={top_p}]")

# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the trained Mamba-MoE model")
    parser.add_argument("--auto",        action="store_true",
                        help="Run the built-in auto test suite instead of interactive chat")
    parser.add_argument("--temp",        type=float, default=0.2,
                        help="Sampling temperature (default: 0.2)")
    parser.add_argument("--top_p",       type=float, default=0.9,
                        help="Nucleus sampling top-p (default: 0.9)")
    parser.add_argument("--max_new",     type=int,   default=200,
                        help="Max new tokens per response (default: 200)")
    args = parser.parse_args()

    model, sp = load_model()

    if args.auto:
        run_auto_tests(model, sp, args.temp, args.top_p)
    else:
        run_interactive(model, sp, args.temp, args.top_p)
