import re
import random
from datasets import load_dataset
import requests

# ══════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════
CONFIG = {
    "max_dolly"     : 3000,
    "max_gsm"       : 1500,
    "max_alpaca"    : 3000,
    "max_oasst"     : 2000,
    "max_eli5"      : 2000,
    "max_sciq"      : 2000,
    "max_everyday"  : 2260,   # use ALL — only 2260 exist
    "seq_len"       : 128,
    "batch_size"    : 4,
    "d_model"       : 320,
    "d_state"       : 32,
    "d_conv"        : 4,
    "expand"        : 2,
    "n_layers"      : 6,
    "num_experts"   : 4,
    "top_k"         : 2,
    "moe_aux_w"     : 0.02,
    "epochs"        : 4,
    "lr"            : 2e-4,
    "warmup_steps"  : 500,
    "grad_clip"     : 1.0,
    "weight_decay"  : 0.05,
    "vocab_size"    : 6000,
}

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
# ══════════════════════════════════════════════════════
def clean_gsm(answer: str) -> str:
    answer = re.sub(r"<<.*?>>", "", answer)
    answer = re.sub(r"####\s*(\S+)", r"The answer is \1.", answer)
    answer = re.sub(r" +", " ", answer).strip()
    return answer

# ══════════════════════════════════════════════════════
#  FULL load_all_data
# ══════════════════════════════════════════════════════
def load_all_data(cfg: dict) -> list:
    texts = []

    # ── 1. DOLLY-15K ──────────────────────────────────
    # Keep : open_qa, general_qa, brainstorming,
    #        creative_writing, classification (len>20)
    # Drop : closed_qa, information_extraction, summarization
    #        (need context paragraph — model has none at inference)
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
    # Clean <<calc>> and #### markers before training
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
    # Drop outputs < 30 chars (single word / trivial answers)
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

    # ── 4. OPENASSISTANT ──────────────────────────────────
    print("Loading OpenAssistant ...")
    try:
        oasst = load_dataset("OpenAssistant/oasst1", split="train")
        id_to_row = {}
        for row in oasst:
            if row.get("message_id"):
                id_to_row[row["message_id"]] = row
        count = 0
        for row in oasst:
            if row.get("role") != "assistant": continue
            if row.get("lang") != "en":        continue
            if row.get("deleted") == True:        continue
            if row.get("rank", 1) != 0:           continue
            text = row.get("text", "").strip()
            if not text:                            continue
            if len(text) < 20 or len(text) > 1200: continue

            # labels are parallel lists — zip name+value to look up
            labels = row.get("labels", {})
            names = labels.get("name", [])
            values = labels.get("value", [])
            label_map = dict(zip(names, values))

            if label_map.get("quality", 0) < 0.5: continue
            if label_map.get("toxicity", 1) >= 0.3: continue
            if label_map.get("spam", 1) > 0.0: continue
            if label_map.get("not_appropriate", 1) > 0.0: continue

            parent = id_to_row.get(row.get("parent_id", ""), {})
            parent_text = parent.get("text", "").strip()
            if not parent_text: continue

            texts.append(fmt_chat(parent_text, text))
            count += 1
            if count >= cfg["max_oasst"]:
                break
        print(f"  oasst    : {count:,} samples")
    except Exception as e:
        print(f"  oasst failed: {e}")

    # ── 5. ELI5 ───────────────────────────────────────────
    print("Loading ELI5 ...")
    try:
        eli5 = load_dataset("Pavithree/eli5", split="train")
        count = 0
        for row in eli5:
            question = row.get("title", "").strip()
            answers = row.get("answers", {})
            texts_list = answers.get("text", [])
            scores = answers.get("score", [])
            if not question or not texts_list:
                continue
            # pick highest scored answer
            if scores:
                best = texts_list[scores.index(max(scores))]
            else:
                best = texts_list[0]
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
    # Science Q&A — builds proper explanation from
    # support paragraph + correct_answer combined
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
            # combine into a real answer not just one word
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
    # ALL 2260 samples used — built by HuggingFace
    # specifically to fix "hii", "how are you",
    # "who are you" in small LLMs
    # Extract every user→assistant turn pair from messages
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

    # ── shuffle + cap ─────────────────────────────────
    random.shuffle(texts)
    total = (
        cfg["max_dolly"]   + cfg["max_gsm"]  + cfg["max_alpaca"] +
        cfg["max_oasst"]   + cfg["max_eli5"] + cfg["max_sciq"]   +
        cfg["max_everyday"]
    )
    texts = texts[:total]
    print(f"\n  TOTAL    : {len(texts):,} samples")
    return texts


# ══════════════════════════════════════════════════════
#  TEST — run directly to verify all datasets load
# ══════════════════════════════════════════════════════
if __name__ == "__main__":
    texts = load_all_data(CONFIG)

    print("\n=== SAMPLE OUTPUTS ===")
    for i in [0, 100, 500, 1000, 3000, 6000, 9000]:
        if i < len(texts):
            print(f"\n[Sample {i}]")
            print(texts[i][:300])
            print("...")