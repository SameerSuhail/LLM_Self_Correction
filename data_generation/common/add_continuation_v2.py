import argparse
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

DEEPSEEK = "/scratch/user/sameersuhail/hf_cache/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-14B/snapshots/1df8507178afcc1bef68cd8c393f61a886323761"
WRONG_STEPS = "wrong_steps_50k.jsonl"
MAX_NEW_TOKENS = 600

t0 = time.time()
def log(m): print(f"[{time.time()-t0:.1f}s] {m}", flush=True)

def load_gold_lookup(path: str) -> dict:
    """Map dataset_index -> {question, gold_solution, gold_answer}."""
    log(f"Loading gold lookup from {path}")
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            di = r.get("dataset_index")
            if di is None:
                continue
            gs = r.get("gold_solution", "") or ""
            ga = r.get("gold_answer", "") or ""
            if not ga:
                m = re.search(r"####\s*([\d\.,\-]+)", gs)
                if m:
                    ga = m.group(1).strip()
            out[di] = {
                "question":      r.get("question", "") or "",
                "gold_solution": gs,
                "gold_answer":   ga,
            }
    log(f"Loaded {len(out)} gold records")
    return out

def reasoning_so_far_d1(asst: str, wrong_step_text: str, marker: str) -> str:
    """For D1: take everything before wrong_step_text (prefix) plus the line
    starting at the marker (Corrected step: ...) until end of asst."""
    wrong_idx = asst.find(wrong_step_text)
    prefix = asst[:wrong_idx].rstrip() if wrong_idx != -1 else ""
    m_idx = asst.lower().rfind(marker.lower())
    if m_idx == -1:
        return prefix.strip()
    block = asst[m_idx:].strip()
    return (prefix + "\n" + block).strip()

def reasoning_so_far_d2(asst: str, marker: str) -> str:
    """For D2: take everything from start through the line starting with marker
    (Correction: ...) until end of asst. The wrong+downstream span sits inside
    the assistant turn but is followed by Detection/Retrace/Error/Diagnosis/Correction.
    For continuation purposes the model should see prefix + the corrected chain only."""
    m_idx = asst.lower().rfind(marker.lower())
    if m_idx == -1:
        return asst.strip()
    block = asst[m_idx:].strip()
                                                                              
    det_idx = asst.lower().find("detection:")
    if det_idx != -1:
        prefix = asst[:det_idx].rstrip()
                                                                                    
                                                                                           
                                                                                          
                                                                                             
        return (prefix + "\n" + block).strip()
    return block

def build_prompt(question: str, gold_solution: str, reasoning: str, gold_answer: str, tokenizer) -> str:
    content = (
        f"Problem:\n{question.strip()}\n\n"
        f"Reference solution:\n{gold_solution.strip()}\n\n"
        f"Reasoning written so far:\n{reasoning.strip()}\n\n"
        f"Using the reference solution as a guide, write the remaining steps that "
        f"continue from the corrected step and arrive at the final answer.\n"
        f"The final answer must be {gold_answer}.\n"
        f"End your response with: #### {gold_answer}"
    )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )

def truncate_after_hash(text: str, gold_answer: str) -> str:
    """Truncate the generated text right after the first `#### <gold_answer>` it emits."""
    pat = re.compile(rf"####\s*{re.escape(gold_answer)}\b")
    m = pat.search(text)
    if m:
        return text[:m.end()]
                                                              
    pat2 = re.compile(r"####\s*[\d\.,\-]+")
    m2 = pat2.search(text)
    if m2:
                                                      
        return text[:m2.end()].rstrip() + f"\n#### {gold_answer}"
                             
    return text.rstrip() + f"\n#### {gold_answer}"

def generate(prompt: str, model, tokenizer, gold_answer: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
                                                                                               
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    return truncate_after_hash(raw, gold_answer)

def process_split(records, gold_lookup, model, tokenizer, marker, mode, split_name):
    out = []
    skipped = 0
    for i, rec in enumerate(records):
        di = rec.get("dataset_index")
        gold = gold_lookup.get(di)
        if gold is None or not gold["gold_answer"]:
            skipped += 1
            continue

        question      = gold["question"]
        gold_solution = gold["gold_solution"]
        gold_answer   = gold["gold_answer"]

        asst = rec["messages"][1]["content"]
        wrong_step_text = rec.get("wrong_step_text", "")

        if mode == "d1":
            reasoning = reasoning_so_far_d1(asst, wrong_step_text, marker)
        else:
            reasoning = reasoning_so_far_d2(asst, marker)

        prompt = build_prompt(question, gold_solution, reasoning, gold_answer, tokenizer)
        cont = generate(prompt, model, tokenizer, gold_answer)

        new_asst = asst.rstrip() + "\n" + cont
        out.append({**rec, "messages": [rec["messages"][0], {"role": "assistant", "content": new_asst}]})

        if (i + 1) % 50 == 0:
            log(f"[{split_name}] {i+1}/{len(records)} | skipped={skipped}")

    log(f"[{split_name}] Done. {len(out)} records written, {skipped} skipped.")
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_in",   required=True)
    ap.add_argument("--val_in",     required=True)
    ap.add_argument("--out_train",  required=True)
    ap.add_argument("--out_val",    required=True)
    ap.add_argument("--marker",     required=True, choices=["corrected step:", "correction:"])
    ap.add_argument("--mode",       required=True, choices=["d1", "d2"])
    ap.add_argument("--gold_path",  default=WRONG_STEPS)
    ap.add_argument("--shard",        type=int, default=0)
    ap.add_argument("--total_shards", type=int, default=1)
    args = ap.parse_args()

    gold_lookup = load_gold_lookup(args.gold_path)

    log(f"Loading DeepSeek-R1-Distill-Qwen-14B from {DEEPSEEK}")
    tokenizer = AutoTokenizer.from_pretrained(DEEPSEEK)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(DEEPSEEK, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    log("Model ready.")

    train_full = [json.loads(l) for l in open(args.train_in) if l.strip()]
    val_full   = [json.loads(l) for l in open(args.val_in)   if l.strip()]
                                                       
    train = train_full[args.shard::args.total_shards]
    val   = val_full[args.shard::args.total_shards]
    log(f"Shard {args.shard}/{args.total_shards}: Train {len(train)}/{len(train_full)} | Val {len(val)}/{len(val_full)}")

    train_out = process_split(train, gold_lookup, model, tokenizer, args.marker, args.mode, "train")
    val_out   = process_split(val,   gold_lookup, model, tokenizer, args.marker, args.mode, "val")

    Path(args.out_train).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_train, "w") as f:
        for r in train_out:
            f.write(json.dumps(r) + "\n")
    with open(args.out_val, "w") as f:
        for r in val_out:
            f.write(json.dumps(r) + "\n")

    log(f"Wrote {args.out_train}: {len(train_out)} records")
    log(f"Wrote {args.out_val}:   {len(val_out)} records")

if __name__ == "__main__":
    main()
