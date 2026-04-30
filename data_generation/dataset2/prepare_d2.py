import argparse
import json
import random
from pathlib import Path

REQUIRED_SFT_FIELDS = ("detection", "retrace", "error_trace", "diagnosis", "correction")

def load_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]

def is_valid(rec):
    sft = rec.get("sft_output", {})
    if not sft:
        return False
    return all(sft.get(k, "").strip() for k in REQUIRED_SFT_FIELDS)

def build_sample(rec):
    question = rec["question"].strip()
    prefix_text = rec["prefix_text"].strip()
    wrong_step = rec["wrong_step"].strip()
    downstream = [s.strip() for s in rec.get("downstream_context", [])]
    downstream_block = "\n".join(downstream)

    sft = rec["sft_output"]
    detection  = sft["detection"].strip()
    retrace    = sft["retrace"].strip()
    err_trace  = sft["error_trace"].strip()
    diagnosis  = sft["diagnosis"].strip()
    correction = sft["correction"].strip()

    user_content = f"Problem:\n{question}\n\nSolve step by step."

    if downstream_block:
        wrong_span = f"{wrong_step}\n{downstream_block}"
    else:
        wrong_span = wrong_step

    assistant_content = (
        f"{prefix_text}\n"
        f"{wrong_span}\n"
        f"Detection: {detection}\n"
        f"Retrace: {retrace}\n"
        f"Error trace: {err_trace}\n"
        f"Diagnosis: {diagnosis}\n"
        f"Correction: {correction}"
    )

    return {
        "dataset_index":     rec.get("dataset_index"),
        "root_cause_source": rec.get("root_cause_source"),
        "detection_label":   rec.get("detection_label"),
        "messages": [
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "wrong_step_text": wrong_span,
        "correction_text": correction,
        "prefix_text":     prefix_text,
        "question":        question,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",     default="error_trace_sft.jsonl")
    ap.add_argument("--out_train", default="sft_dataset2_spon_train.jsonl")
    ap.add_argument("--out_val",   default="sft_dataset2_spon_val.jsonl")
    ap.add_argument("--val_ratio", type=float, default=0.10)
    ap.add_argument("--seed",      type=int,   default=42)
    args = ap.parse_args()

    records = load_jsonl(args.input)
    print(f"Loaded {len(records)} records from {args.input}")

    valid = [r for r in records if is_valid(r)]
    print(f"Valid: {len(valid)} | Skipped (empty sft_output fields): {len(records) - len(valid)}")

    rng = random.Random(args.seed)
    rng.shuffle(valid)
    n_val = int(len(valid) * args.val_ratio)
    val_recs   = valid[:n_val]
    train_recs = valid[n_val:]

    Path(args.out_train).parent.mkdir(parents=True, exist_ok=True)

    with open(args.out_train, "w") as f:
        for r in train_recs:
            f.write(json.dumps(build_sample(r)) + "\n")
    with open(args.out_val, "w") as f:
        for r in val_recs:
            f.write(json.dumps(build_sample(r)) + "\n")

    print(f"Wrote {args.out_train}: {len(train_recs)} records")
    print(f"Wrote {args.out_val}:   {len(val_recs)} records")

    s = build_sample(train_recs[0])
    asst = s["messages"][1]["content"]
    mask_span = s["wrong_step_text"]
    found = asst.find(mask_span)
    print(f"\nSanity check: wrong_step_text found in assistant turn at char {found} "
          f"(len={len(mask_span)})")

    print("\n--- SAMPLE (train[0]) ---")
    print("USER:\n" + s["messages"][0]["content"])
    print("\nASSISTANT:\n" + asst)

if __name__ == "__main__":
    main()
