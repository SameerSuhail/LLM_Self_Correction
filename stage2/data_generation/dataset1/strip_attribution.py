import argparse
import json
import re

PATTERNS = [
    (re.compile(r"Let me look back at the question\b", re.IGNORECASE), "Let me think about this"),
    (re.compile(r"Let me rethink this step\b",          re.IGNORECASE), "Let me think about this"),
    (re.compile(r"Let me recheck Step\s*\d+\b",         re.IGNORECASE), "Let me think about this"),
]

def strip(text: str):
    hits = 0
    for pat, repl in PATTERNS:
        text, n = pat.subn(repl, text)
        hits += n
    return text, hits

def process(inp, outp):
    changed = 0
    total_hits = 0
    with open(inp) as f:
        records = [json.loads(l) for l in f if l.strip()]

    with open(outp, "w") as f:
        for r in records:
            asst = r["messages"][1]["content"]
            new_asst, hits = strip(asst)
            if hits > 0:
                changed += 1
            total_hits += hits
            r["messages"][1]["content"] = new_asst
            f.write(json.dumps(r) + "\n")

    print(f"{inp} -> {outp}: {len(records)} records, {changed} altered, {total_hits} phrase replacements")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_train",  default="sft_dataset1_cont_train.jsonl")
    ap.add_argument("--in_val",    default="sft_dataset1_cont_val.jsonl")
    ap.add_argument("--out_train", default="sft_dataset1_cont_noattr_train.jsonl")
    ap.add_argument("--out_val",   default="sft_dataset1_cont_noattr_val.jsonl")
    args = ap.parse_args()

    process(args.in_train, args.out_train)
    process(args.in_val,   args.out_val)

if __name__ == "__main__":
    main()
