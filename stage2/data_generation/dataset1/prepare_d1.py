import argparse
import json

def load_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]

def is_valid(record: dict) -> bool:
    sc = record.get("self_correction", {})
    return bool(
        sc.get("error_trace") and
        sc.get("error_diagnosis") and
        sc.get("corrected_step")
    )

def build_sample(record: dict) -> dict:
    question     = record["question"].strip()
    prefix_text  = record["prefix_text"].strip()
    wrong_step   = record["wrong_step"].strip()
    sc           = record["self_correction"]
    error_trace  = sc["error_trace"].strip()
    diagnosis    = sc["error_diagnosis"].strip()
    corrected    = sc["corrected_step"].strip()

    user_content = (
        f"Problem:\n{question}\n\n"
        f"Solve step by step."
    )

    assistant_content = (
        f"{prefix_text}\n"
        f"{wrong_step}\n"
        f"Error trace: {error_trace}\n"
        f"Diagnosis: {diagnosis}\n"
        f"Corrected step: {corrected}"
    )

    return {
        "dataset_index":     record["dataset_index"],
        "root_cause_source": record["llm_judge"]["root_cause_source"],
        "messages": [
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "wrong_step_text": wrong_step,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  default="self_correction_50k.jsonl")
    ap.add_argument("--output", default="sft_dataset1.jsonl")
    args = ap.parse_args()

    records = load_jsonl(args.input)
    print(f"Loaded {len(records)} records from {args.input}")

    valid   = [r for r in records if is_valid(r)]
    skipped = len(records) - len(valid)
    print(f"Valid: {len(valid)} | Skipped (empty fields): {skipped}")

    with open(args.output, "w") as f:
        for r in valid:
            f.write(json.dumps(build_sample(r)) + "\n")
    print(f"Written {len(valid)} samples to {args.output}")

    sample = build_sample(valid[0])
    print("\n=== SAMPLE ===")
    print("USER:")
    print(sample["messages"][0]["content"])
    print("\nASSISTANT:")
    print(sample["messages"][1]["content"])
    print("\nWRONG STEP TO MASK:")
    print(sample["wrong_step_text"])

if __name__ == "__main__":
    main()
