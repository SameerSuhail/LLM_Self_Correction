# Stage 3 training data (shipped)

These JSONL files are the **exact** supervision used in the paper runs:

- `cc_sft_dataset_baseline.jsonl` — standard chain-of-thought + final code (no injected error trace).
- `cc_sft_dataset_qwen_mixed.jsonl` — mix of error-trace corrective examples and duplicated baseline-style negatives (1 positive : 5 negatives by construction).

Both are in Qwen ChatML-style `text` records (see any line: `{"text": "<|im_start|>user\n..."}`).

To **rebuild from scratch**, follow `docs/REPRODUCTION.md` Phase B (intermediate shards) and Phase C (upstream GPU jobs), then run `scripts/data_generation/assemble_sft_jsonl.py`.
