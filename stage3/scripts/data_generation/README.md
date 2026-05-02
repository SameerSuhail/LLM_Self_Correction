# Lightweight dataset assembly (no GPU)

From a directory that already contains:

- `code_contests_wrong_steps_part_*.jsonl`
- `cc_self_correction_part_*.jsonl`

run:

```bash
python scripts/data_generation/assemble_sft_jsonl.py \
  --work-dir /path/to/shards \
  --out-dir /path/to/stage3/data
```

This recreates `cc_sft_dataset_qwen.jsonl`, `cc_sft_dataset_baseline.jsonl`, and `cc_sft_dataset_qwen_mixed.jsonl`.

To merge wrong-step shards before attribution (see `upstream/README.md`):

```bash
python scripts/data_generation/merge_wrong_steps_parts.py --work-dir /path/to/shards
```
