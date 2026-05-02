# Upstream CodeContests data generation (GPU-heavy)

These scripts regenerate the **intermediate** JSONL shards from the CodeContests benchmark. They are **compute-intensive** (multi-GPU recommended) and download large models through Hugging Face.

| Script | Role |
|--------|------|
| `code_contests_data_gen.py` | Sample wrong reasoning steps on CodeContests problems; writes `code_contests_wrong_steps_part_*.jsonl` (use `--part`). |
| `merge_wrong_steps_parts.py` | One directory level up: merge parts → `code_contests_wrong_steps_all.jsonl` for attribution. |
| `root_cause_attribution_code.py` | **Stage 3 / CodeContests.** Attribute root cause of each wrong step → `cc_attribution_comparison.jsonl`. Math analogue: `stage2/data_generation/common/root_cause_attribution.py`. |
| `cc_self_correction_gen.py` | Given attributions, produce `self_correction` fields → `cc_self_correction_part_*.jsonl`. |

Typical order (from an empty working directory on a node with HF + CUDA):

```bash
# 1) Wrong-step corpus (example: 4 parallel jobs; adjust --part / --num-* per script -h)
python code_contests_data_gen.py --part 0   # ... etc.

# 2) Merge for attribution input
python ../merge_wrong_steps_parts.py --work-dir .

# 3) Attribution (see script docstring for GPU layout and flags)
python root_cause_attribution_code.py --input code_contests_wrong_steps_all.jsonl

# 4) Self-correction text generation (split with --part/--num-parts for parallel runs)
python cc_self_correction_gen.py --input cc_attribution_comparison.jsonl --part 0 --num-parts 4
```

Then assemble final SFT JSONL:

```bash
python ../assemble_sft_jsonl.py --work-dir . --out-dir /path/to/stage3/data
```

**Note:** Defaults inside the upstream scripts assume particular CUDA device placement (see each file header). Adapt `device_map` or run on a smaller machine only for smoke tests.
