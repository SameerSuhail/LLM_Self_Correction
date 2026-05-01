# Stage 3: Training-based code self-correction (Qwen2.5-Coder-7B)

End-to-end documentation for reviewers and replicators: **`docs/REPRODUCTION.md`**.

## Contents

| Path | Purpose |
|------|---------|
| `configs/stage3_defaults.yaml` | Reference hyperparameters and path conventions (`$SCRATCH`). |
| `data/` | Shipped training JSONL used in experiments (`data/README.md`). |
| `scripts/training/` | Baseline full SFT and LoRA SFT (`run_baseline.py`, `run_sft.py`). |
| `scripts/evaluation/` | Benchmark harness `eval_benchmarks.py` (HumanEval, MBPP, `codeforces-a`). |
| `scripts/data_generation/` | Rebuild datasets from shards (`assemble_sft_jsonl.py`, `merge_wrong_steps_parts.py`). |
| `scripts/data_generation/upstream/` | GPU-heavy CodeContests → wrong steps → attribution → self-correction text. |
| `scripts/slurm/` | Grace-style job templates + `generate_eval_slurm.py`. |
| `results/json/` | Example evaluation outputs committed for the paper snapshot. |

## Minimal replication (paper numbers)

Use committed `data/*.jsonl`, train to `$SCRATCH/checkpoints`, then evaluate. Exact commands live in **`docs/REPRODUCTION.md`** §§1–2.

```bash
cd stage3
sbatch scripts/slurm/run_baseline_grace.sh
sbatch scripts/slurm/run_sft_grace.sh
python scripts/evaluation/eval_benchmarks.py ...   # see REPRODUCTION.md
```

## Full pipeline (datasets from CodeContests)

See **`docs/REPRODUCTION.md`** §§3–4 and `scripts/data_generation/upstream/README.md`.

## Snapshot metrics (committed `results/json/`)

These match the bundled JSON artifact files:

- Baseline HumanEval: **59.33%**
- Baseline MBPP: **28.87%**
- Baseline Codeforces-A: **37.83%** (Evaluated by Sonnet 4.6)
- LoRA HumanEval: **53.46%**
- LoRA MBPP: **18.60%**
- LoRA Codeforces-A: **22.61%** (Evaluated by Sonnet 4.6)

Fresh runs must write new JSON under `results/json/` if you compare against checkpoints you trained locally.
For sonner 4.6 evaluation, prompt the model to evaluate the results based on the logic given by the QWEN models rather than output because the correctness of the code can be better evaluated by understanding the logic given.