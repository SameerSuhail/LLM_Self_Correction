# Stage 3 end-to-end reproduction

This document is the canonical path someone else can follow to replicate Stage 3: **datasets → training → evaluation**. Hyperparameters are also summarized in `configs/stage3_defaults.yaml`.

## 0. Environment

From the repository root (`LLM_Self_Correction/`):

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

On Grace-style clusters we used a conda environment at **`$SCRATCH/envs/sft_env`** plus modules as in `scripts/slurm/*.sh`.

Export Hugging Face cache (recommended on clusters):

```bash
export SCRATCH=/scratch/user/$USER          # cluster-specific example
export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_CACHE=$SCRATCH/hf_cache/hub
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache/transformers
```

## 1. Fast path — use shipped training JSONL

The paper training files are committed under `stage3/data/` (`data/README.md` explains them).

Baseline full SFT and LoRA SFT scripts default to those paths relative to **`stage3/`** (implemented via `Path(__file__).parents[2] / "data" / …`).

From `stage3/`:

```bash
# Interactive / single GPU (adjust device_map usage as needed — production used multi-GPU via torchrun)
python scripts/training/run_baseline.py \
  --cache_dir "$SCRATCH/hf_cache" \
  --output_dir "$SCRATCH/checkpoints/baseline_sft_cc_qwen_fast2x"

python scripts/training/run_sft.py \
  --cache_dir "$SCRATCH/hf_cache" \
  --output_dir "$SCRATCH/checkpoints/lora_cc_qwen_fast4x"
```

Cluster template (fills in `cwd` automatically):

```bash
cd stage3
sbatch scripts/slurm/run_baseline_grace.sh
sbatch scripts/slurm/run_sft_grace.sh
```

Checkpoints land under **`$SCRATCH/checkpoints/...`** as in `configs/stage3_defaults.yaml`.

## 2. Evaluation

From **`stage3/`**:

```bash
python scripts/evaluation/eval_benchmarks.py \
  --model_path "$SCRATCH/checkpoints/baseline_sft_cc_qwen_fast2x/final" \
  --dataset openai/human-eval \
  --output results/json/baseline_openai_human-eval.json

python scripts/evaluation/eval_benchmarks.py \
  --model_path Qwen/Qwen2.5-Coder-7B-Instruct \
  --adapter "$SCRATCH/checkpoints/lora_cc_qwen_fast4x/checkpoint-1206" \
  --dataset mbpp \
  --output results/json/lora_mbpp.json
```

Codeforces Task A subset (streams `deepmind/code_contests`):

```bash
python scripts/evaluation/eval_benchmarks.py \
  --model_path Qwen/Qwen2.5-Coder-7B-Instruct \
  --adapter "$SCRATCH/checkpoints/lora_cc_qwen_fast4x/checkpoint-1206" \
  --dataset codeforces-a \
  --output results/json/lora_cf.json
```

Bulk SLURM (writes `stage3/slurm/eval_jobs/`):

```bash
cd stage3
python scripts/slurm/generate_eval_slurm.py
for f in slurm/eval_jobs/*.sh; do sbatch "$f"; sleep 1; done
```

Use `--verbose` on `eval_benchmarks.py` only when debugging extraction or execution failures.

## 3. Phase B — Rebuild training JSONL from intermediate shards only

Assuming some directory **`WORK`** contains:

- `code_contests_wrong_steps_part_*.jsonl`
- `cc_self_correction_part_*.jsonl`

Run:

```bash
python scripts/data_generation/assemble_sft_jsonl.py \
  --work-dir "$WORK" \
  --out-dir "$(pwd)/data"
```

This overwrites the three filenames used by training (`see scripts/data_generation/README.md`).

## 4. Phase C — Full raw pipeline (upstream, GPU-heavy)

Scripts live in `scripts/data_generation/upstream/`. Order and expectations are documented in **`upstream/README.md`**.

Outline:

1. `code_contests_data_gen.py` → wrong-step shards.  
2. `merge_wrong_steps_parts.py` → `code_contests_wrong_steps_all.jsonl`.  
3. `root_cause_attribution_code.py` → `cc_attribution_comparison.jsonl`.  
4. `cc_self_correction_gen.py` → `cc_self_correction_part_*.jsonl`.  
5. `assemble_sft_jsonl.py` → final `data/*.jsonl`.

## 5. Sanity checks

| Check | Expected |
|-------|-----------|
| `data/cc_sft_dataset_baseline.jsonl` exists | Thousands of ChatML-style lines |
| `run_baseline.py` default `--data_path` | Points at `stage3/data/cc_sft_dataset_baseline.jsonl` |
| `sbatch scripts/slurm/*.sh` | `cd`'s into repo `stage3` via `SCRIPT_DIR` resolution |
| `eval_benchmarks.py` prompt | Matches user prefix `"Problem:\n...\nSolve step by step."` used in datasets |
