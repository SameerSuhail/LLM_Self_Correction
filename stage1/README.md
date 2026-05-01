# Stage 1 — Prompting-Based Self-Correction

Controlled study of whether asking an LLM to "review your answer" actually helps. Three open-source 7–8B models, five correction strategies, four datasets. Per-instance correction matrix. **96 conditions, 71,595 question–strategy pairs.**

## Headline finding
Across the full grid, prompting-based self-correction broke **14.7%** of correct answers and helped only **4.4%** — a **3.4× damage ratio**. Self-consistency at near-equal compute beats simple-review correction in 11 of 12 conditions and the strongest correction strategy (S2 structured) in 8 of 12.

## Models
- Llama-3.1-8B-Instruct
- Mistral-7B-Instruct-v0.3
- Qwen-2.5-7B-Instruct

## Datasets (test splits, in `data/`)
| Dataset | Size | Task |
|---|---:|---|
| GSM8K | 1,319 | Math |
| TriviaQA | 1,000 (subsampled, seed 42) | Factual QA |
| StrategyQA | 2,290 | Yes/no commonsense |
| HumanEval | 164 | Code generation |

## Correction strategies (S1–S5 + 3 baselines)
- **S1 Simple**: "Review your answer."
- **S2 Structured**: task-specific verification (check arithmetic / verify each fact / trace through code).
- **S3 Iterative**: apply S1 three times in sequence.
- **S4 Confidence-Gated**: ask self-reported confidence on a 1–10 scale, correct only if confidence < 5.
- **S5 Explain-Then-Verify**: explain the reasoning behind the answer, then judge whether it follows.
- **Baselines**: None (no correction), CoT ("let's think step by step"), Self-Consistency (N=3, T=0.7, majority vote).

## Repo layout
```
stage1/
├── src/
│   ├── inference.py            # model loading + generation pipeline
│   ├── evaluation.py           # answer extractors per dataset, code sandbox
│   └── confidence_parser.py    # parses "8/10", "about 7", etc. into integers
├── scripts/
│   ├── run_experiment.py       # main runner: --model {llama,mistral,qwen} --dataset --strategy
│   ├── prepare_datasets.py     # builds the test JSONs in data/
│   ├── smoke_test.py           # 10-question sanity check on each model
│   ├── analyze_results.py      # builds main_results.csv from raw/
│   ├── build_predictor.py      # decision-tree framework (Exp 6)
│   ├── run_threshold_ablation.py    # S4 confidence threshold sweep (Exp 2)
│   ├── run_compute_matched.py  # S1 vs Self-Consistency at equal compute (Exp 4)
│   ├── run_calibration.py      # ECE + calibration curves (Exp 7)
│   ├── run_error_analysis.py   # 600 C→W regression categorization (Exp 5)
│   ├── run_feedback_quality.py # does S2 critique identify real errors? (Exp 8)
│   ├── generate_figures.py     # produces figures/*.png from results/
│   └── generate_*_slurm.py     # SLURM batch generators for the cluster
├── configs/prompts/
│   ├── initial.yaml            # initial-generation prompts per dataset
│   └── correction.yaml         # S1–S5 prompt templates
├── slurm/                      # SLURM wrappers
├── data/                       # 4 dataset test JSONs (preprocessed)
├── results/
│   ├── raw/                    # 96 condition JSONs (one per model × dataset × strategy)
│   ├── main_results.csv        # aggregate correction matrix per condition
│   ├── threshold_ablation.csv  # S4 sweep
│   ├── compute_matched.csv     # S1 vs SC
│   ├── calibration.csv, calibration_ece.csv
│   ├── error_analysis.csv      # 600 C→W categorization
│   ├── feedback_quality.csv    # 50 GSM8K cases
│   ├── decision_features.csv   # features for the decision tree
│   └── prediction_model/       # depth-4 decision tree + extracted rules
├── figures/                    # 6 PNGs (heatmap + per-experiment plots)
├── requirements.txt
└── README.md (this file)
```

## How to reproduce a single condition
```bash
python scripts/run_experiment.py \
    --model llama \
    --dataset gsm8k \
    --strategy s1 \
    --output results/raw/llama_gsm8k_s1.json
```

## Full grid
The 96 SLURM scripts are produced by:
```bash
python scripts/generate_slurm.py        # Llama + Mistral
python scripts/generate_qwen_slurm.py   # Qwen-2.5-7B
```

After all conditions finish:
```bash
python scripts/analyze_results.py       # builds main_results.csv
python scripts/build_predictor.py       # decision tree (will-help / will-hurt classifier)
python scripts/run_threshold_ablation.py
python scripts/run_compute_matched.py
python scripts/run_calibration.py
python scripts/run_error_analysis.py
python scripts/run_feedback_quality.py
python scripts/generate_figures.py
```

## Cluster note
Built and run on TAMU HPRC Grace; A100 40GB GPUs, greedy decoding for correction (T=0), bf16/fp16 weights. Self-Consistency uses T=0.7 with N=3 independent samples.

## Decision framework
Depth-4 decision tree on 71,595 examples:
- **75.7%** cross-validated accuracy on the will-help binary classifier
- **70.2%** on will-hurt
- Top features: `confidence` (37%), `task_type` (25%), `model` (22%)
- Extracted rule (most useful): *if self-reported confidence > 4, do not correct*

See `results/prediction_model/rules.txt` for the full extracted rule set.
