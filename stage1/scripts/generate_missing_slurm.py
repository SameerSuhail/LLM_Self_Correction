"""Generate SLURM scripts for the 13 missing correction conditions."""

import os

PROJECT_DIR = "/scratch/user/ashish_molakalapalli/nlp_project"
SLURM_DIR = os.path.join(PROJECT_DIR, "slurm", "phase3_remaining")
os.makedirs(SLURM_DIR, exist_ok=True)

MODEL_PATHS = {
    'llama': 'meta-llama/Llama-3.1-8B-Instruct',
    'mistral': 'mistralai/Mistral-7B-Instruct-v0.3',
}

# Missing conditions with time estimates
MISSING = [
    # Timed-out GSM8K resubmits (need longer time limits)
    ('llama', 'gsm8k', 's2', '10:00:00'),
    ('llama', 'gsm8k', 's5', '10:00:00'),
    ('mistral', 'gsm8k', 's5', '10:00:00'),
    # S5 StrategyQA (never run)
    ('llama', 'strategyqa', 's5', '08:00:00'),
    ('mistral', 'strategyqa', 's5', '08:00:00'),
    # S3 iterative (all 8 conditions — 4 generations per question)
    ('llama', 'gsm8k', 's3', '16:00:00'),
    ('llama', 'triviaqa', 's3', '10:00:00'),
    ('llama', 'strategyqa', 's3', '20:00:00'),
    ('llama', 'humaneval', 's3', '03:00:00'),
    ('mistral', 'gsm8k', 's3', '12:00:00'),
    ('mistral', 'triviaqa', 's3', '08:00:00'),
    ('mistral', 'strategyqa', 's3', '16:00:00'),
    ('mistral', 'humaneval', 's3', '03:00:00'),
]

TEMPLATE = """#!/bin/bash
#SBATCH --job-name=selfcorrect_{model}_{dataset}_{strategy}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time={time_limit}
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output={project}/logs/%x_%j.out
#SBATCH --error={project}/logs/%x_%j.err
#SBATCH --account=132681707964

module load CUDA/12.4.0
module load WebProxy

export LD_LIBRARY_PATH=/scratch/user/ashish_molakalapalli/envs/selfcorrect/lib/python3.10/site-packages/nvidia/cusparselt/lib:$LD_LIBRARY_PATH
export TRANSFORMERS_CACHE=/scratch/user/ashish_molakalapalli/nlp_project/data/models

cd {project}

/scratch/user/ashish_molakalapalli/envs/selfcorrect/bin/python scripts/run_experiment.py \\
    --model {model} --dataset {dataset} --strategy {strategy} \\
    --output results/raw/{model}_{dataset}_{strategy}.json
"""

for model, dataset, strategy, time_limit in MISSING:
    script = TEMPLATE.format(
        model=model,
        dataset=dataset,
        strategy=strategy,
        time_limit=time_limit,
        project=PROJECT_DIR,
    )
    filename = f"{model}_{dataset}_{strategy}.sh"
    filepath = os.path.join(SLURM_DIR, filename)
    with open(filepath, 'w') as f:
        f.write(script)
    print(f"  Generated: {filename} (time={time_limit})")

print(f"\n{len(MISSING)} scripts written to {SLURM_DIR}/")
print("\nTo submit non-S3 jobs first:")
print(f"  for f in {SLURM_DIR}/{{llama_gsm8k_s2,llama_gsm8k_s5,mistral_gsm8k_s5,llama_strategyqa_s5,mistral_strategyqa_s5}}.sh; do sbatch $f; sleep 1; done")
print("\nTo submit S3 jobs:")
print(f"  for f in {SLURM_DIR}/*_s3.sh; do sbatch $f; sleep 1; done")
