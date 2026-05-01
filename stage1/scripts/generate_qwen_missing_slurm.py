"""Generate SLURM scripts for the 8 missing Qwen conditions.

For teammate to run on their own SLURM account when Ashish's budget is exhausted.

Usage:
    # 1. Edit the ACCOUNT variable below to your SLURM account number
    # 2. Run:
    python scripts/generate_qwen_missing_slurm.py
    # 3. Submit:
    for f in slurm/qwen_missing/*.sh; do sbatch $f; sleep 1; done

Expected SU cost: ~400-500 SUs total
"""

import os
from pathlib import Path

# ============================================================================
# EDIT THIS LINE: Replace with your SLURM account number
# ============================================================================
ACCOUNT = 'REPLACE_WITH_YOUR_ACCOUNT'
# ============================================================================

PROJECT_DIR = '/scratch/user/ashish_molakalapalli/nlp_project'
PYTHON = '/scratch/user/ashish_molakalapalli/envs/selfcorrect/bin/python'
SLURM_DIR = Path(PROJECT_DIR) / 'slurm' / 'qwen_missing'

HEADER = """#!/bin/bash
#SBATCH --job-name=selfcorrect_{name}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time={time}
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output={project}/logs/%x_%j.out
#SBATCH --error={project}/logs/%x_%j.err
#SBATCH --account={account}

module load CUDA/12.4.0
module load WebProxy
export LD_LIBRARY_PATH=/scratch/user/ashish_molakalapalli/envs/selfcorrect/lib/python3.10/site-packages/nvidia/cusparselt/lib:$LD_LIBRARY_PATH
cd {project}
"""

# 8 missing conditions with extended time limits (S3 = 3 correction rounds, S5 = 3 generations per Q)
# Format: (dataset, strategy, time_limit)
MISSING = [
    ('triviaqa', 'baseline', '04:00:00'),       # short -- single-pass baseline
    ('gsm8k', 's3', '14:00:00'),                # iterative on 1319 problems
    ('gsm8k', 's5', '14:00:00'),                # explain-verify on 1319 problems
    ('triviaqa', 's3', '10:00:00'),             # iterative on 1000 problems
    ('triviaqa', 's5', '10:00:00'),             # explain-verify on 1000 problems
    ('strategyqa', 's3', '20:00:00'),           # iterative on 2290 problems -- LONGEST
    ('strategyqa', 's4', '14:00:00'),           # confidence-gated on 2290
    ('strategyqa', 's5', '20:00:00'),           # explain-verify on 2290
]


def main():
    if ACCOUNT == 'REPLACE_WITH_YOUR_ACCOUNT':
        print("ERROR: Edit this file and set ACCOUNT to your SLURM account number first.")
        print("       Open scripts/generate_qwen_missing_slurm.py and change line 22.")
        return

    SLURM_DIR.mkdir(parents=True, exist_ok=True)

    for dataset, strategy, time in MISSING:
        name = f'qwen_{dataset}_{strategy}'
        output_path = f'results/raw/{name}.json'
        cmd = (
            f'{PYTHON} scripts/run_experiment.py '
            f'--model qwen --dataset {dataset} --strategy {strategy} '
            f'--output {output_path}'
        )
        script = HEADER.format(name=name, time=time, project=PROJECT_DIR, account=ACCOUNT) + cmd + '\n'
        path = SLURM_DIR / f'{name}.sh'
        with open(path, 'w') as f:
            f.write(script)
        print(f'  wrote {path}')

    print()
    print('To submit all 8 missing-condition jobs:')
    print(f'  for f in {SLURM_DIR}/*.sh; do sbatch $f; sleep 1; done')
    print()
    print('Estimated total SU cost: ~400-500 SUs')
    print('Wall time per job (longest = strategyqa S3/S5): up to 20 hours')


if __name__ == '__main__':
    main()
