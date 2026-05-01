"""Regenerate Qwen SLURM scripts with EXTENDED time limits for jobs that timed out.

Usage:
    python scripts/generate_qwen_rerun_slurm.py
    for f in slurm/qwen_rerun/*.sh; do sbatch $f; sleep 1; done
"""

import os
from pathlib import Path

PROJECT_DIR = '/scratch/user/ashish_molakalapalli/nlp_project'
PYTHON = '/scratch/user/ashish_molakalapalli/envs/selfcorrect/bin/python'
ACCOUNT = '132681707964'
SLURM_DIR = Path(PROJECT_DIR) / 'slurm' / 'qwen_rerun'

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

# 7 timed-out jobs with extended time limits (~2x original)
# Format: (dataset, strategy, time)
RERUNS = [
    ('gsm8k', 's4', '12:00:00'),
    ('strategyqa', 's4', '14:00:00'),
    ('triviaqa', 's4', '08:00:00'),
    ('strategyqa', 's1', '14:00:00'),
    ('triviaqa', 's1', '08:00:00'),
    ('triviaqa', 's2', '08:00:00'),
    ('strategyqa', 'self_consistency', '14:00:00'),
]


def main():
    SLURM_DIR.mkdir(parents=True, exist_ok=True)
    for dataset, strategy, time in RERUNS:
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


if __name__ == '__main__':
    main()
