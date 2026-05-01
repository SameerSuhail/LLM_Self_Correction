"""Generate SLURM batch scripts for Qwen-2.5-7B-Instruct experiments.

This is a copy of generate_slurm.py but hardcoded for Qwen only.
Your teammate can edit the ACCOUNT variable below and submit everything.

Usage:
    # 1. Edit the ACCOUNT variable below to the teammate's SLURM account
    # 2. Run:
    python scripts/generate_qwen_slurm.py --phase full
    # 3. Submit:
    for f in slurm/qwen/*.sh; do sbatch $f; sleep 1; done

Scope options:
    --phase full     : All 23 runs (5 strategies + 3 baselines) x 4 datasets
    --phase minimal  : 6 runs (S2+S4 on GSM8K+TriviaQA + 2 baselines)
    --phase baselines: 12 baseline/cot/SC runs only
    --phase main     : 20 correction strategy runs only
"""

import argparse
import os

# ============================================================================
# EDIT THIS LINE: Replace with your SLURM account number
# ============================================================================
ACCOUNT = '132681707964'
# ============================================================================

SLURM_DIR = os.path.join(os.path.dirname(__file__), '..', 'slurm', 'qwen')
PROJECT_DIR = '/scratch/user/ashish_molakalapalli/nlp_project'
PYTHON = '/scratch/user/ashish_molakalapalli/envs/selfcorrect/bin/python'

SLURM_HEADER = """#!/bin/bash
#SBATCH --job-name=selfcorrect_{job_name}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time={time}
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output={project_dir}/logs/%x_%j.out
#SBATCH --error={project_dir}/logs/%x_%j.err
#SBATCH --account={account}

module load CUDA/12.4.0
module load WebProxy

export LD_LIBRARY_PATH=/scratch/user/ashish_molakalapalli/envs/selfcorrect/lib/python3.10/site-packages/nvidia/cusparselt/lib:$LD_LIBRARY_PATH

cd {project_dir}
"""

MODEL = 'qwen'
DATASETS = ['gsm8k', 'triviaqa', 'strategyqa', 'humaneval']
STRATEGIES = ['s1', 's2', 's3', 's4', 's5']
BASELINES = ['baseline', 'cot', 'self_consistency']

TIME_ESTIMATES = {
    ('baseline', 'gsm8k'): '04:00:00',
    ('baseline', 'triviaqa'): '02:00:00',
    ('baseline', 'strategyqa'): '04:00:00',
    ('baseline', 'humaneval'): '01:00:00',
    ('cot', 'gsm8k'): '04:00:00',
    ('cot', 'triviaqa'): '02:00:00',
    ('cot', 'strategyqa'): '04:00:00',
    ('cot', 'humaneval'): '01:00:00',
    ('self_consistency', 'gsm8k'): '06:00:00',
    ('self_consistency', 'triviaqa'): '04:00:00',
    ('self_consistency', 'strategyqa'): '06:00:00',
    ('self_consistency', 'humaneval'): '02:00:00',
    ('s1', 'gsm8k'): '06:00:00',
    ('s1', 'triviaqa'): '04:00:00',
    ('s1', 'strategyqa'): '06:00:00',
    ('s1', 'humaneval'): '02:00:00',
    ('s2', 'gsm8k'): '06:00:00',
    ('s2', 'triviaqa'): '04:00:00',
    ('s2', 'strategyqa'): '06:00:00',
    ('s2', 'humaneval'): '02:00:00',
    ('s3', 'gsm8k'): '12:00:00',
    ('s3', 'triviaqa'): '08:00:00',
    ('s3', 'strategyqa'): '12:00:00',
    ('s3', 'humaneval'): '04:00:00',
    ('s4', 'gsm8k'): '06:00:00',
    ('s4', 'triviaqa'): '04:00:00',
    ('s4', 'strategyqa'): '06:00:00',
    ('s4', 'humaneval'): '02:00:00',
    ('s5', 'gsm8k'): '12:00:00',
    ('s5', 'triviaqa'): '06:00:00',
    ('s5', 'strategyqa'): '16:00:00',
    ('s5', 'humaneval'): '03:00:00',
}


def write_script(job_name, dataset, strategy):
    time_est = TIME_ESTIMATES.get((strategy, dataset), '04:00:00')

    header = SLURM_HEADER.format(
        job_name=job_name,
        time=time_est,
        project_dir=PROJECT_DIR,
        account=ACCOUNT,
    )

    output_path = f'results/raw/{MODEL}_{dataset}_{strategy}.json'
    cmd = f'{PYTHON} scripts/run_experiment.py --model {MODEL} --dataset {dataset} --strategy {strategy} --output {output_path}'

    script = header + cmd + '\n'
    path = os.path.join(SLURM_DIR, f'{job_name}.sh')
    with open(path, 'w') as f:
        f.write(script)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', required=True,
                        choices=['full', 'minimal', 'baselines', 'main'])
    args = parser.parse_args()

    if ACCOUNT == 'REPLACE_WITH_YOUR_ACCOUNT':
        print('ERROR: Please edit ACCOUNT in this file before generating scripts.')
        print('       Open scripts/generate_qwen_slurm.py and set ACCOUNT = "your-slurm-account"')
        return

    os.makedirs(SLURM_DIR, exist_ok=True)
    scripts = []

    if args.phase == 'full':
        # Complete grid: 23 runs
        for dataset in DATASETS:
            for strategy in BASELINES + STRATEGIES:
                job_name = f'{MODEL}_{dataset}_{strategy}'
                scripts.append(write_script(job_name, dataset, strategy))
    elif args.phase == 'minimal':
        # Minimal viable: S2 + S4 on GSM8K + TriviaQA + baselines
        for dataset in ['gsm8k', 'triviaqa']:
            for strategy in ['baseline', 's2', 's4']:
                job_name = f'{MODEL}_{dataset}_{strategy}'
                scripts.append(write_script(job_name, dataset, strategy))
    elif args.phase == 'baselines':
        for dataset in DATASETS:
            for strategy in BASELINES:
                job_name = f'{MODEL}_{dataset}_{strategy}'
                scripts.append(write_script(job_name, dataset, strategy))
    elif args.phase == 'main':
        for dataset in DATASETS:
            for strategy in STRATEGIES:
                job_name = f'{MODEL}_{dataset}_{strategy}'
                scripts.append(write_script(job_name, dataset, strategy))

    print(f'Generated {len(scripts)} Qwen scripts in {SLURM_DIR}')
    print()
    print(f'To submit all scripts:')
    print(f'  for f in {SLURM_DIR}/*.sh; do sbatch $f; sleep 1; done')
    print()
    print(f'Estimated SU cost:')
    if args.phase == 'full':
        print(f'  ~100-140 GPU-hours, ~900-1260 SUs')
    elif args.phase == 'minimal':
        print(f'  ~15-20 GPU-hours, ~140-180 SUs')
    elif args.phase == 'baselines':
        print(f'  ~30-40 GPU-hours, ~270-360 SUs')
    elif args.phase == 'main':
        print(f'  ~70-100 GPU-hours, ~630-900 SUs')


if __name__ == '__main__':
    main()
