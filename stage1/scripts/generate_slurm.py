"""Generate SLURM batch scripts for all experiment conditions.

Usage:
    python scripts/generate_slurm.py --phase baselines
    python scripts/generate_slurm.py --phase main
    python scripts/generate_slurm.py --phase threshold
"""

import argparse
import os

SLURM_DIR = os.path.join(os.path.dirname(__file__), '..', 'slurm', 'generated')
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
#SBATCH --account=132681707964

module load CUDA/12.4.0
module load WebProxy

export LD_LIBRARY_PATH=/scratch/user/ashish_molakalapalli/envs/selfcorrect/lib/python3.10/site-packages/nvidia/cusparselt/lib:$LD_LIBRARY_PATH

cd {project_dir}
"""

MODELS = ['llama', 'mistral']
DATASETS = ['gsm8k', 'triviaqa', 'strategyqa', 'humaneval']
STRATEGIES = ['s1', 's2', 's3', 's4', 's5']

# Time estimates per (strategy, dataset) — conservative
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
    ('s5', 'gsm8k'): '06:00:00',
    ('s5', 'triviaqa'): '04:00:00',
    ('s5', 'strategyqa'): '08:00:00',
    ('s5', 'humaneval'): '02:00:00',
}


def write_script(job_name, model, dataset, strategy, extra_args=''):
    time_est = TIME_ESTIMATES.get((strategy, dataset), '04:00:00')

    header = SLURM_HEADER.format(
        job_name=job_name,
        time=time_est,
        project_dir=PROJECT_DIR,
    )

    output_path = f'results/raw/{model}_{dataset}_{strategy}.json'
    cmd = f'{PYTHON} scripts/run_experiment.py --model {model} --dataset {dataset} --strategy {strategy} --output {output_path}'
    if extra_args:
        cmd += f' {extra_args}'

    script = header + cmd + '\n'

    path = os.path.join(SLURM_DIR, f'{job_name}.sh')
    with open(path, 'w') as f:
        f.write(script)

    return path


def generate_baselines():
    """Generate baseline scripts: baseline + cot + self_consistency for all models × datasets."""
    scripts = []
    for model in MODELS:
        for dataset in DATASETS:
            for strategy in ['baseline', 'cot', 'self_consistency']:
                job_name = f'{model}_{dataset}_{strategy}'
                path = write_script(job_name, model, dataset, strategy)
                scripts.append(path)
    return scripts


def generate_main():
    """Generate main experiment scripts: S1-S5 for all models × datasets."""
    scripts = []
    for model in MODELS:
        for dataset in DATASETS:
            for strategy in STRATEGIES:
                job_name = f'{model}_{dataset}_{strategy}'
                path = write_script(job_name, model, dataset, strategy)
                scripts.append(path)
    return scripts


def generate_threshold():
    """Generate threshold ablation scripts: S4 with τ ∈ {1,...,9}."""
    scripts = []
    for model in MODELS:
        for dataset in DATASETS:
            for tau in range(1, 10):
                job_name = f'{model}_{dataset}_s4_tau{tau}'
                output_path = f'results/raw/{model}_{dataset}_s4_tau{tau}.json'
                extra = f'--threshold {tau} --output {output_path}'
                # Override output in the script
                path = write_script(job_name, model, dataset, 's4', extra_args=f'--threshold {tau}')
                # Fix: rewrite to use correct output path
                with open(path) as f:
                    content = f.read()
                content = content.replace(
                    f'results/raw/{model}_{dataset}_s4.json',
                    f'results/raw/{model}_{dataset}_s4_tau{tau}.json'
                )
                with open(path, 'w') as f:
                    f.write(content)
                scripts.append(path)
    return scripts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', required=True,
                        choices=['baselines', 'main', 'threshold', 'all'])
    args = parser.parse_args()

    os.makedirs(SLURM_DIR, exist_ok=True)

    if args.phase in ('baselines', 'all'):
        scripts = generate_baselines()
        print(f"Generated {len(scripts)} baseline scripts")

    if args.phase in ('main', 'all'):
        scripts = generate_main()
        print(f"Generated {len(scripts)} main experiment scripts")

    if args.phase in ('threshold', 'all'):
        scripts = generate_threshold()
        print(f"Generated {len(scripts)} threshold ablation scripts")

    print(f"\nScripts saved to: {os.path.abspath(SLURM_DIR)}")
    print(f"\nTo submit all scripts in a phase:")
    print(f"  for f in {SLURM_DIR}/*.sh; do sbatch $f; sleep 1; done")


if __name__ == '__main__':
    main()
