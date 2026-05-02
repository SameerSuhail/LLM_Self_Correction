from pathlib import Path

# ============================================================================

PROJECT_DIR = str(Path(__file__).resolve().parent.parent.parent)
ENV_PATH = '$SCRATCH/envs/sft_env'
SLURM_DIR = Path(PROJECT_DIR) / 'slurm' / 'eval_jobs'

HEADER = """#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

module purge
module load WebProxy
module load GCC/12.3.0 OpenMPI/4.1.5 PyTorch/2.1.2-CUDA-12.1.1

export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_CACHE=$SCRATCH/hf_cache/hub
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache/transformers

source {env}/bin/activate

cd {project}
echo "Starting evaluation job: {name}"
"""

DATASETS = ["openai/human-eval", "mbpp", "codeforces-a"]

MODELS = {
    "baseline": {
        "model_path": "$SCRATCH/checkpoints/baseline_sft_cc_qwen_fast2x/final",
        "adapter": None
    },
    "lora": {
        "model_path": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "adapter": "$SCRATCH/checkpoints/lora_cc_qwen_fast4x/checkpoint-1206"
    }
}

def main():
    SLURM_DIR.mkdir(parents=True, exist_ok=True)
    Path(PROJECT_DIR, 'results', 'json').mkdir(parents=True, exist_ok=True)

    for model_name, config in MODELS.items():
        for dataset in DATASETS:
            safe_ds_name = dataset.replace('/', '_')
            job_name = f'eval_{model_name}_{safe_ds_name}'
            output_file = f'results/json/{model_name}_{safe_ds_name}.json'
            
            cmd = f'python scripts/evaluation/eval_benchmarks.py \\\n'
            cmd += f'    --model_path "{config["model_path"]}" \\\n'
            if config["adapter"]:
                cmd += f'    --adapter "{config["adapter"]}" \\\n'
            cmd += f'    --dataset "{dataset}" \\\n'
            cmd += f'    --output "{output_file}"\n'

            script = HEADER.format(name=job_name, project=PROJECT_DIR, env=ENV_PATH) + cmd
            
            script_path = SLURM_DIR / f'{job_name}.sh'
            with open(script_path, 'w') as f:
                f.write(script)
            print(f'Wrote {script_path}')

    n_scripts = len(DATASETS) * len(MODELS)
    print(f'\nGenerated {n_scripts} SLURM scripts in {SLURM_DIR}')
    print('Submit all jobs with:')
    print(f'  for f in {SLURM_DIR}/*.sh; do sbatch $f; sleep 1; done')

if __name__ == '__main__':
    main()
