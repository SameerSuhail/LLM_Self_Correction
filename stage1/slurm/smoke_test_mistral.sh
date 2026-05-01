#!/bin/bash
#SBATCH --job-name=selfcorrect_smoke_mistral
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/scratch/user/ashish_molakalapalli/nlp_project/logs/%x_%j.out
#SBATCH --error=/scratch/user/ashish_molakalapalli/nlp_project/logs/%x_%j.err
#SBATCH --account=132681707964

module load CUDA/12.4.0
module load WebProxy

export LD_LIBRARY_PATH=/scratch/user/ashish_molakalapalli/envs/selfcorrect/lib/python3.10/site-packages/nvidia/cusparselt/lib:$LD_LIBRARY_PATH

cd /scratch/user/ashish_molakalapalli/nlp_project

/scratch/user/ashish_molakalapalli/envs/selfcorrect/bin/python scripts/smoke_test.py --model mistral --no-vllm
