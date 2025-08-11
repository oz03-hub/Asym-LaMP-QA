#!/bin/bash
#SBATCH --job-name=asym_exp
#SBATCH --output=logs/asym_%j.out
#SBATCH --error=logs/asym_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu-preempt
#SBATCH --gpus=a100:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4

module load conda/latest
module load cuda/12.6

conda activate lamp

python asymmetric_baselines.py --cache_dir /work/pi_hzamani_umass_edu/ozel_cache/ \
  --keep_domains lp,sc \
  --model_addr Qwen/Qwen2.5-7B-Instruct \
  --inputs_addr data/processed/ae_processed_test.json \
  --output_addr data/out/rag/asym/test_ae_output.json --num_contexts 2 --rag

python asymmetric_baselines.py --cache_dir /work/pi_hzamani_umass_edu/ozel_cache/ \
  --keep_domains ae,lp,sc \
  --model_addr Qwen/Qwen2.5-7B-Instruct \
  --inputs_addr data/processed/ae_processed_test.json \
  --output_addr data/out/rag/full_profile/test_ae_output.json --num_contexts 2 --rag
