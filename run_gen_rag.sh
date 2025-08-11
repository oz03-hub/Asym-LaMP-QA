#!/bin/bash
#SBATCH --job-name=rag_gen_exp
#SBATCH --output=logs/rag_gen_%j.out
#SBATCH --error=logs/rag_gen_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH -C vram80
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=oyilmazel@umass.edu


module load conda/latest
module load cuda/12.6

conda activate lamp

NUM_CONTEXTS=10
ENTROPY_PERCENTILE=80

# Full Profile
python -B asymmetric_baselines.py --cache_dir /scratch3/workspace/oyilmazel_umass_edu-lampqa_cache/ \
 --keep_domains ae,lp,sc \
 --model_addr Qwen/Qwen2.5-7B-Instruct \
 --inputs_addr data/processed/ae_processed_test.json \
 --output_addr data/out/rag/full_profile/ae_rag_full_profile_test_${ENTROPY_PERCENTILE}_${NUM_CONTEXTS}_output.json \
 --num_contexts "${NUM_CONTEXTS}" --rag \
 --entropy_percentile "${ENTROPY_PERCENTILE}"

python -B asymmetric_baselines.py --cache_dir /scratch3/workspace/oyilmazel_umass_edu-lampqa_cache/ \
  --keep_domains ae,lp,sc \
  --model_addr Qwen/Qwen2.5-7B-Instruct \
  --inputs_addr data/processed/lp_processed_test.json \
  --output_addr data/out/rag/full_profile/lp_rag_full_profile_test_${ENTROPY_PERCENTILE}_${NUM_CONTEXTS}_output.json \
  --num_contexts "${NUM_CONTEXTS}" --rag \
  --entropy_percentile "${ENTROPY_PERCENTILE}"

python -B asymmetric_baselines.py --cache_dir /scratch3/workspace/oyilmazel_umass_edu-lampqa_cache/ \
  --keep_domains ae,lp,sc \
  --model_addr Qwen/Qwen2.5-7B-Instruct \
  --inputs_addr data/processed/sc_processed_test.json \
  --output_addr data/out/rag/full_profile/sc_rag_full_profile_test_${ENTROPY_PERCENTILE}_${NUM_CONTEXTS}_output.json \
  --num_contexts "${NUM_CONTEXTS}" --rag \
  --entropy_percentile "${ENTROPY_PERCENTILE}"

# Asym
python -B asymmetric_baselines.py --cache_dir /scratch3/workspace/oyilmazel_umass_edu-lampqa_cache/ \
  --keep_domains lp,sc \
  --model_addr Qwen/Qwen2.5-7B-Instruct \
  --inputs_addr data/processed/ae_processed_test.json \
  --output_addr data/out/rag/asym/ae_rag_lp_sc_test_${ENTROPY_PERCENTILE}_${NUM_CONTEXTS}_output.json \
  --num_contexts "${NUM_CONTEXTS}" --rag \
  --entropy_percentile "${ENTROPY_PERCENTILE}"

python -B asymmetric_baselines.py --cache_dir /scratch3/workspace/oyilmazel_umass_edu-lampqa_cache/ \
  --keep_domains ae,sc \
  --model_addr Qwen/Qwen2.5-7B-Instruct \
  --inputs_addr data/processed/lp_processed_test.json \
  --output_addr data/out/rag/asym/lp_rag_ae_sc_test_${ENTROPY_PERCENTILE}_${NUM_CONTEXTS}_output.json \
  --num_contexts "${NUM_CONTEXTS}" --rag \
  --entropy_percentile "${ENTROPY_PERCENTILE}"

python -B asymmetric_baselines.py --cache_dir /scratch3/workspace/oyilmazel_umass_edu-lampqa_cache/ \
  --keep_domains ae,lp \
  --model_addr Qwen/Qwen2.5-7B-Instruct \
  --inputs_addr data/processed/sc_processed_test.json \
  --output_addr data/out/rag/asym/sc_rag_ae_lp_test_${ENTROPY_PERCENTILE}_${NUM_CONTEXTS}_output.json \
  --num_contexts "${NUM_CONTEXTS}" --rag \
  --entropy_percentile "${ENTROPY_PERCENTILE}"
