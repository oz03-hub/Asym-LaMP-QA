#!/bin/bash
#SBATCH --job-name=rag_gen_exp
#SBATCH --output=logs/rag_gen_%j.out
#SBATCH --error=logs/rag_gen_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH -C vram80
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=oyilmazel@umass.edu


module load conda/latest
module load cuda/12.6

echo "Job started on $(hostname)"
nvidia-smi
lscpu

conda activate lamp

NUM_CONTEXTS=10
LIMIT_TARGET=1

python -B asymmetric_baselines.py --cache_dir /scratch3/workspace/oyilmazel_umass_edu-lampqa_cache/ \
 --keep_domains ae,lp,sc \
 --model_addr Qwen/Qwen2.5-7B-Instruct \
 --inputs_addr data/processed/ae_processed_train.json \
 --output_addr data/out/rag/full_profile/ae_rag_full_profile_train_${NUM_CONTEXTS}_limit_${LIMIT_TARGET}_output.json \
 --num_contexts "${NUM_CONTEXTS}" --rag \
 --limit_target "${LIMIT_TARGET}"

python -B asymmetric_baselines.py --cache_dir /scratch3/workspace/oyilmazel_umass_edu-lampqa_cache/ \
  --keep_domains ae,lp,sc \
  --model_addr Qwen/Qwen2.5-7B-Instruct \
  --inputs_addr data/processed/lp_processed_train.json \
  --output_addr data/out/rag/full_profile/lp_rag_full_profile_train_${NUM_CONTEXTS}_limit_${LIMIT_TARGET}_output.json \
  --num_contexts "${NUM_CONTEXTS}" --rag \
  --limit_target "${LIMIT_TARGET}"

python -B asymmetric_baselines.py --cache_dir /scratch3/workspace/oyilmazel_umass_edu-lampqa_cache/ \
  --keep_domains ae,lp,sc \
  --model_addr Qwen/Qwen2.5-7B-Instruct \
  --inputs_addr data/processed/sc_processed_train.json \
  --output_addr data/out/rag/full_profile/sc_rag_full_profile_train_${NUM_CONTEXTS}_limit_${LIMIT_TARGET}_output.json \
  --num_contexts "${NUM_CONTEXTS}" --rag \
  --limit_target "${LIMIT_TARGET}"


# python -B asymmetric_baselines.py --cache_dir /scratch3/workspace/oyilmazel_umass_edu-lampqa_cache/ \
#  --keep_domains ae,lp,sc \
#  --model_addr Qwen/Qwen2.5-7B-Instruct \
#  --inputs_addr data/processed/ae_processed_validation.json \
#  --output_addr data/out/rag/full_profile/ae_rag_full_profile_validation_${NUM_CONTEXTS}_limit_${LIMIT_TARGET}_output.json \
#  --num_contexts "${NUM_CONTEXTS}" --rag \
#  --limit_target "${LIMIT_TARGET}"

# python -B asymmetric_baselines.py --cache_dir /scratch3/workspace/oyilmazel_umass_edu-lampqa_cache/ \
#   --keep_domains ae,lp,sc \
#   --model_addr Qwen/Qwen2.5-7B-Instruct \
#   --inputs_addr data/processed/lp_processed_validation.json \
#   --output_addr data/out/rag/full_profile/lp_rag_full_profile_validation_${NUM_CONTEXTS}_limit_${LIMIT_TARGET}_output.json \
#   --num_contexts "${NUM_CONTEXTS}" --rag \
#   --limit_target "${LIMIT_TARGET}"

# python -B asymmetric_baselines.py --cache_dir /scratch3/workspace/oyilmazel_umass_edu-lampqa_cache/ \
#   --keep_domains ae,lp,sc \
#   --model_addr Qwen/Qwen2.5-7B-Instruct \
#   --inputs_addr data/processed/sc_processed_validation.json \
#   --output_addr data/out/rag/full_profile/sc_rag_full_profile_validation_${NUM_CONTEXTS}_limit_${LIMIT_TARGET}_output.json \
#   --num_contexts "${NUM_CONTEXTS}" --rag \
#   --limit_target "${LIMIT_TARGET}"


# Full Profile
# python -B asymmetric_baselines.py --cache_dir /scratch3/workspace/oyilmazel_umass_edu-lampqa_cache/ \
#  --keep_domains ae,lp,sc \
#  --model_addr Qwen/Qwen2.5-7B-Instruct \
#  --inputs_addr data/processed/ae_processed_test.json \
#  --output_addr data/out/rag/full_profile/ae_rag_full_profile_test_${NUM_CONTEXTS}_limit_${LIMIT_TARGET}_output.json \
#  --num_contexts "${NUM_CONTEXTS}" --rag \
#  --limit_target "${LIMIT_TARGET}"

# python -B asymmetric_baselines.py --cache_dir /scratch3/workspace/oyilmazel_umass_edu-lampqa_cache/ \
#   --keep_domains ae,lp,sc \
#   --model_addr Qwen/Qwen2.5-7B-Instruct \
#   --inputs_addr data/processed/lp_processed_test.json \
#   --output_addr data/out/rag/full_profile/lp_rag_full_profile_test_${NUM_CONTEXTS}_limit_${LIMIT_TARGET}_output.json \
#   --num_contexts "${NUM_CONTEXTS}" --rag \
#   --limit_target "${LIMIT_TARGET}"

# python -B asymmetric_baselines.py --cache_dir /scratch3/workspace/oyilmazel_umass_edu-lampqa_cache/ \
#   --keep_domains ae,lp,sc \
#   --model_addr Qwen/Qwen2.5-7B-Instruct \
#   --inputs_addr data/processed/sc_processed_test.json \
#   --output_addr data/out/rag/full_profile/sc_rag_full_profile_test_${NUM_CONTEXTS}_limit_${LIMIT_TARGET}_output.json \
#   --num_contexts "${NUM_CONTEXTS}" --rag \
#   --limit_target "${LIMIT_TARGET}"
