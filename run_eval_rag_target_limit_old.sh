# #!/bin/bash
# #SBATCH --job-name=eval
# #SBATCH --output=logs/eval_%j.out
# #SBATCH --error=logs/eval_%j.err
# #SBATCH --time=08:00:00
# #SBATCH --partition=gpu-preempt
# #SBATCH --gres=gpu:1
# #SBATCH --mem=128G
# #SBATCH -C vram80
# #SBATCH --cpus-per-task=4
# #SBATCH --mail-type=END,FAIL
# #SBATCH --mail-user=oyilmazel@umass.edu

# module load conda/latest
# module load cuda/12.6

# conda activate lamp

# NUM_CONTEXTS=10
# LIMIT_TARGET=1

# python evaluate_responses.py \
#  --cache_dir /scratch3/workspace/oyilmazel_umass_edu-lampqa_cache/ \
#  --evaluator_llm Qwen/Qwen2.5-32B-Instruct \
#  --inputs_addr data/processed/ae_processed_validation.json \
#  --response_addr data/out/rag/full_profile/ae_rag_full_profile_validation_${NUM_CONTEXTS}_limit_${LIMIT_TARGET}_output.json \
#  --score_addr data/scores/rag/full_profile/ae_rag_full_profile_validation_${NUM_CONTEXTS}_limit_${LIMIT_TARGET}_scores.json

# python evaluate_responses.py \
#  --cache_dir /scratch3/workspace/oyilmazel_umass_edu-lampqa_cache/ \
#  --evaluator_llm Qwen/Qwen2.5-32B-Instruct \
#  --inputs_addr data/processed/lp_processed_validation.json \
#  --response_addr data/out/rag/full_profile/lp_rag_full_profile_validation_${NUM_CONTEXTS}_limit_${LIMIT_TARGET}_output.json \
#  --score_addr data/scores/rag/full_profile/lp_rag_full_profile_validation_${NUM_CONTEXTS}_limit_${LIMIT_TARGET}_scores.json

# python evaluate_responses.py \
#  --cache_dir /scratch3/workspace/oyilmazel_umass_edu-lampqa_cache/ \
#  --evaluator_llm Qwen/Qwen2.5-32B-Instruct \
#  --inputs_addr data/processed/sc_processed_validation.json \
#  --response_addr data/out/rag/full_profile/sc_rag_full_profile_validation_${NUM_CONTEXTS}_limit_${LIMIT_TARGET}_output.json \
#  --score_addr data/scores/rag/full_profile/sc_rag_full_profile_validation_${NUM_CONTEXTS}_limit_${LIMIT_TARGET}_scores.json


# python evaluate_responses.py \
#  --cache_dir /scratch3/workspace/oyilmazel_umass_edu-lampqa_cache/ \
#  --evaluator_llm Qwen/Qwen2.5-32B-Instruct \
#  --inputs_addr data/processed/ae_processed_test.json \
#  --response_addr data/out/rag/full_profile/ae_rag_full_profile_test_${NUM_CONTEXTS}_limit_${LIMIT_TARGET}_output.json \
#  --score_addr data/scores/rag/full_profile/ae_rag_full_profile_test_${NUM_CONTEXTS}_limit_${LIMIT_TARGET}_scores.json

# python evaluate_responses.py \
#  --cache_dir /scratch3/workspace/oyilmazel_umass_edu-lampqa_cache/ \
#  --evaluator_llm Qwen/Qwen2.5-32B-Instruct \
#  --inputs_addr data/processed/lp_processed_test.json \
#  --response_addr data/out/rag/full_profile/lp_rag_full_profile_test_${NUM_CONTEXTS}_limit_${LIMIT_TARGET}_output.json \
#  --score_addr data/scores/rag/full_profile/lp_rag_full_profile_test_${NUM_CONTEXTS}_limit_${LIMIT_TARGET}_scores.json

# python evaluate_responses.py \
#  --cache_dir /scratch3/workspace/oyilmazel_umass_edu-lampqa_cache/ \
#  --evaluator_llm Qwen/Qwen2.5-32B-Instruct \
#  --inputs_addr data/processed/sc_processed_test.json \
#  --response_addr data/out/rag/full_profile/sc_rag_full_profile_test_${NUM_CONTEXTS}_limit_${LIMIT_TARGET}_output.json \
#  --score_addr data/scores/rag/full_profile/sc_rag_full_profile_test_${NUM_CONTEXTS}_limit_${LIMIT_TARGET}_scores.json
