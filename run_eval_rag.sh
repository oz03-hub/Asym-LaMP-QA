#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
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
ENTROPY=80

# Asym Profile
python evaluate_responses.py \
 --cache_dir /scratch3/workspace/oyilmazel_umass_edu-lampqa_cache/ \
 --evaluator_llm Qwen/Qwen2.5-32B-Instruct \
 --inputs_addr data/processed/ae_processed_test.json \
 --response_addr data/out/rag/asym/ae_rag_lp_sc_test_${ENTROPY}_${NUM_CONTEXTS}_output.json \
 --score_addr data/scores/rag/asym/ae_rag_lp_sc_${ENTROPY}_${NUM_CONTEXTS}.json

python evaluate_responses.py\
 --cache_dir /scratch3/workspace/oyilmazel_umass_edu-lampqa_cache/ \
 --evaluator_llm Qwen/Qwen2.5-32B-Instruct \
 --inputs_addr data/processed/lp_processed_test.json \
 --response_addr data/out/rag/asym/lp_rag_ae_sc_test_${ENTROPY}_${NUM_CONTEXTS}_output.json \
 --score_addr data/scores/rag/asym/lp_rag_ae_sc_${ENTROPY}_${NUM_CONTEXTS}.json

python evaluate_responses.py \
 --cache_dir /scratch3/workspace/oyilmazel_umass_edu-lampqa_cache/ \
 --evaluator_llm Qwen/Qwen2.5-32B-Instruct \
 --inputs_addr data/processed/sc_processed_test.json \
 --response_addr data/out/rag/asym/sc_rag_ae_lp_test_${ENTROPY}_${NUM_CONTEXTS}_output.json \
 --score_addr data/scores/rag/asym/sc_rag_ae_lp_${ENTROPY}_${NUM_CONTEXTS}.json

# Full Profile
python evaluate_responses.py \
 --cache_dir /scratch3/workspace/oyilmazel_umass_edu-lampqa_cache/ \
 --evaluator_llm Qwen/Qwen2.5-32B-Instruct \
 --inputs_addr data/processed/ae_processed_test.json \
 --response_addr data/out/rag/full_profile/ae_rag_full_profile_test_${ENTROPY}_${NUM_CONTEXTS}_output.json \
 --score_addr data/scores/rag/full_profile/ae_full_profile_test_${ENTROPY}_${NUM_CONTEXTS}.json

python evaluate_responses.py \
 --cache_dir /scratch3/workspace/oyilmazel_umass_edu-lampqa_cache/ \
 --evaluator_llm Qwen/Qwen2.5-32B-Instruct \
 --inputs_addr data/processed/lp_processed_test.json \
 --response_addr data/out/rag/full_profile/lp_rag_full_profile_test_${ENTROPY}_${NUM_CONTEXTS}_output.json \
 --score_addr data/scores/rag/full_profile/lp_full_profile_test_${ENTROPY}_${NUM_CONTEXTS}.json

python evaluate_responses.py \
 --cache_dir /scratch3/workspace/oyilmazel_umass_edu-lampqa_cache/ \
 --evaluator_llm Qwen/Qwen2.5-32B-Instruct \
 --inputs_addr data/processed/sc_processed_test.json \
 --response_addr data/out/rag/full_profile/sc_rag_full_profile_test_${ENTROPY}_${NUM_CONTEXTS}_output.json \
 --score_addr data/scores/rag/full_profile/sc_full_profile_test_${ENTROPY}_${NUM_CONTEXTS}.json
