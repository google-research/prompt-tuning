#!/usr/bin/env bash

EVAL_DIR=${1:-${EVAL_DIR}}
TFDS_DATA_DIR=${2:-${TFDS_DATA_DIR}}

if [ -z ${EVAL_DIR} ] || [ -z ${TFDS_DATA_DIR} ]; then
  echo "usage: ./sst2-demo-eval.sh gs://your-bucket/path/to/eval_dir gs://your-bucket/path/to/tfds/cache"
  exit 1
fi

T5X_DIR="`python3 -m prompt_tuning.scripts.find_module t5x`/.."
FLAXFORMER_DIR="`python3 -m prompt_tuning.scripts.find_module flaxformer`/.."
PROMPT_DIR="`python3 -m prompt_tuning.scripts.find_module prompt_tuning`/.."
echo "Searching for gin configs in:"
echo "- ${T5X_DIR}"
echo "- ${FLAXFORMER_DIR}"
echo "- ${PROMPT_DIR}"
echo "============================="
PRETRAINED_MODEL="gs://t5-data/pretrained_models/t5x/t5_1_1_lm100k_base/checkpoint_1100000"
PROMPT_FILE="`python3 -m prompt_tuning.scripts.find_module prompt_tuning`/pretrained_prompts/t5_1_1_lm100k_base/sst2.npy"

python3 -m t5x.eval \
  --gin_search_paths="${T5X_DIR},${FLAXFORMER_DIR},${PROMPT_DIR}" \
  --gin_file="prompt_tuning/configs/models/t5_1_1_base_prompt.gin" \
  --gin_file="prompt_tuning/configs/runs/prompt_eval.gin" \
  --gin.MIXTURE_OR_TASK_NAME="'taskless_glue_sst2_v200_examples'" \
  --gin.MIXTURE_OR_TASK_MODULE="'prompt_tuning.data.glue'" \
  --gin.TASK_FEATURE_LENGTHS="{'inputs': 512, 'targets': 8}" \
  --gin.CHECKPOINT_PATH="'${PRETRAINED_MODEL}'" \
  --gin.EVAL_OUTPUT_DIR="'${EVAL_DIR}'" \
  --gin.utils.DatasetConfig.split="'validation'" \
  --gin.utils.DatasetConfig.batch_size="128" \
  --gin.USE_CACHED_TASKS="False" \
  --gin.PROMPT_FILE="'${PROMPT_FILE}'" \
  --tfds_data_dir=${TFDS_DATA_DIR}
