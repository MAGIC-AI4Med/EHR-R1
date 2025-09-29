
OUTPUT_ROOT="{YOUR_PATH_TO_SAVE_RESULTS}"

DATASET="{TRAINING_INDEX_FILE}"
DATA_NAME="{TRAINING_DATA_NAME}"

MODEL_PATH="{LLM_PATH}"
MODEL_NAME="{LLM_NAME}"

mkdir -p ${OUTPUT_ROOT}/${DATA_NAME}/${MODEL_NAME}
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py \
        --dataset_name ${DATASET} \
        --output_path ${OUTPUT_ROOT}/${DATA_NAME}/${MODEL_NAME} \
        --model_name_or_path ${MODEL_PATH} \
        --gpu_memory_utilization 0.85 \
        --max_seq_len 32000 \
        --use_vllm \
        --batch 1 \
        --prompt \
        --resume