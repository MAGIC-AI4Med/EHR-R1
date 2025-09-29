OUTPUT_ROOT="{YOUR_PATH_TO_SAVE_RESULTS}"

DATASET="{TRAINING_INDEX_FILE}"
DATA_NAME="{TRAINING_DATA_NAME}"

MODEL_PATH="{LLM_PATH}"
MODEL_NAME="{LLM_NAME}"

mkdir -p ${OUTPUT_ROOT}/${DATA_NAME}/${MODEL_NAME}
CHUNK_NUM=8
for CUDA_ID in $(seq 0 $((CHUNK_NUM - 1)))
do
    echo "Run on GPU $((CUDA_ID))"
    CUDA_VISIBLE_DEVICES=$((CUDA_ID)) python test.py \
        --dataset_name ${DATASET} \
        --output_path ${OUTPUT_ROOT}/${DATA_NAME}/${MODEL_NAME} \
        --model_name_or_path ${MODEL_PATH} \
        --use_vllm \
        --batch 1 \
        --sample_num 1 \
        --temperature 0.0 \
        --prompt \
        --lazzy_mode \
        --chunk_num ${CHUNK_NUM} \
        --chunk_idx ${CUDA_ID} \
        --resume 2>&1 | tee ${OUTPUT_ROOT}/${DATA_NAME}/${MODEL_NAME}/infer_${CHUNK_NUM}_${CUDA_ID}.log &
done