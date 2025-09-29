
OUTPUT_ROOT="{YOUR_PATH_TO_SAVE_RESULTS}"

DATASET="{TRAINING_INDEX_FILE}"
DATA_NAME="{TRAINING_DATA_NAME}"

MODEL_PATH="{LLM_PATH}"
MODEL_NAME="{LLM_NAME}"

TP=2 # size of tensor parallel
DP=4 # size of data parallel

CHUNK_NUM=8
mapfile -t GPU_IDS < <(nvidia-smi --query-gpu=index --format=csv,noheader)
TOTAL_GPUS=${#GPU_IDS[@]}
mkdir -p ${OUTPUT_ROOT}/${DATA_NAME}/${MODEL_NAME}

for ((CHUNK_IDX=0; CHUNK_IDX<DP; CHUNK_IDX++)); do
    START_GPU_ID=$((CHUNK_IDX * TP))
    slice=( "${GPU_IDS[@]:START_GPU_ID:TP}" )
    IFS=, gpu_list="${slice[*]}"

    echo "Launching process $CHUNK_IDX on GPUs: $gpu_list"
    CUDA_VISIBLE_DEVICES="$gpu_list" python test.py \
        --dataset_name ${DATASET} \
        --output_path ${OUTPUT_ROOT}/${DATA_NAME}/${MODEL_NAME} \
        --model_name_or_path ${MODEL_PATH} \
        --chunk_num ${CHUNK_NUM} \
        --chunk_idx $((CHUNK_IDX+4)) \
        --use_vllm \
        --batch 1 \
        --prompt \
        --resume | tee ${OUTPUT_ROOT}/${DATA_NAME}/${MODEL_NAME}/infer_${CHUNK_NUM}_${CHUNK_IDX}.log &
#     sleep 10
done