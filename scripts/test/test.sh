
OUTPUT_ROOT="./results"

DATASET="/sfs/rhome/liaoyusheng/projects/EHR-R1/data/ehr_bench_decision_making.jsonl"
DATA_NAME="decision_making"

MODEL_PATH="/sfs/rhome/liaoyusheng/data/ShareModels/LLMs/EHR-R1-1.7B"
MODEL_NAME="ehr_r1_1.7b"

mkdir -p ${OUTPUT_ROOT}/${DATA_NAME}/${MODEL_NAME}
CUDA_VISIBLE_DEVICES=0 python test.py \
        --dataset_name ${DATASET} \
        --output_path ${OUTPUT_ROOT}/${DATA_NAME}/${MODEL_NAME} \
        --model_name_or_path ${MODEL_PATH} \
        --gpu_memory_utilization 0.85 \
        --max_seq_len 32000 \
        --direct_answer \
        --use_vllm \
        --batch 1 \
        --resume

# note that use `--propmt` only for baseline models