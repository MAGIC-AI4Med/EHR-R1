OUTPUT_ROOT="{YOUR_PATH_TO_SAVE_CHECKPOINT}"
LOG_ROOT="{YOUR_PATH_TO_LOG_TRAINING_PROCESS}"

DATA_PATH="{TRAINING_INDEX_FILE}"
DATA_NAME="{TRAINING_DATA_NAME}"

MODEL_PATH="{LLM_PATH}"
MODEL_NAME="{LLM_NAME}"

CKPT_NAME=${DATA_NAME}-${MODEL_NAME}

LOG_PATH=${LOG_ROOT}/${CKPT_NAME}
mkdir -p ${LOG_PATH}

export TOKENIZERS_PARALLELISM="false"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_PROJECT=EHR_Reasoning_Model

accelerate launch --config_file=./scripts/accelerate_configs/deepspeed_zero3.yaml --num_processes 8 \
    sft_reasoning.py \
    --bf16 True \
    --use_liger_kernel \
    --accelerator_config='{"split_batches": true, "dispatch_batches": true}' \
    --load_dataset_mode "lazzy" \
    --model_name_or_path ${MODEL_PATH} \
    --dataset_name ${DATA_PATH} \
    --output_dir ${OUTPUT_ROOT}/${CKPT_NAME} \
    --max_seq_length 12288 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 8 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 300 \
    --save_total_limit 10 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --dataset_num_proc 1 \
    --dataloader_num_workers 8 \
    --dataloader_pin_memory \
    --ignore_data_skip \
    --logging_dir ${LOG_PATH} \
    --report_to "wandb" 2>&1 | tee ${LOG_PATH}/train.log