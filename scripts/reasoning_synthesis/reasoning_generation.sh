DATA_INDEX_DIR={YOUR_EVIDECNCE_FILE}

python evidence_preprocess/reasoning_generation.py \
    --data_index_dir ${DATA_INDEX_DIR} \
    --filter_nograph \
    --without_knowledge \
    --model "gpt-4o" \
    --num_worker 20
