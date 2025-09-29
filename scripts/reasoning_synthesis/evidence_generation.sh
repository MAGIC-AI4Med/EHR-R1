WORK_NUM=10
DATA_DIR={YOUR_INDEX_FILE}

for WORK_IDX in $(seq 0 $((WORK_NUM-1)))
do
    echo ${WORK_IDX}
    python evidence_preprocess/evidence_generation.py \
        --data_index_path ${DATA_DIR} \
        --chunk_num ${WORK_NUM} \
        --chunk_idx ${WORK_IDX} \
        --threshold 5 &
done