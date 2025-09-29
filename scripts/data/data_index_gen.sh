DATASET=test
DATA_CONFIG="risk_prediction"
SAMPLE_NUM=500

DATA_CONFIG_PATH="./scripts/data_configs/${DATA_CONFIG}.json"
python ./mimiciv_dataset/data_index_gen.py \
  --data_index_dir ./datas/task_index/all \
  --subject_id_path ./datas/patient_data/${DATASET}.csv \
  --data_config ${DATA_CONFIG_PATH} \
  --output_path ./datas/task_index/${DATA_CONFIG}/${DATASET}_${DATA_CONFIG}_${SAMPLE_NUM}_force.csv \
  --force_task_num ${SAMPLE_NUM} \
  --balance_force