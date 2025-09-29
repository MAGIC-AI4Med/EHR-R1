

DATASET=patient
python mimiciv_dataset/task_sample_info_gen.py \
  --patient_id ./datas/patients.csv \
  --output_path ./datas/task_index/${DATASET} \
  --group "patient" \
