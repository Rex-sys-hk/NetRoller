folder_name=$1
log_folder="./outputs/${folder_name}"

/mnt/csi-data-aly/shared/public/xinren/miniconda3/envs/llama_bev/bin/python tools/extract_llm.py \
--model_path $log_folder/latest.pth \
--save_path $log_folder/llm.pth