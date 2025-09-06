python tools/data_converter/vad_nuscenes_converter.py nuscenes \
    --root-path ./data/nuscenes \
    --out-dir ./data/nuscenes \
    --extra-tag rollnet_nuscenes_mini \
    --version v1.0-mini \
    --canbus ./data/nuscenes \
    --train_text_root_path ../data/v1_1_train_nus.json \
    --val_text_root_path ../data/v1_1_val_nus_q_only.json \
    --workers 8 