mkdir ~/.cache/clip &
cp /mnt/csi-data-aly/shared/public/xinren/clip/ViT-L-14.pt ~/.cache/clip/ViT-L-14.pt &&

./exps/finetune.sh \
/mnt/csi-data-aly/shared/public/xinren/Llama-2-7b \
/mnt/csi-data-aly/shared/public/xinren/DriveLM/challenge/llama_adapter_v2_multimodal7b/output/checkpoint-3.pth \
finetune_data_config.yaml \
/mnt/csi-data-aly/shared/public/xinren/DriveLM/challenge/llama_adapter_v2_multimodal7b/output


# /mnt/csi-data-aly/shared/public/xinren/DriveLM/challenge/llama_adapter_v2_multimodal7b/data/1bcbffc43484332672092e0024a8699a6eb5f558161aebf98a7c6b1db67224d1_LORA-BIAS-7B.pth \
# /path/to/llama_model_weights \
# /path/to/pre-trained/checkpoint.pth \
# finetune_data_config.yaml \
# /output/path