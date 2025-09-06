mkdir ~/.cache/clip &
cp /mnt/csi-data-aly/shared/public/xinren/clip/ViT-L-14.pt ~/.cache/clip/ViT-L-14.pt &&

./exps/pretrain.sh \
/mnt/csi-data-aly/shared/public/xinren/Llama-2-7b \
pretrain_data_config.yaml \
/mnt/csi-data-aly/shared/public/xinren/DriveLM/challenge/llama_adapter_v2_multimodal7b/output

# /mnt/csi-data-aly/shared/public/xinren/DriveLM/challenge/llama_adapter_v2_multimodal7b/data/1bcbffc43484332672092e0024a8699a6eb5f558161aebf98a7c6b1db67224d1_LORA-BIAS-7B.pth \