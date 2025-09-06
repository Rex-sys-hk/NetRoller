folder_name=$1
mkdir ~/.cache/clip &
cp ../../clip_model/ViT-L-14.pt ~/.cache/clip/ViT-L-14.pt &&

# /mnt/csi-data-aly/shared/public/xinren/miniconda3/envs/llama_adapter_v2/bin/python demo.py \
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0
export NCCL_DEBUG=INFO
python demo.py \
--llama_dir ../../Llama-2-7b \
--data ../test_llama.json  \
--output ../output_$folder_name.json \
--batch_size 1 \
--num_processes 1 \
--checkpoint ./output/checkpoint-3.pth \
# --checkpoint ../RollNet/outputs/$folder_name/llm.pth
# --checkpoint data/5088aeb63a89746b90bcfd5cb819e1c7411b2771b267c6d131ce73e250a8abf0_CAPTION-7B.pth
# --checkpoint data/1bcbffc43484332672092e0024a8699a6eb5f558161aebf98a7c6b1db67224d1_LORA-BIAS-7B.pth
# --checkpoint data/7fa55208379faf2dd862565284101b0e4a2a72114d6490a95e432cf9d9b6c813_BIAS-7B.pth
