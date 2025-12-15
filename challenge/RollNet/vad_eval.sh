# cd /path/to/VAD
# conda activate vad
# CUDA_VISIBLE_DEVICES=0 python tools/test.py ./projects/configs/VAD/VAD_base_stage_2.py ./ckpt/VAD_base.pth --launcher none --eval bbox --tmpdir tmp
# CUDA_VISIBLE_DEVICES=0 /mnt/csi-data-aly/shared/public/xinren/miniconda3/envs/llama_bev/bin/python tools/test.py ./projects/configs/VAD/VAD_base_stage_2.py ./ckpt/VAD_base.pth --launcher none --eval bbox --tmpdir tmp
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
    ./projects/configs/VAD/VAD_base_stage_2.py \
    ./ckpt/VAD_base.pth \
    --launcher none \
    --eval bbox \
    --tmpdir tmp
