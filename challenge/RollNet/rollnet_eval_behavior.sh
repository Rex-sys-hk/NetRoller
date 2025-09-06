# cd /path/to/VAD
# conda activate vad
# CUDA_VISIBLE_DEVICES=0 python tools/test.py ./projects/configs/VAD/VAD_base_stage_2.py ./ckpt/VAD_base.pth --launcher none --eval bbox --tmpdir tmp
folder_name=$1
kill -9 $(ps ax | grep 'rollnet_train.py' | fgrep -v grep | awk '{ print $1 }')
# log_folder="./outputs/250412131103_rolling_first"
log_folder="./outputs/${folder_name}"
GPU_num=`nvidia-smi --query-gpu=name --format=csv,noheader | wc -l`
time_stamp=`date '+%y%m%d%H%M%S'`
# export NCCL_SOCKET_IFNAME=eth0
echo $MASTER_ADDR
export MASTER_ADDR=127.0.0.1
echo $MASTER_PORT
echo $WORLD_SIZE
echo $RANK
# CUDA_VISIBLE_DEVICES=0 
# /mnt/csi-data-aly/shared/public/xinren/miniconda3/envs/llama_bev/bin/python \
# CUDA_LAUNCH_BLOCKING=1 
/mnt/csi-data-aly/shared/public/xinren/miniconda3/envs/llama_bev/bin/python -m torch.distributed.run \
    --nproc_per_node=$GPU_num \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    --nnodes=${WORLD_SIZE} \
    --node_rank=${RANK} \
    tools/rollnet_test.py \
    $log_folder/rollnet_behavior*.py \
    $log_folder/latest.pth \
    --eval bbox \
    --tmpdir $time_stamp \
    --launcher pytorch \
    # --launcher none 
