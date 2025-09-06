# cd /path/to/VAD
# conda activate vad
kill -9 $(ps ax | grep 'rollnet_train.py' | fgrep -v grep | awk '{ print $1 }')
time_stamp=`date '+%y%m%d%H%M%S'`
GPU_num=`nvidia-smi --query-gpu=name --format=csv,noheader | wc -l`
config_path=./projects/configs/$1
# export NCCL_SOCKET_IFNAME=eth0
echo $MASTER_ADDR
export MASTER_ADDR=127.0.0.1
echo $MASTER_PORT
echo $WORLD_SIZE
echo $RANK
# export NCCL_DEBUG=INFO
# export TORCH_DISTRIBUTED_DEBUG=INFO
# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_MIN_NRINGS=1
# GPU_num=1
# CUDA_VISIBLE_DEVICES=0,1 \
# /mnt/csi-data-aly/shared/public/xinren/miniconda3/envs/llama_bev/bin/deepspeed --num_gpus=$GPU_num\
# python -m torch.distributed.run \
/mnt/csi-data-aly/shared/public/xinren/miniconda3/envs/llama_bev/bin/python -m torch.distributed.run \
    --nproc_per_node=$GPU_num \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    --nnodes=${WORLD_SIZE} \
    --node_rank=${RANK} \
    tools/rollnet_train.py \
        $config_path \
    --deterministic \
    --work-dir ./outputs/$time_stamp \
    --launcher pytorch
    # --launcher deepspeed

    # ./projects/configs/rollnet/rollnet_base.py \
# kill -9 $(ps ax | grep 'rollnet_train.py' | fgrep -v grep | awk '{ print $1 }')