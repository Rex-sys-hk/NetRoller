export MASTER_ADDR=127.0.0.1
export MASTER_PORT=1112
export WORLD_SIZE=1
export RANK=0
bash ./rollnet_train.sh $1
# bash ./rollnet_train.sh rollnet/rollnet_base.py