# cd /path/to/VAD
# conda activate vad
python -m torch.distributed.run --nproc_per_node=1 --master_port=2333 tools/train.py ./projects/configs/VAD/VAD_base_e2e.py --launcher pytorch --deterministic --work-dir ./outputs