#! /bin/bash

# Change for multinode config

NUM_WORKERS=1
NUM_GPUS_PER_WORKER=1
MP_SIZE=1   #并行数
MASTER_PORT=$(shuf -n 1 -i 10000-65535) #从10000到65535中间随机生成一个端口号，然后赋给MASTER_PORT

source $1
DATESTR=$(date +"%m-%d-%H-%M")  #读取当前的时间

OPTIONS_NCCL="NCCL_DEBUG=info NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2"
HOST_FILE_PATH="/workspace/hostfile"

mkdir logs
run_cmd="${OPTIONS_NCCL} deepspeed --master_port ${MASTER_PORT} --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER} --hostfile ${HOST_FILE_PATH} pretrain_glm.py ${gpt_options} 2>&1 | tee logs/log-${DATESTR}.txt"
echo ${run_cmd}
eval ${run_cmd}

set +x
