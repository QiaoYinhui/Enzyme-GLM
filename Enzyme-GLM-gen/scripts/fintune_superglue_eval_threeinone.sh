#! /bin/bash
# Change for multinode config

DATA_ROOT=/home/qyh_vm/GLM_Linux/data 
#CHECKPOINT_PATH=/home/qyh_vm/GLM_Linux/data/trained_model
CHECKPOINT_PATH=none 
SAVE_PATH=/home/qyh_vm/GLM_Linux/data/trained_model

#model_blockm_large.sh
MODEL_TYPE="ENZYME-GLM"
MODEL_ARGS="--block-lm \
            --num-layers 16 \
            --hidden-size 1024 \
            --num-attention-heads 16 \
            --max-position-embeddings 324 \
            --tokenizer-model-type gpt2 \
            --tokenizer-type GPT2BPETokenizer \
            --load-pretrained ${CHECKPOINT_PATH}/none" 

#tsak_ESPJ.sh
EXPERIMENT_NAME=${MODEL_TYPE}-2024.01.22-testinglmdata 
TASK_NAME=esp-judgement
DATA_PATH="${DATA_ROOT}/my_fin_data"
MAX_SEQ_LEN=324

LR_SINGLE=0 #0
#LR_SINGLE=1e-4
#EPOCH_SINGLE=50
#XXLARGE_EPOCH=50
EPOCH_SINGLE=0 #30
XXLARGE_EPOCH=0 #0

TRAIN_ARGS="--lr-decay-style linear \
            --warmup 0 \
            --weight-decay 0 \
            --pattern-id 0 \
	          --train-iters 0
	          --task-mask"              

#TRAIN_ARGS="--lr-decay-style linear \
#            --lr-decay-ratio 0.01
#            --warmup 0.05 \
#            --weight-decay 5.0e-2 \
#            --pattern-id 0 \
#	          --train-iters 20000
#	          --task-mask"

COMMON_ARGS="--save-interval 1000 \
             --log-interval 0 \
             --eval-interval 0 \
             --eval-iters 0"        

#PATTERN_IDS=(0 1 2 3)
PATTERN_IDS=(0)                       
PROMPT_IDS=(1 2 3)                    
 
BATCH_SIZE=16                         

#finetune_superglue.sh

source $1    # Model
source $2    # Task

if [ -z $N_GPU ];then
  N_GPU=1
fi

MASTER_PORT=$(shuf -n 1 -i 10000-65535)
DISTRIBUTED_ARGS="--nproc_per_node ${N_GPU} --nnodes 1 --node_rank 0 --master_addr localhost --master_port $MASTER_PORT"

PER_GPU_BS=$(($BATCH_SIZE/$N_GPU))
DATESTR=$(date +"%m-%d-%H-%M")
EXPERIMENT_NAME=${EXPERIMENT_NAME}-${DATESTR}

mkdir logs
python -m torch.distributed.launch $DISTRIBUTED_ARGS test_finetune_glm.py \
       --finetune \
       --cloze-eval \
       --experiment-name ${EXPERIMENT_NAME} \
       --task ${TASK_NAME} \
       --data-dir ${DATA_PATH} \
       --save ${SAVE_PATH} \
       --seq-length ${MAX_SEQ_LEN} \
       --checkpoint-activations \
       --eval-batch-size 16 \
       --save-epoch 10000 \
       $MODEL_ARGS \
       $TRAIN_ARGS \
       $COMMON_ARGS \
       --fp16 \
       --batch-size ${PER_GPU_BS} \
       --epochs ${EPOCH_SINGLE} \
       --lr ${LR_SINGLE} \
       --overwrite \
       --summary-dir ${SAVE_PATH}/summmary \
       2>&1 | tee logs/log-${EXPERIMENT_NAME}.txt
