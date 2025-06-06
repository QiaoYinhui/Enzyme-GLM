#!/bin/bash
CHECKPOINT_PATH=/mnt/d/111QYH/progect/GLM/训练记录

source $1

MPSIZE=1
MAXSEQLEN=512
MASTER_PORT=$(shuf -n 1 -i 10000-65535)

#SAMPLING ARGS
TEMP=0.9
#If TOPK/TOPP are 0 it defaults to greedy sampling, top-k will also override top-p
TOPK=40
TOPP=0

#script_path=$(realpath $0)
#script_dir=$(dirname $script_path)

#config_json="$script_dir/ds_config.json"

MODEL_ARGS="--block-lm \
            --num-layers 24 \
            --hidden-size 1024 \
            --num-attention-heads 16 \
            --max-position-embeddings 512 \
            --tokenizer-model-type gpt2 \
            --tokenizer-type GPT2BPETokenizer \
            --load-pretrained ${CHECKPOINT_PATH}/GLM_Enzyme-large-300w \
            --task-mask" 

python -m torch.distributed.launch --nproc_per_node=$MPSIZE --master_port $MASTER_PORT generate_samples.py \
       --DDP-impl none \
       --model-parallel-size $MPSIZE \
       $MODEL_ARGS \
       --fp16 \
       --cache-dir cache \
       --out-seq-length $MAXSEQLEN \
       --seq-length 512 \
       --temperature $TEMP \
       --top-k $TOPK \
       --top-p $TOPP
