#!/bin/bash
source $1

MPSIZE=1
MAXSEQLEN=1024
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
            --max-position-embeddings 1024 \
            --tokenizer-model-type gpt2 \
            --tokenizer-type GPT2BPETokenizer \
            --load-pretrained /mnt/d/111QYH/progect/GLM/generate_task/Enzyme-GLM-304M-s2eV2_2-8_10e01-22-13-02 \
            --task-mask" 

python -m torch.distributed.launch --nproc_per_node=$MPSIZE --master_port $MASTER_PORT generate_Pro_uncondition_gen.py \
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
