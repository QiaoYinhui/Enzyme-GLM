#! /bin/bash

script_path=$(realpath $BASH_SOURCE)  #realpath,获取给定文件的绝对路径 $BASH_SOURCE：包含当前脚本文件的路径.
                                      # 获取当前运行的Bash脚本文件的绝对路径，并将其存储在 script_path 变量中，以便后续在脚本中使用
script_dir=$(dirname $script_path)    #dirname,提取文件路径的目录的部分  $script_path:包含当前运行的脚本文件的绝对路径
                                      # 获取当前运行的Bash脚本文件的目录路径，并将其存储在 script_dir 变量中,#62500+3100*10=93500.train_iters

config_json="$script_dir/config_block_large.json"
gpt_options=" \
       --block-lm \
       --bert-prob 0.5 \
       --avg-block-length 3 \
       --experiment-name Enzyme-GLM-base-1024 \
       --model-parallel-size ${MP_SIZE} \
       --num-layers 12 \
       --hidden-size 768 \
       --num-attention-heads 12 \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --save /home/qyh_vm/GLM_Linux/save/data/checkpoints/Enzyme-GLM-base-1024_iter15000 \
       --train-iters 15000 \
       --resume-dataloader \
       --train-data wudao \
       --tokenizer-type GPT2BPETokenizer \
       --tokenizer-model-type gpt2\
       --split 945,50,1 \
       --distributed-backend nccl \
       --lr-decay-style cosine \
       --lr-decay-iters 160000 \
       --lr-decay-ratio 0.05 \
       --warmup .05 \
       --checkpoint-activations \
       --deepspeed-activation-checkpointing \
       --fp16 \
       --batch-size 16 \
       --sample-one-document \
       --task-mask \
       --summary-dir /home/qyh_vm/GLM_Linux/save/data/summer \
"
gpt_options="${gpt_options}
               --deepspeed \
               --deepspeed_config ${config_json} \
"
