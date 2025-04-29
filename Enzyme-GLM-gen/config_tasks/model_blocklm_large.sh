#MODEL_TYPE="blank-large"
#MODEL_ARGS="--block-lm \
#            --num-layers 24 \
#            --hidden-size 1024 \
#            --num-attention-heads 16 \
#            --max-position-embeddings 512 \
#            --tokenizer-model-type bert-large-uncased \
#            --tokenizer-type BertWordPieceTokenizer \
#--load-pretrained ${CHECKPOINT_PATH}/blocklm-large-blank"
#new
MODEL_TYPE="ENZYME-GLM"
MODEL_ARGS="--block-lm \
            --num-layers 4 \
            --hidden-size 4096 \
            --num-attention-heads 4 \
            --max-position-embeddings 512 \
            --tokenizer-model-type gpt2 \
            --tokenizer-type GPT2BPETokenizer \
            --load-pretrained ${CHECKPOINT_PATH}/Enzyme-GLM-wudao-1125-test1_5000iters/Enzyme-GLM-wudao-1125-test1_5000iters11-25-14-10/5000"
