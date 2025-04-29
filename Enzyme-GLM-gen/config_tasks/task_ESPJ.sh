EXPERIMENT_NAME=${MODEL_TYPE}-esp-judgement
TASK_NAME=esp-judgement
DATA_PATH="${DATA_ROOT}/esp-judgement"
MAX_SEQ_LEN=512

LR_SINGLE=1e-5
#EPOCH_SINGLE=50
#XXLARGE_EPOCH=50
EPOCH_SINGLE=1
XXLARGE_EPOCH=1

TRAIN_ARGS="--lr-decay-style linear \
            --warmup 0.1 \
            --weight-decay 1.0e-1 \
            --pattern-id 0 \
	    --train-iters 1000
	    --task-mask"

COMMON_ARGS="--save-interval 1000 \
             --log-interval 50 \
             --eval-interval 1000 \
             --eval-iters 100"

PATTERN_IDS=(0 1 2 3)
PROMPT_IDS=(1 2 3)

BATCH_SIZE=16
