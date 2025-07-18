GPU_NUM=2
WORLD_SIZE=1
RANK=0
MASTER_ADDR=localhost
MASTER_PORT=12345

DISTRIBUTED_ARGS="
    --nproc_per_node $GPU_NUM \
    --nnodes $WORLD_SIZE \
    --node_rank $RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

MODEL="SAFE"
RESUME_PATH="./results/SAFE/SAFE_add_EMA_use_texture20250717_175750"
CHECKPOINT_FILE="checkpoint-19.pth"

eval_datasets=(
    "/home/jqsj/hqs/data/dataset/UniversalFakeDetect/test" \
    "/home/jqsj/hqs/data/dataset/Self-Synthesis/test" \
    "/home/jqsj/hqs/data/dataset/WildRF/test"
)
for eval_dataset in "${eval_datasets[@]}"
do
    DATASET_NAME=$(basename $(dirname "$eval_dataset"))
    CHECKPOINT_NAME=$(basename "$CHECKPOINT_FILE" .pth)

    torchrun $DISTRIBUTED_ARGS main_finetune.py \
        --input_size 256 \
        --transform_mode 'crop' \
        --model $MODEL \
        --eval_data_path "$eval_dataset" \
        --batch_size 256 \
        --num_workers 16 \
        --output_dir "$RESUME_PATH/eval_results" \
        --resume "$RESUME_PATH/$CHECKPOINT_FILE" \
        --use_swanlab \
        --project_name "SAFE_RESNET_EMA_texture" \
        --run_name "eval_${CHECKPOINT_NAME}_${DATASET_NAME}" \
        --eval True
done
