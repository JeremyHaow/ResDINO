
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

MODEL="DINOv2"
RESUME_PATH="./results/DINOv2/resumed_from_ckpt49_20250625_111632"

eval_datasets=(
    "/home/jqsj/hqs/data/dataset/UniversalFakeDetect/test" \
    "/home/jqsj/hqs/data/dataset/Self-Synthesis/test" \
    "/home/jqsj/hqs/data/dataset/WildRF/test"
)
for eval_dataset in "${eval_datasets[@]}"
do
    torchrun $DISTRIBUTED_ARGS main_finetune.py \
        --input_size 224 \
        --transform_mode 'crop' \
        --model $MODEL \
        --eval_data_path $eval_dataset \
        --batch_size 256 \
        --num_workers 16 \
        --output_dir $RESUME_PATH/eval_results \
        --resume $RESUME_PATH/checkpoint-best.pth \
        --eval True
done
