GPU_NUM=2
WORLD_SIZE=1
RANK=0
MASTER_ADDR=localhost
MASTER_PORT=12588

DISTRIBUTED_ARGS="
    --nproc_per_node $GPU_NUM \
    --nnodes $WORLD_SIZE \
    --node_rank $RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

train_datasets=(
    "/home/jqsj/hqs/data/dataset/train_ForenSynths/train"
)
eval_datasets=(
    "/home/jqsj/hqs/data/dataset/train_ForenSynths/val"
)

MODEL="SAFE"

for train_dataset in "${train_datasets[@]}" 
do
    for eval_dataset in "${eval_datasets[@]}" 
    do

        current_time=$(date +"%Y%m%d_%H%M%S")
        RUN_NAME="SAFE_add_EMA_use_texture${current_time}"
        OUTPUT_PATH="results/$MODEL/$RUN_NAME"
        mkdir -p $OUTPUT_PATH

        torchrun $DISTRIBUTED_ARGS main_finetune.py \
            --input_size 224 \
            --transform_mode 'texture' \
            --model $MODEL \
            --data_path "$train_dataset" \
            --eval_data_path "$eval_dataset" \
            --save_ckpt_freq 5 \
            --batch_size 32 \
            --blr 1e-3 \
            --weight_decay 0.01 \
            --warmup_epochs 2 \
            --epochs 30 \
            --num_workers 16 \
            --output_dir $OUTPUT_PATH \
            --use_swanlab \
            --project_name "RESNET_EMA" \
            --run_name "$RUN_NAME" \
        2>&1 | tee -a $OUTPUT_PATH/log_train.txt

    done
done
