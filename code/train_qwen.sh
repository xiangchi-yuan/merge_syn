#export MODEL_PATH='Qwen/Qwen2.5-0.5B-Instruct'
#export MODEL_PATH='../../../../../../../data/xiangchi/model/self_model/iter_2'
export MODEL_PATH='../../../../../../../research/data/transfer/data/xiangchi/model/merge_model/iter_3'
#export SAVE_PATH='../../../../../../../data/xiangchi/model/self_model/iter_1'
#export SAVE_PATH='../../../../../../../research/data/transfer/data/xiangchi/model/self_model/iter_3'
export SAVE_PATH='Qwen/Qwen2.5-0.5B-Instruct'

export MASTER_ADDR="localhost"
export MASTER_PORT="1231"cd
export GLOO_SOCKET_IFNAME="lo"
export NCCL_SOCKET_IFNAME="lo"
export WANDB_DISABLED=true
wandb offline

#--data_path "../../../../../../../data/xiangchi/data/self/GSM8K/gsm8k_train_SCComplexCoT_answer_qwen_right__1.json" \

#--data_path "GSM8K/gsm8k_train_SCComplexCoT_answer_qwen_right_.json" \
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 train_math.py \
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.launch --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} --nproc_per_node=8 --use_env train_math.py \
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 train_math.py \
#    --model_name_or_path $MODEL_PATH \
#    --data_path "../../../../../../../research/data/transfer/data/xiangchi/data/merge/GSM8K/gsm8k_train_SCComplexCoT_answer_qwen_right__3.json" \
#    --data_length 10000000 \
#    --bf16 True \
#    --output_dir $SAVE_PATH \
#    --num_train_epochs 3 \
#    --per_device_train_batch_size 2 \
#    --per_device_eval_batch_size 4 \
#    --gradient_accumulation_steps 4 \
#    --evaluation_strategy "no" \
#    --save_strategy "steps" \
#    --save_steps 1000 \
#    --save_total_limit 2 \
#    --learning_rate 2e-5 \
#    --weight_decay 0. \
#    --warmup_ratio 0.03 \
#    --lr_scheduler_type "cosine" \
#    --logging_steps 1 \
#    --tf32 True

#    --fsdp "full_shard auto_wrap" \
#    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \


python eval_gsm8k.py --model $SAVE_PATH --data_file ./data/test/GSM8K_test.jsonl
#python test_generate.py --model $MODEL_PATH --data_file ./data/test/GSM8K_test.jsonl
