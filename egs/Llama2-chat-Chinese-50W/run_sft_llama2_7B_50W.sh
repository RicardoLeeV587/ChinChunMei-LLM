#lr=1e-4
lr=2e-4
lora_rank=8
lora_alpha=32
lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
modules_to_save="embed_tokens,lm_head"
lora_dropout=0.05

pretrained_model=path/to/Llama2-chat-7B/HF/dir
chinese_tokenizer_path=path/to/tokenizer/model/dir
dataset_dir=path/to/training/dataset/dir # must contain json format! not jsonline
per_device_train_batch_size=16
per_device_eval_batch_size=16
gradient_accumulation_steps=1
training_steps=11718 # if your total train sample is 50W and your global batch size is 128, you want to train 3 epoch, this should be int(500000 / 128 * 3) = 11718. You can also use --num_train_epochs 3 instead.
output_dir=path/to/output/model/dir
validation_file=path/to/validation/dataset/file # this is json file! 

#deepspeed_config_file=ds_zero2_no_offload.json
deepspeed_config_file="config/ds_zero2_offload.json" # Please note that currently (2023.08.01) Zero3 does not support LoRA!
#deepspeed_config_file=ds_zero3_offload.json

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes 1 --nproc_per_node 8 --master_port=25001 ./script/training/run_clm_sft_with_peft.py \
    --deepspeed ${deepspeed_config_file} \
    --model_name_or_path ${pretrained_model} \
    --tokenizer_name_or_path ${chinese_tokenizer_path} \
    --dataset_dir ${dataset_dir} \
    --validation_split_percentage 0.001 \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --do_train \
    --do_eval \
    --seed $RANDOM \
    --fp16 \
    --max_steps ${training_steps} \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_total_limit 3 \
    --evaluation_strategy steps \
    --eval_steps 1000 \
    --save_steps 1000 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --preprocessing_num_workers 8 \
    --max_seq_length 1024 \
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --lora_rank ${lora_rank} \
    --lora_alpha ${lora_alpha} \
    --trainable ${lora_trainable} \
    --modules_to_save ${modules_to_save} \
    --lora_dropout ${lora_dropout} \
    --torch_dtype float16 \
    --validation_file ${validation_file} \
    --gradient_checkpointing \
    --ddp_find_unused_parameters False

    #--num_train_epochs 1 \
