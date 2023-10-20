export CUDA_VISIBLE_DEVICES=1
export WANDB_PROJECT=bitllama-korean-stack-hq

python run_clm.py \
--train_file='../falcon_korean_stack_hq_w_trans_10k_sample.json' \
--model_type='llama' \
--config_name='./test-config' \
--tokenizer_name='beomi/llama-2-ko-7b' \
--num_train_epochs=10 \
--block_size=2048 \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--optim adafactor \
--learning_rate=8e-4 \
--torch_dtype bfloat16 \
--bf16 \
--output_dir='bitllama-train-clm-falcon-korean-stack-hq' \
--do_train \
--save_strategy='epoch' \
--logging_strategy='steps' \
--logging_first_step \
--logging_steps=10 \
--save_total_limit=1 \
--run_name='bitllama-train-clm-falcon-korean-stack-hq' \
--overwrite_output_dir \
--report_to='wandb' \
--low_cpu_mem_usage
