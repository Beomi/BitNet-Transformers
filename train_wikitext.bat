@echo off
set WANDB_PROJECT=bitllama-wikitext
python run_clm.py ^
--dataset_name="wikitext" ^
--dataset_config_name="wikitext-103-v1" ^
--model_type="llama" ^
--config_name="./bitllama-wikitext" ^
--tokenizer_name="beomi/llama-2-ko-7b" ^
--num_train_epochs=10 ^
--block_size=2048 ^
--per_device_train_batch_size=1 ^
--gradient_accumulation_steps=1 ^
--optim adafactor ^
--learning_rate=8e-4 ^
--torch_dtype bfloat16 ^
--bf16 ^
--output_dir="bitllama-wikitext" ^
--do_train ^
--save_strategy="epoch" ^
--logging_strategy="steps" ^
--logging_first_step ^
--logging_steps=10 ^
--save_total_limit=1 ^
--run_name="bitllama-wikitext" ^
--overwrite_output_dir ^
--report_to="wandb" ^
--low_cpu_mem_usage
