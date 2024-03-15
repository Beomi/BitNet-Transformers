import torch
import time
#from bitnet_llama import LlamaForCausalLM,AutoTokenizer
from transformers import AutoTokenizer, LlamaForCausalLM


def run_model():
    model_peth = "./bitllama-wikitext"
    model = LlamaForCausalLM.from_pretrained(model_peth)
    tokenizer = AutoTokenizer.from_pretrained(model_peth)

    prompt = "Hey, are you conscious? Can you talk to me?"
    inputs = tokenizer(prompt, return_tensors="pt")
    #with open('littleprince.txt', 'r', encoding='utf-8') as file:
    #    prompt = file.read()
    #    inputs = tokenizer(prompt, return_tensors="pt")
        
    start_time = time.time()
    # Generate
    generate_ids = model.generate(inputs.input_ids, max_length=30)
    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds")

    result =  tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(result)
run_model()