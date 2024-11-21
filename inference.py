import os
import transformers
import torch
import time
from tqdm import tqdm
import json
import numpy as np
import math
import wandb

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import BitsAndBytesConfig
from torch.nn import CrossEntropyLoss
from torch.cuda.amp import autocast


os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def loss_fn(logits, labels, vocab_size):
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss()
    shift_logits = shift_logits.view(-1,vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)
    return loss


def load_model(model_name, quantization_mode, use_flash_attn, torch_dtype):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if quantization_mode == "fp8":
        use_fp8 = True
        quantization_config = False
    elif quantization_mode == "fp4":
        use_fp8 = False
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True)
    else:
        use_fp8 = False
        quantization_config = None

    attn_implementation = "flash_attention_2" if use_flash_attn else None

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=use_fp8,
        torch_dtype=torch_dtype,
        device_map={"": int(os.environ.get("LOCAL_RANK") or 0)},
        trust_remote_code=True,
        attn_implementation=attn_implementation
    )
   
    return tokenizer, model



def model_forward(model, dataloader, limit, torch_dtype, vocab_size):
    
    list_inference_times = []
    for i, j in enumerate(tqdm(range(limit))):
        batch = next(iter(dataloader))
        start = time.time()
        with torch.no_grad():
            with autocast(dtype=torch_dtype):
                outputs = model(**batch)
        end = time.time()
        inference_time = end - start
        logits = outputs["logits"]
        loss = loss_fn(logits, batch["input_ids"], vocab_size)

        wandb.log(
            {
                "batch": i+1, 
                "loss": loss, 
                "inference_time": inference_time,
            })
        list_inference_times.append(inference_time)

    return list_inference_times




    

def run_inference(
    model_name,
    batch_size=64,
    limit=10,
    seq_len=1024,
    quantization_mode=None,
    use_flash_attn=False,
    use_compile=False,
    dtype_str="float16",
    debug=True,
):
    
    ##set up the model
    torch_dtype = dtype = getattr(torch, dtype_str)
    tokenizer, model = load_model(model_name, quantization_mode, use_flash_attn, torch_dtype)

    ##load model   
    dataset = load_dataset("allenai/c4", data_files="en/c4-train.00001-of-01024.json.gz",
                           trust_remote_code=True)["train"]
    if debug: 
        dataset = dataset.select(range(2000))
        limit = 3


    ##data processing
    def encode(examples):
        return tokenizer(examples['text'], return_tensors="pt", truncation=True, padding='max_length', max_length=seq_len)
    dataset = dataset.map(encode, batched=True, remove_columns=["text", "timestamp", "url"])
    dataset.set_format("torch", device="cuda") 
    dataloader = DataLoader(dataset, batch_size=batch_size)

    inference_fn = torch.compile(model_forward) if use_compile else model_forward

    list_inference_times  = inference_fn(model, dataloader, limit, torch_dtype, model.config.vocab_size)

    return  list_inference_times 



    
   
         


if __name__ == "__main__":
    import argparse

    args = argparse.ArgumentParser()
    args.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B")
    args.add_argument("--quantization_mode", choices=["fp4","fp8","null"], default="null")
    args.add_argument("--use_flash_attn", action="store_true", default=True)
    args.add_argument("--torch_compile", action="store_true", default=False)
    args.add_argument("--batch", type=int, default=8)
    args.add_argument("--seq_len", type=int, default=2048)
    args.add_argument("--limit", type=int, default=100)
    args.add_argument("--torch_dtype", type=str, default="float16")
    args.add_argument("--debug", action="store_true", default=False)



    args = args.parse_args()

    config_wandb = {}
    config_wandb['model'] = args.model.split("/")[-1]
    config_wandb['bs'] = args.batch
    config_wandb['limit'] = args.limit
    config_wandb['seq_len'] = args.seq_len
    config_wandb['bs'] = args.batch
    config_wandb['quantization_mode'] = args.quantization_mode
    config_wandb['flash_attn'] = str(args.use_flash_attn)
    config_wandb['compile'] = str(args.torch_compile)
    config_wandb['dtype'] = args.torch_dtype
    wandb_name=""
    for key in config_wandb.keys():
        wandb_name += "{}_{}_".format(key,config_wandb[key])

    
    wandb.init(project="inference_llama", name=wandb_name, config=config_wandb)


    list_inference_times = run_inference(
        args.model,
        batch_size=args.batch,
        limit=args.limit,
        seq_len=args.seq_len,
        quantization_mode=args.quantization_mode,
        use_flash_attn=args.use_flash_attn,
        use_compile=args.torch_compile,
        dtype_str=args.torch_dtype,
        debug=args.debug,
    )

    np_array_time = np.array(list_inference_times)
    mean_time = np.mean(np_array_time)
    std_time = np.std(np_array_time)
    total_time = np.sum(np_array_time)

    print(f"Total time {total_time} seconds; time per batch: {mean_time} +- {std_time} seconds.\n")
    print(f"List of times {np_array_time}")
    
