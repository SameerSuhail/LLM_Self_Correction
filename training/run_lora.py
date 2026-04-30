import json
import argparse
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)

                                                                                

MODEL_NAME  = "/scratch/user/sameersuhail/hf_cache/hub/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/c170c708c41dac9275d15a8fff4eca08d52bab71"
MAX_SEQ_LEN = 2048

                                                                                

class SelfCorrectionDataset(Dataset):
    def __init__(self, records: list, tokenizer):
        self.samples = [tokenize_and_mask(r, tokenizer) for r in records]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import numpy as np
        s = self.samples[idx]
        return {
            "input_ids":      torch.tensor(np.array(s["input_ids"]),      dtype=torch.long),
            "attention_mask": torch.tensor(np.array(s["attention_mask"]), dtype=torch.long),
            "labels":         torch.tensor(np.array(s["labels"]),         dtype=torch.long),
        }

                                                                                

def tokenize_and_mask(record: dict, tokenizer) -> dict:
    messages   = record["messages"]
    wrong_step = record["wrong_step_text"]

    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    tokenised = tokenizer(
        full_text,
        truncation=True,
        max_length=MAX_SEQ_LEN,
        return_offsets_mapping=True,
    )
    input_ids      = tokenised["input_ids"]
    offset_mapping = tokenised["offset_mapping"]
    labels         = list(input_ids)

                    
    user_text = tokenizer.apply_chat_template(
        [messages[0]],
        tokenize=False,
        add_generation_prompt=True,
    )
    user_char_end = len(user_text)
    for i, (tok_start, tok_end) in enumerate(offset_mapping):
        if tok_start < user_char_end:
            labels[i] = -100

                     
    wrong_char_start = full_text.find(wrong_step)
    if wrong_char_start != -1:
        wrong_char_end = wrong_char_start + len(wrong_step)
        for i, (tok_start, tok_end) in enumerate(offset_mapping):
            if tok_start < wrong_char_end and tok_end > wrong_char_start:
                labels[i] = -100

    return {
        "input_ids":      input_ids,
        "attention_mask": tokenised["attention_mask"],
        "labels":         labels,
    }

                                                                                

def load_jsonl(path: str) -> list:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

                                                                                

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",   default="sft_dataset1_train.jsonl")
    parser.add_argument("--val",     default="sft_dataset1_val.jsonl")
    parser.add_argument("--output",  default="checkpoints/lora_dataset1")
    parser.add_argument("--epochs",  type=int,   default=10)
    parser.add_argument("--lr",      type=float, default=2e-4)
    parser.add_argument("--lora_r",  type=int,   default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--max",     type=int,   default=None)
    args = parser.parse_args()

    import time
    t0 = time.time()
    def log(msg):
        print(f"[{time.time()-t0:.1f}s] {msg}", flush=True)

    Path(args.output).mkdir(parents=True, exist_ok=True)

                     
    log(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    log("Tokenizer loaded.")

                        
    log("Loading JSONL records...")
    train_raw = load_jsonl(args.train)
    val_raw   = load_jsonl(args.val)
    if args.max:
        train_raw = train_raw[:args.max]
        val_raw   = val_raw[:max(1, args.max // 5)]
    log(f"Train: {len(train_raw)} | Val: {len(val_raw)}")

                    
    log("Tokenising train set...")
    train_dataset = SelfCorrectionDataset(train_raw, tokenizer)
    log("Tokenising val set...")
    val_dataset   = SelfCorrectionDataset(val_raw,   tokenizer)
    log("Tokenisation done.")

                 
    log(f"Loading model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.use_cache = False

                
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    log(f"LoRA applied: r={args.lora_r}, alpha={args.lora_alpha}")

                         
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,                         
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        optim="adamw_torch",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        dataloader_num_workers=4,
        gradient_checkpointing=True,
    )

                   
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            padding=True,
            pad_to_multiple_of=8,
            label_pad_token_id=-100,
        ),
    )

    log("Starting training...")
    trainer.train()
    log("Training complete.")

    log(f"Saving LoRA adapter to {args.output}")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    log("Done.")

if __name__ == "__main__":
    main()
