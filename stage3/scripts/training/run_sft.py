import os
import json
import torch
import argparse
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM


def load_dataset_from_jsonl(filepath):
    """
    Loads a JSONL file where each line has a 'text' field containing
    the full pre-formatted ChatML string for Qwen2.5.
    """
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if 'text' in record and record['text']:
                    data.append(record)
            except json.JSONDecodeError:
                continue
    print(f"Loaded {len(data)} samples from {filepath}")
    return Dataset.from_list(data)


def main():
    stage3_root = Path(__file__).resolve().parents[2]
    default_data = stage3_root / "data" / "cc_sft_dataset_qwen_mixed.jsonl"

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--data_path", type=str, default=str(default_data))
    parser.add_argument("--output_dir", type=str, default="checkpoints/lora_cc_qwen")
    parser.add_argument("--cache_dir", type=str, default="$SCRATCH/hf_cache")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--local-rank", type=int, default=-1, help="For distributed training")
    args = parser.parse_args()

    # Expand env vars (for defaults like $SCRATCH/...) and fallback if path is missing.
    resolved_cache_dir = os.path.expandvars(args.cache_dir) if args.cache_dir else None
    cache_dir = resolved_cache_dir if resolved_cache_dir and os.path.exists(resolved_cache_dir) else None

    # ── Tokenizer ──────────────────────────────────────────────────────────────
    print(f"Loading tokenizer: {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        cache_dir=cache_dir,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Dataset ────────────────────────────────────────────────────────────────
    print(f"Loading dataset from: {args.data_path}...")
    dataset = load_dataset_from_jsonl(args.data_path)
    print(f"Total training samples: {len(dataset)}")

    # ── Model ──────────────────────────────────────────────────────────────────
    print(f"Loading model: {args.model_id}...")
    
    # Check if we are running in a multi-GPU environment (Accelerate)
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    device_map = {"": local_rank} if local_rank != -1 else "auto"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        cache_dir=cache_dir,
        trust_remote_code=True
    )
    model.config.use_cache = False  # Required for gradient checkpointing

    # ── LoRA ───────────────────────────────────────────────────────────────────
    # Target all linear projection layers in Qwen2.5 for maximum expressivity
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.enable_input_require_grads() # Fix for "tensors does not require grad" error

    # ── Training Args ──────────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        num_train_epochs=args.epochs,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        gradient_checkpointing=True,
        # Mandatory fix for DDP + Gradient Checkpointing conflict:
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        optim="adamw_torch",
        report_to="none",
        dataloader_num_workers=4,
    )

    # ── Loss Masking ───────────────────────────────────────────────────────────
    # Mask the user prompt — only compute loss on assistant tokens.
    # Qwen ChatML format: assistant responses start with <|im_start|>assistant\n
    response_template = "<|im_start|>assistant\n"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False
    )

    # ── Trainer ────────────────────────────────────────────────────────────────
    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        max_seq_length=args.max_seq_length,
        dataset_text_field="text",
    )

    print("Starting Training...")
    trainer.train()

    # ── Save ───────────────────────────────────────────────────────────────────
    final_dir = os.path.join(args.output_dir, "final")
    print(f"Saving final LoRA adapter to {final_dir}...")
    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print("Done!")


if __name__ == "__main__":
    main()
