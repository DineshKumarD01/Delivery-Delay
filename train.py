import os
import json
import random
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ------------------------------
# 0. Basics & helpers
# ------------------------------
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

ASSISTANT_TAG = "<|assistant|>"
ASSISTANT_TAG_NL = "<|assistant|>\n"
USER_TAG = "<|user|>"

def has_bf16() -> bool:
    # Conservative check
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()

def compute_dtype_4bit():
    # If bf16 is supported, use it for 4-bit matmul compute; else fp16
    return torch.bfloat16 if has_bf16() else torch.float16

# ------------------------------
# 1. Load your JSON dataset
# ------------------------------
with open("synthetic_nl_explanations.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)  # list of dicts: {"input": {...}, "output": [str, ...]}

# Expand: each output becomes one row
expanded: List[Dict[str, Any]] = []
for item in raw_data:
    inp = item["input"]
    for out in item["output"]:
        expanded.append({"input": inp, "output": out})

# Train/test split 90/10
random.shuffle(expanded)
split_idx = int(0.9 * len(expanded))
train_data = expanded[:split_idx]
test_data = expanded[split_idx:]

dataset = DatasetDict(
    {
        "train": Dataset.from_list(train_data),
        "test": Dataset.from_list(test_data),
    }
)

# ------------------------------
# 2. Load tokenizer + model (QLoRA 4-bit)
# ------------------------------
model_id = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

# Ensure pad_token exists (needed for padding)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 4-bit quantization config (QLoRA)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=compute_dtype_4bit(),
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
)

# --- IMPORTANT: enable gradient checkpointing BEFORE k-bit prep & LoRA ---
# This reduces memory but requires special k-bit preparation.
if hasattr(model, "gradient_checkpointing_enable"):
    model.gradient_checkpointing_enable()

# --- Prepare model for k-bit training (VERY IMPORTANT) ---
# This does several internal fixes needed when using bitsandbytes k-bit + checkpointing:
#  - disable use_cache
#  - make sure layernorm & other types are in correct dtype
#  - enable input requires_grad hooks so checkpointing works
model = prepare_model_for_kbit_training(model)

# Also ensure use_cache is False
model.config.use_cache = False

# ------------------------------
# 3. Format function (prompt + response)
# ------------------------------
def format_example(example: Dict[str, Any]) -> Dict[str, str]:
    """
    Turn {"input": dict, "output": str} into a single training string:
    <|user|>
    Explain this prediction:
    {JSON}
    <|assistant|>
    {assistant_answer}
    """
    user_prompt = f"Explain this prediction:\n{json.dumps(example['input'], ensure_ascii=False)}\n"
    assistant_answer = example["output"].strip()

    # Optional: add EOS to help the model stop
    if not assistant_answer.endswith(tokenizer.eos_token):
        assistant_answer = assistant_answer + tokenizer.eos_token

    full_prompt = f"{USER_TAG}\n{user_prompt}{ASSISTANT_TAG}\n{assistant_answer}"
    return {"text": full_prompt}

dataset = dataset.map(format_example, remove_columns=dataset["train"].column_names)

# ------------------------------
# 4. Tokenization with masking (loss only on assistant part)
# ------------------------------
def tokenize_batch(batch):
    """
    - Tokenize the whole sample.
    - Mask everything up to and including the assistant tag, so loss is only on assistant content.
    - We DO NOT pad here; we let the custom collator pad dynamically.
    """
    texts = batch["text"]
    toks = tokenizer(texts, truncation=True, max_length=1024)  # no padding here

    labels = []
    for txt, ids in zip(texts, toks["input_ids"]):
        # Find the first occurrence of the assistant tag (with or without newline)
        idx = txt.find(ASSISTANT_TAG_NL)
        tag_len_chars = len(ASSISTANT_TAG_NL)
        if idx == -1:
            idx = txt.find(ASSISTANT_TAG)
            tag_len_chars = len(ASSISTANT_TAG)

        # Default: no masking if tag not found (shouldn't happen given our format)
        mask_len = 0
        if idx != -1:
            # Prompt part = everything up to & including the assistant tag we found
            prompt_part = txt[: idx + tag_len_chars]
            prompt_ids = tokenizer(prompt_part, add_special_tokens=False)["input_ids"]
            mask_len = len(prompt_ids)

        lab = ids.copy()
        # Mask the prompt part (no loss contribution there)
        for i in range(min(mask_len, len(lab))):
            lab[i] = -100
        labels.append(lab)

    toks["labels"] = labels
    return toks

tokenized_dataset = dataset.map(
    tokenize_batch,
    batched=True,
    remove_columns=dataset["train"].column_names,  # keep only tokenized fields
)

# ------------------------------
# 5. LoRA config (PEFT)
# ------------------------------
peft_config = LoraConfig(
    r=16,                         # leaner rank; 64 works but uses more VRAM
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # common set for LLaMA
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)

# Optional: print trainable params and a short listing for debug
def print_trainable_parameters(m):
    t, tr = 0, 0
    for n, p in m.named_parameters():
        num = p.numel()
        t += num
        if p.requires_grad:
            tr += num
    print(f"trainable params: {tr:,} || all params: {t:,} || trainable%: {100 * tr / t:.4f}")
print_trainable_parameters(model)

# set model to train mode
model.train()

# ------------------------------
# 6. Custom data collator (dynamic padding for Causal LM with masked labels)
# ------------------------------
@dataclass
class DataCollatorForCausalLMWithPadding:
    tokenizer: AutoTokenizer
    pad_to_multiple_of: int | None = 8  # better tensor cores utilization

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Extract lists
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]

        # Compute max length in this batch
        max_len = max(len(ids) for ids in input_ids)
        if self.pad_to_multiple_of is not None:
            # round up to nearest multiple (helps speed on GPUs)
            if max_len % self.pad_to_multiple_of != 0:
                max_len = ((max_len // self.pad_to_multiple_of) + 1) * self.pad_to_multiple_of

        pad_id = self.tokenizer.pad_token_id
        attns = []
        padded_inputs = []
        padded_labels = []

        for ids, labs in zip(input_ids, labels):
            # pad input_ids
            pad_len = max_len - len(ids)
            padded = ids + [pad_id] * pad_len
            padded_inputs.append(padded)

            # attention_mask: 1 for real tokens, 0 for pad
            attn = [1] * len(ids) + [0] * pad_len
            attns.append(attn)

            # pad labels with -100 so loss ignores padding
            padded_lab = labs + [-100] * pad_len
            padded_labels.append(padded_lab)

        batch = {
            "input_ids": torch.tensor(padded_inputs, dtype=torch.long),
            "attention_mask": torch.tensor(attns, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }
        return batch

data_collator = DataCollatorForCausalLMWithPadding(tokenizer=tokenizer)

# ------------------------------
# 7. Training arguments
# ------------------------------
# Use bf16 if available; otherwise fp16 (A5000 -> usually fp16)
use_bf16 = has_bf16()
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    logging_steps=50,
    learning_rate=2e-4,
    per_device_train_batch_size=8,   # try 8 on 24GB; adjust if OOM
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,   # effective batch size = 16
    num_train_epochs=3,
    weight_decay=0.01,
    warmup_steps=50,
    fp16=not use_bf16,
    bf16=use_bf16,
    push_to_hub=True,
    hub_model_id="Dinesh2001/Llama3.2-1B-QLoRA-Explainer",
    report_to="none",
)

# ------------------------------
# 8. Trainer
# ------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,  # ok (FutureWarning in v5; fine for now)
    data_collator=data_collator,
)

# ------------------------------
# 9. Train & push
# ------------------------------
trainer.train()
trainer.push_to_hub()