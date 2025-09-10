import json
import random
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model

# ------------------------------
# 1. Load your JSON dataset
# ------------------------------
with open("synthetic_nl_explanations.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)  # list of dicts

# Expand dataset: each output becomes a separate row
expanded = []
for item in raw_data:
    inp = item["input"]
    for out in item["output"]:
        expanded.append({"input": inp, "output": out})

# Train/test split (90/10)
random.shuffle(expanded)
split_idx = int(0.9 * len(expanded))
train_data = expanded[:split_idx]
test_data = expanded[split_idx:]

dataset = DatasetDict({
    "train": Dataset.from_list(train_data),
    "test": Dataset.from_list(test_data)
})

# ------------------------------
# 2. Load tokenizer + model (QLoRA)
# ------------------------------
model_id = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # ensure padding

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="bfloat16",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)

# ------------------------------
# 3. Format function (prompt + response)
# ------------------------------
def format_example(example):
    # stringify input dict in compact form
    user_prompt = f"Explain this prediction:\n{json.dumps(example['input'])}\n"
    assistant_answer = example["output"]

    # Add chat-like formatting
    full_prompt = f"<|user|>\n{user_prompt}<|assistant|>\n{assistant_answer}"
    return {"text": full_prompt}

dataset = dataset.map(format_example)

# ------------------------------
# 4. Tokenization with masking (loss only on assistant part)
# ------------------------------
def tokenize(batch):
    texts = batch["text"]
    toks = tokenizer(texts, truncation=True, max_length=512)
    
    labels = []
    for txt, ids in zip(texts, toks["input_ids"]):
        # find assistant start
        split_token = "<|assistant|>\n"
        split_ids = tokenizer(split_token, add_special_tokens=False)["input_ids"]
        # tokenize prompt separately
        prompt_part = txt.split(split_token)[0] + split_token
        prompt_ids = tokenizer(prompt_part, add_special_tokens=False)["input_ids"]

        lab = ids.copy()
        # mask prompt tokens
        lab[:len(prompt_ids)] = [-100] * len(prompt_ids)
        labels.append(lab)

    toks["labels"] = labels
    return toks

tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=dataset["train"].column_names)

# ------------------------------
# 5. LoRA config (PEFT)
# ------------------------------
peft_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # safe defaults for Llama
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)

# ------------------------------
# 6. Data collator (dynamic padding)
# ------------------------------
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

# ------------------------------
# 7. Training arguments
# ------------------------------
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    logging_steps=50,
    learning_rate=2e-4,
    per_device_train_batch_size=8,   # A5000: try 8; increase if VRAM allows
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,   # effective batch = 16
    num_train_epochs=3,
    weight_decay=0.01,
    warmup_steps=50,
    fp16=False,  # using bfloat16 already in QLoRA
    bf16=True,
    push_to_hub=True,
    hub_model_id="Dinesh2001/Llama3.2-1B-QLoRA-Explainer",
    report_to="none"
)

# ------------------------------
# 8. Trainer
# ------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator
)

# ------------------------------
# 9. Train
# ------------------------------
trainer.train()
trainer.push_to_hub()