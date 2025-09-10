from transformers import AutoModelForCausalLM
from peft import PeftModel

base_model_path = "models/llama-3.2-1b"
adapter_path    = "Dinesh2001/Llama3.2-1B-QLoRA-Explainer"

# Load base + adapter
base_model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype="float16")
model = PeftModel.from_pretrained(base_model, adapter_path)

# Merge adapter into base
model = model.merge_and_unload()

# Save merged weights in HF format
model.save_pretrained("merged_model")