import os
from llama_cpp import Llama, llama_log_set
import ctypes

import sys
import os
from llama_cpp import Llama
from contextlib import redirect_stdout, redirect_stderr
import io

# Load your quantized GGUF model
llm = Llama(
    model_path="llama32-1b-merged-Q4_K_M.gguf",
    n_threads=8,        # adjust based on your CPU cores
    n_ctx=2048,         # context length (tokens)
    verbose=False       # set to True for detailed logging
)

# Prompt
prompt = """Explain the below predictions:

{
  "predicted_class": "early",
  "predicted_probabilities": {
    "early": 0.5011,
    "on_time": 0.0414,
    "delay": 0.4775
  },
  "top_positive_features": [
    {
      "order_item_discount": 0.1268
    },
    {
      "order_profit_per_order": 0.0878
    }
  ],
  "top_negative_features": [
    {
      "performance_score_shipping_hour": -0.0474
    },
    {
      "performance_score_customer_full_location": -0.1137
    }
  ]
}
"""

# Create dummy streams to suppress output
f = io.StringIO()
with redirect_stdout(f), redirect_stderr(f):
    output = llm(prompt, max_tokens=150, temperature=1.2, top_p=0.9)

print(output["choices"][0]["text"])