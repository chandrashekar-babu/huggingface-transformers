import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import IA3Config, get_peft_model

# 1. Load the foundation model
model_id = "google/gemma-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)

# 2. Configure Bottleneck Scaling Adapters (IA3)
# IA3 introduces learned vectors that scale inner activations (Keys, Values, and Feed-Forward neurons)
peft_config = IA3Config(
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj", "down_proj"], # Target layers to inject bottleneck changes
    feedforward_modules=["down_proj"],               # Specifically isolate the feed-forward bottleneck
)

# 3. Create the adapter-enabled model
# This completely freezes the baseline layers and injects the bottleneck vectors
adapter_model = get_peft_model(model, peft_config)

# 4. Inspect efficiency
adapter_model.print_trainable_parameters()
# Typically uses less than 0.05% trainable parameters!

# 5. Execute inference or training
prompt = "Explain quantum physics using a cooking metaphor."
inputs = tokenizer(prompt, return_tensors="pt").to(adapter_model.device)

with torch.no_grad():
    outputs = adapter_model.generate(**inputs, max_new_tokens=64)
    print(tokenizer.decode(outputs, skip_special_tokens=True))
