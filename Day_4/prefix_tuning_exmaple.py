import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PrefixTuningConfig, get_peft_model

# 1. Load your base model and tokenizer
model_id = "google/gemma-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)

# 2. Configure Prefix Tuning parameters
peft_config = PrefixTuningConfig(
    task_type="CAUSAL_LM",
    num_virtual_tokens=30,           # Number of virtual prefix tokens inserted per attention layer
    encoder_hidden_size=512,         # Hidden size of the temporary MLP re-parameterization network
    prefix_projection=True,          # Uses an MLP projection during training for stability (recommended)
)

# 3. Wrap the base model with PEFT
# This freezes the base model and hooks trainable prefix matrices into all attention layers
prefix_model = get_peft_model(model, peft_config)

# 4. Verify parameter usage
prefix_model.print_trainable_parameters()
# Trainable params will be higher than Prompt Tuning, but still < 1% of the total model size.

# 5. Run standard inference or pass to a Trainer loop
prompt = "Summarize the history of quantum computing in one sentence."
inputs = tokenizer(prompt, return_tensors="pt").to(prefix_model.device)

with torch.no_grad():
    outputs = prefix_model.generate(**inputs, max_new_tokens=50)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
