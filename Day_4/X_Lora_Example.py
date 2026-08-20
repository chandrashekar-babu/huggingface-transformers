import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import XLoraConfig, get_peft_model

# 1. Define your base model and tokenizer
model_id = "meta-llama/Meta-Llama-3-8B-Instruct" 
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load the base model in half-precision or bfloat16
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)

# 2. Map expert names to their respective Hugging Face Hub IDs or local directories
# These are your individual, pre-trained frozen LoRA adapters.
adapters_mapping = {
    "coding_expert": "your-username/llama3-8b-coding-lora",
    "creative_expert": "your-username/llama3-8b-creative-lora",
    "math_expert": "your-username/llama3-8b-math-lora",
}

# 3. Instantiate the X-LoRA Configuration
# Note: Ensure hidden_size matches the hidden_size of your chosen base model architecture
xlora_config = XLoraConfig(
    hidden_size=4096,               # Llama 3 8B standard hidden size
    adapters=adapters_mapping,      # Dictionary of our LoRA experts
    enable_softmax=True,            # Softmax dense routing over experts
    layerwise_scalings=True,        # Compute dynamic gating scalings independently per layer
    top_k_lora=None,                # Set an integer (e.g. 2) to use Sparse Top-K gating instead of dense
    device=torch.accelerator.current_accelerator().name
)

# 4. Initialize the Mixture of LoRA Experts model
# In native Hugging Face PEFT integration, pass the config directly to get_peft_model
# For the standalone 'xlora' package version, use: xlora.add_xlora_to_model(model, xlora_config, ...)
xlora_model = get_peft_model(model, xlora_config)

# 5. Run inference with dynamic, token-level routing
prompt = """Write a Python script that calculates prime numbers, 
but explain it like a poet."""

inputs = tokenizer(prompt, return_tensors="pt").to(xlora_model.device)

with torch.no_grad():
    output_tokens = xlora_model.generate(
        **inputs, 
        max_new_tokens=128, 
        temperature=0.7, 
        do_sample=True
    )

# 6. Decode and output the resulting text
generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
print(generated_text)
