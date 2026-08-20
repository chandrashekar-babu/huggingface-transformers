import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PromptTuningConfig, PromptTuningInit, get_peft_model

# 1. Define base model and tokenizer
model_id = "google/gemma-3b-it"  # Using a lightweight, instruct-tuned base model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

# 2. Configure Prompt Tuning parameters
peft_config = PromptTuningConfig(
    task_type="CAUSAL_LM",
    num_virtual_tokens=8,                # Allocate 8 continuous virtual token embeddings
    prompt_tuning_init=PromptTuningInit.TEXT,  # Initialize using text vectors instead of random noise
    prompt_tuning_init_text="Classify the sentiment of this text:", # Initial hint for the weights
    tokenizer_name_or_path=model_id,     # Needed to pull text embeddings for initialization
)

# 3. Create the Prompt Tuned Model
# This freezes all original model layers and injects trainable virtual token variables.
peft_model = get_peft_model(model, peft_config)

# Verify that only a fraction of parameters are trainable
peft_model.print_trainable_parameters()
# Output will show that less than 0.01% of parameters are active for training!

# 4. Preparation for training or inference
# When you pass text through the tokenizer and model, PEFT automatically 
# prepends the 8 virtual token embeddings to the front of your embedded input sequence.
text_input = "Movie Review: The plot was entirely predictable and boring."
inputs = tokenizer(text_input, return_tensors="pt").to(peft_model.device)

# You can now feed 'inputs' directly into a standard training loop (like Hugging Face Trainer)
# Or run generation directly:
with torch.no_grad():
    outputs = peft_model.generate(**inputs, max_new_tokens=10)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
