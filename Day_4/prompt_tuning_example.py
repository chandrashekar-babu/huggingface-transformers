import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PromptTuningConfig, PromptTuningInit, TaskType, get_peft_model

# 1. Define Model and Tokenizer Information
model_id = "bigscience/bloom-560m" 
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 2. Setup the Prompt Tuning Configuration
peft_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    prompt_tuning_init=PromptTuningInit.TEXT,
    num_virtual_tokens=8,
    prompt_tuning_init_text="Classify if the sentence sentiment is positive or negative:",
    tokenizer_name_or_path=model_id,
)

# 3. Load the Base Pretrained Model
base_model = AutoModelForCausalLM.from_pretrained(model_id)

# 4. Wrap the Base Model with PEFT
model = get_peft_model(base_model, peft_config)

# 5. Verify Trainable Parameters
# This outputs the tiny fraction (~0.001%) of parameters actually being trained
model.print_trainable_parameters()

# 6. Save the Light Weight Prompt Tuning Adapter After Training
model.save_pretrained("../../data/bloom560m_peft_prompt_tuning_adapter")
