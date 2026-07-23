import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer

# 1. Setup Model and Tokenizer
model_id = "meta-llama/Meta-Llama-3-8B"  # Replace with your desired model ID

# 2. Configure 4-bit Quantization (The "Q" in QLoRA)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",               # Information-theoretically optimal 4-bit data type
    bnb_4bit_compute_dtype=torch.bfloat16,   # Computation datatype (keeps training stable)
    bnb_4bit_use_double_quant=True,          # Extra memory savings by quantizing constants
)

# Load the quantized base model
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)

# Load and configure tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# 3. Freeze Base Weights & Prepare Model
base_model = prepare_model_for_kbit_training(base_model)

# 4. Configure LoRA (The "LoRA" in QLoRA)
lora_config = LoraConfig(
    r=16,                                    # Rank size of the low-rank update matrices
    lora_alpha=32,                           # Scaling factor
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], # Attention layers to adapt
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Wrap the model with PEFT adapters
model = get_peft_model(base_model, lora_config)

# 5. Load and Format Dataset
# Example dataset: Stanford Alpaca format or any instruction-following format
dataset = load_dataset("tatsu-lab/alpaca", split="train[:1000]") # Subset for speed

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['instruction'])):
        text = f"### Instruction:\n{example['instruction'][i]}\n\n### Response:\n{example['output'][i]}"
        output_texts.append(text)
    return output_texts

# 6. Define Training Arguments
training_args = TrainingArguments(
    output_dir="./qlora_output",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    max_steps=100,                           # Limit steps for demonstration
    optim="paged_adamw_8bit",                # Crucial for QLoRA to handle memory spikes
    bf16=True,                               # Set to True if your GPU supports bfloat16
    fp16=False,
)

# 7. Initialize SFTTrainer and Train
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    formatting_func=formatting_prompts_func,
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_args,
)

# Start training
trainer.train()

# 8. Save the Adapter
trainer.model.save_pretrained("./final_qlora_adapter")
tokenizer.save_pretrained("./final_qlora_adapter")
