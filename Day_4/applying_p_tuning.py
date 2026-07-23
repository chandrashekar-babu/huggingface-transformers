from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PrefixTuningConfig, TaskType, get_peft_model
import torch

# 1. Load your base NLU model (e.g., RoBERTa, BERT, or DeBERTa)
model_name = "FacebookAI/roberta-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# P-Tuning v2 uses a standard linear head rather than a verbalizer
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 2. Configure P-Tuning v2 (Deep Prefix Tuning)
peft_config = PrefixTuningConfig(
    task_type=TaskType.SEQ_CLS,          # Use SEQ_CLS for classification or TOKEN_CLS for sequence labeling (NER)
    num_virtual_tokens=20,               # Shorter prompts (10-20) for classification; use ~100 for sequence labeling
    prefix_projection=True,              # True enables the MLP reparameterization encoder mentioned in the paper
)

# 3. Wrap the model
model = get_peft_model(model, peft_config)

# 4. Verify parameter savings (typically 0.1% - 3% trainable parameters)
model.print_trainable_parameters()

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./ptuning_v2_output",
    learning_rate=1e-2,            # P-Tuning v2 accommodates larger learning rates safely
    per_device_train_batch_size=16,
    num_train_epochs=5,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=your_tokenized_dataset,
)

trainer.train()

# Saves only the lightweight layer-wise prefix matrices (~0.1% of original size)
model.save_pretrained("./saved_ptuning_v2_weights")
