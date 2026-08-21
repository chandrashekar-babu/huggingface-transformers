import numpy as np
import evaluate
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoModelForImageClassification,
    AutoImageProcessor,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

DATA_DIR = "../../data/Chessman-image-dataset/Chess"
CHECKPOINT = "facebook/dinov2-base"
OUTPUT_DIR = "../../data/models/dinov2-full-finetuned-chessman"

# Load the dataset
dataset = load_dataset(DATA_DIR)

train_test = dataset["train"].train_test_split(test_size=0.2, seed=42)
test_val = train_test["test"].train_test_split(test_size=0.5, seed=42)

dataset = DatasetDict({
    "train": train_test["train"],
    "validation": test_val["train"],
    "test": test_val["test"],
})
print(dataset)

num_classes = dataset["train"].features["label"].num_classes

# Prepare the dataset
image_processor = AutoImageProcessor.from_pretrained(CHECKPOINT, use_fast=True)

def preprocess_function(examples):
    images = [image.convert("RGB") for image in examples["image"]]
    inputs = image_processor(images=images, return_tensors="pt")
    inputs["label"] = examples["label"]
    return inputs

prepared_dataset = dataset.map(preprocess_function, batched=True)

# Load and initialize the model
model = AutoModelForImageClassification.from_pretrained(CHECKPOINT, num_labels=num_classes)
print(f"Trainable parameters: {model.num_parameters(only_trainable=True):,}")

# Setup evaluation metrics
metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Setup the Training parameters
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=30,
    learning_rate=3e-5,
    warmup_steps=30,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    logging_strategy="epoch",
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    push_to_hub=False,
    dataloader_pin_memory=False,
    report_to="none",
)

early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=6,
    early_stopping_threshold=0.0,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=prepared_dataset["train"],
    eval_dataset=prepared_dataset["validation"],
    compute_metrics=compute_metrics,
    callbacks=[early_stopping_callback],
)

result = trainer.train()
print("TRAIN RESULT:", result)

print("LOG HISTORY:")
for entry in trainer.state.log_history:
    print(entry)

test_metrics = trainer.predict(prepared_dataset["test"]).metrics
print("TEST METRICS:", test_metrics)
