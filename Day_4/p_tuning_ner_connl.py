from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForTokenClassification

model_name = "FacebookAI/roberta-large"
tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True) # Essential for RoBERTa/BART NER

# Load CoNLL-2003 NER Dataset
dataset = load_dataset("conll2003")
label_list = dataset["train"].features["ner_tags"].feature.names
num_labels = len(label_list)

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100) # Hugging Face cross-entropy ignores -100
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100) # Only label the first sub-token
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Process the dataset splits
tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True, remove_columns=dataset["train"].column_names)
data_collator = DataCollatorForTokenClassification(tokenizer)

from transformers import AutoModelForTokenClassification
from peft import PrefixTuningConfig, TaskType, get_peft_model

# Initialize the model with a standard linear sequence tagging head
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels)

# Configure according to P-Tuning v2 sequence labeling recommendations
peft_config = PrefixTuningConfig(
    task_type=TaskType.TOKEN_CLS,        # Set to Token Classification
    num_virtual_tokens=100,              # Paper suggests longer sequences (~100) for sequence tagging
    prefix_projection=False,             # Raw embeddings perform better than an MLP bottleneck on dense tasks
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters() 
# You will notice that only the sequence tagging head and layer-wise prefixes are trainable!

import evaluate
import numpy as np
from transformers import Trainer, TrainingArguments

metric = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

training_args = TrainingArguments(
    output_dir="./ptuning_v2_ner",
    learning_rate=1e-2, 
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
