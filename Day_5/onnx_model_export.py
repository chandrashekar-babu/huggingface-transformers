from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load fine-tuned model
model = AutoModelForSequenceClassification.from_pretrained("./my-model")
tokenizer = AutoTokenizer.from_pretrained("./my-model")

# Export to ONNX
dummy_input = tokenizer(
    "Sample text for tracing",
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=128,
)

torch.onnx.export(
    model,
    (dummy_input['input_ids'], dummy_input['attention_mask']),
    "model.onnx",
    input_names=['input_ids', 'attention_mask'],
    output_names=['logits'],
    dynamic_axes={
        'input_ids': {0: 'batch_size', 1: 'sequence_length'},
        'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
        'logits': {0: 'batch_size'},
    },
    opset_version=14,
)

# Verify ONNX model
import onnx
onnx_model = onnx.load("model.onnx")
onnx.checker.check_model(onnx_model)
print("✅ ONNX model is valid!")

# Inference with ONNX Runtime
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("model.onnx")
inputs = tokenizer("Great product!", return_tensors="np")
outputs = session.run(
    None,
    {
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask'],
    }
)
print(f"ONNX output shape: {outputs[0].shape}")
