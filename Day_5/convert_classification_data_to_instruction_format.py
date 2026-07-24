# Converting classification data to instruction format
def create_instruction_dataset(examples):
    instructions = []
    
    for text, label in zip(examples['text'], examples['label']):
        # Create instruction-formatted example
        instruction = {
            "instruction": "Classify the sentiment of the following review as positive or negative.",
            "input": text,
            "output": "Positive" if label == 1 else "Negative"
        }
        instructions.append(instruction)
    
    return instructions

# Example transformation:
# Before: {"text": "Great product!", "label": 1}
# After: {
#   "instruction": "Classify the sentiment...",
#   "input": "Great product!",
#   "output": "Positive"
# }

# Formatting for model training
def format_instruction(example):
    """Format instruction for model input"""
    if example.get("input"):
        prompt = f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""
    else:
        prompt = f"""### Instruction:
{example['instruction']}

### Response:
{example['output']}"""
    
    return {"text": prompt}
