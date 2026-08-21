import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def generate_questions(text: str) -> list:
    # Load the fine-tuned end-to-end question generation model and tokenizer
    model_name = "valhalla/t5-base-e2e-qg"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Prefix required by the specific model to trigger question generation
    input_text = f"generate questions: {text}"
    
    # Tokenize input text
    inputs = tokenizer(
        input_text, 
        padding="longest", 
        truncation=True, 
        max_length=512, 
        return_tensors="pt"
    )
    
    # Generate question tokens
    with torch.no_grad():
        output_ids = model.generate(
            inputs["input_ids"],
            max_length=256,
            num_beams=4,
            length_penalty=1.0,
            early_stopping=True
        )
    
    # Decode the output text
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # The model separates generated questions using the <sep> token
    questions = [q.strip() for q in output_text.split("<sep>") if q.strip()]
    
    return questions

# Example Document Context
context_document = (
    "Photosynthesis is a process used by plants and other organisms to convert light energy "
    "into chemical energy. This chemical energy is stored in carbohydrate molecules, such as sugars. "
    "The process takes place primarily inside chloroplasts, using water, carbon dioxide, and light."
)

# Run the generator
generated_questions = generate_questions(context_document)

# Display results
print(f"Document Context:\n{context_document}\n")
print("Generated Questions:")
for i, question in enumerate(generated_questions, 1):
    print(f"{i}. {question}")
