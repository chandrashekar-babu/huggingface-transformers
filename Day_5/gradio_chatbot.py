import gradio as gr
from transformers import pipeline

# 2. Chat-Style Demo
generator = pipeline(
    "text-generation",
    model="google/gemma-3-1b-it",
    tokenizer="google/gemma-3-1b-it",
)

def chat_response(message, history):
    """
    Generate response for chat interface
    """
    # Format message with instruction template
    prompt = f"### Instruction:\nRespond to the user's message helpfully.\n\n### Input:\n{message}\n\n### Response:\n"
    
    response = generator(
        prompt,
        max_new_tokens=320,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )[0]['generated_text']
    
    # Extract only the response part
    response = response.split("### Response:\n")[-1].strip()
    
    return response

# Create Gradio ChatInterface
demo = gr.ChatInterface(
    fn=chat_response,
    title="Instruction-Tuned Chat Assistant",
    description="Ask me anything! I can help with questions, tasks, and more.",
    examples=[
        "What is machine learning?",
        "Write a Python function to sort a list.",
        "Summarize: AI is transforming industries...",
    ]
)

demo.launch()
