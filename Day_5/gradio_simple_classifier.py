import gradio as gr
from transformers import pipeline

# 1. Basic Text Classification Demo
def classify_sentiment(text):
    classifier = pipeline("sentiment-analysis", model="bert-base-uncased")
    result = classifier(text)[0]
    label, score = result['label'], result['score']

    output = {"LABEL_0": "❌", "LABEL_1": "✅"}

    print(f"{output[label]}:{label}: {score:.4f}")
    return {output[label]: score}

demo = gr.Interface(
    fn=classify_sentiment,
    inputs=gr.Textbox(label="Enter text", lines=3, placeholder="Type here..."),
    outputs=gr.Label(label="Sentiment"),
    title="Sentiment Analysis Demo",
    description="BERT (base) for sentiment analysis",
    examples=[
        ["This movie was absolutely fantastic!"],
        ["Terrible waste of time and money."],
        ["It was okay, nothing special."],
    ],
)

# Launch
demo.launch(share=True)  # share=True for public link
