import gradio as gr

def create_multi_tab_demo():
    """Create a multi-functional Gradio demo"""
    
    with gr.Blocks(title="My Fine-Tuned Models", theme="soft") as demo:
        gr.Markdown("# 🤗 Fine-Tuned Models Demo")
        gr.Markdown("Explore different capabilities of fine-tuned models")
        
        with gr.Tab("Sentiment Analysis"):
            gr.Markdown("### Analyze sentiment of any text")
            sentiment_input = gr.Textbox(
                label="Enter text",
                placeholder="Type your review here...",
                lines=3,
            )
            sentiment_btn = gr.Button("Analyze", variant="primary")
            sentiment_output = gr.Label(label="Sentiment Result")
            
            def analyze_sentiment(text):
                # Your sentiment analysis function
                result = classifier(text)[0]
                return {result['label']: result['score']}
            
            sentiment_btn.click(
                fn=analyze_sentiment,
                inputs=sentiment_input,
                outputs=sentiment_output,
            )
        
        with gr.Tab("Text Generation"):
            gr.Markdown("### Generate text with custom prompts")
            gen_prompt = gr.Textbox(
                label="Prompt",
                placeholder="Enter your prompt...",
                lines=2,
            )
            gen_max_length = gr.Slider(
                minimum=20, maximum=200, value=50, step=10,
                label="Max Length"
            )
            gen_temperature = gr.Slider(
                minimum=0.1, maximum=2.0, value=0.7, step=0.1,
                label="Temperature"
            )
            gen_btn = gr.Button("Generate", variant="primary")
            gen_output = gr.Textbox(label="Generated Text", lines=5)
            
            def generate_text(prompt, max_length, temperature):
                # Your text generation function
                result = generator(
                    prompt,
                    max_length=max_length,
                    temperature=temperature,
                )[0]['generated_text']
                return result
            
            gen_btn.click(
                fn=generate_text,
                inputs=[gen_prompt, gen_max_length, gen_temperature],
                outputs=gen_output,
            )
        
        with gr.Tab("Model Info"):
            gr.Markdown("### Model Details")
            gr.Markdown("""
            - **Base Model**: SmolLM2-360M
            - **Fine-tuning Method**: QLoRA (r=16)
            - **Training Data**: Custom instruction dataset
            - **Epochs**: 3
            - **Hardware**: Single GPU
            """)
        
    return demo

# Launch
demo = create_multi_tab_demo()
demo.launch(server_port=7860)
