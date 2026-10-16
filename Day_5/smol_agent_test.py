from smolagents import CodeAgent, InferenceClientModel, TransformersModel

# Initialize a model (using Hugging Face Inference API)
#model = InferenceClientModel()  # Uses a default model
model = TransformersModel(
    model_id="HuggingFaceTB/SmolLM3-3B",
    temperature=0.7,
    max_new_tokens=1000
)
# Create an agent with no tools
agent = CodeAgent(tools=[], model=model)

# Run the agent with a task
result = agent.run("Calculate the sum of numbers from 1 to 10")
print(result)
