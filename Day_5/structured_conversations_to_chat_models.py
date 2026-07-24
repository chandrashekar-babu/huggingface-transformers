from transformers import AutoTokenizer

# Different models use different chat templates

# Llama-2 Chat Template
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the weather like?"},
]
formatted = tokenizer.apply_chat_template(messages, tokenize=False)
print(formatted)
# <s>[INST] <<SYS>>
# You are a helpful assistant.
# <</SYS>>
# What is the weather like? [/INST]

# Mistral Chat Template
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
formatted = tokenizer.apply_chat_template(messages, tokenize=False)
# <s>[INST] What is the weather like? [/INST]

# Phi-3 Chat Template
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
messages = [
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi! How can I help?"},
    {"role": "user", "content": "What is Python?"},
]
formatted = tokenizer.apply_chat_template(messages, tokenize=False)
# <|user|>
# Hello!<|end|>
# <|assistant|>
# Hi! How can I help?<|end|>
# <|user|>
# What is Python?<|end|>
# <|assistant|>