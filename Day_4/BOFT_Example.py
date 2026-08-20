import torch
from transformers import AutoModelForImageClassification
from peft import BOFTConfig, get_peft_model

# 1. Load your base foundation model
# BOFT works remarkably well across LLMs, Vision Transformers (ViTs), and Diffusion Models.
model_id = "facebook/dinov2-base"
model = AutoModelForImageClassification.from_pretrained(model_id)

# 2. Instantiate the BOFT Configuration
# Specify the target modules, block sizes, and butterfly factors
boft_config = BOFTConfig(
    boft_block_size=4,               # Block size of the orthogonal matrix blocks (must divide module dimension)
    boft_n_butterfly_factor=2,       # Number of butterfly factor layers (higher = more expressive but more parameters)
    target_modules=[                 # Target projections/layers to inject BOFT parameters
        "query", 
        "value", 
        "key", 
        "output.dense"
    ],
    boft_dropout=0.1,                # Dropout rate applied to BOFT transformations
    bias="boft_only",                # Train only BOFT-related biases ("none", "all", or "boft_only")
)

# 3. Wrap the base model with PEFT's BOFT implementation
boft_model = get_peft_model(model, boft_config)

# 4. Print trainable parameters to verify efficiency
# BOFT often saves massive overhead compared to full-rank matrices.
boft_model.print_trainable_parameters()

# 5. Use standard training pipelines or manual loops
# Only the injected butterfly factors and specified biases will update during backpropagation.
print("BOFT model successfully initialized and ready for training!")
