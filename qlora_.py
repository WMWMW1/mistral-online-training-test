from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import LoraConfig, get_peft_model

# PEFT configuration
peft_config = LoraConfig(
    inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
)

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1",load_in_4bit=True)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

# Set tokenizer padding token
tokenizer.pad_token = tokenizer.eos_token

# Adapt model with PEFT
# model = get_peft_model(model, peft_config)

# Move model to GPU
model = model.to("cuda")  # This line moves the model to the GPU

# Tokenize inputs and move them to the same device as the model


pipeline = pipeline("text-generation",tokenizer = tokenizer, model=model, device=0)

