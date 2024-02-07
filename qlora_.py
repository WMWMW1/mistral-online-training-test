from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig,get_peft_model
peft_config = LoraConfig(
     inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
)
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
model = get_peft_model(model, peft_config)