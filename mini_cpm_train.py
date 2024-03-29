from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
import torch
import numpy as np
from SupervisedDataset import SupervisedDataset
torch.manual_seed(0)
from peft import LoraConfig, TaskType
from peft import get_peft_model


peft_config = LoraConfig(
    r=8, lora_alpha=32, 
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                    "lm_head"])
# Load model and tokenizer
path = 'openbmb/MiniCPM-2B-dpo-fp16'
tokenizer = AutoTokenizer.from_pretrained(path)
tokenizer.pad_token = tokenizer.eos_token  # Set pad token
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, load_in_4bit = True,device_map='cuda', trust_remote_code=True)
model = get_peft_model(model, peft_config)


data_path = 'data.json'


# Initialize your custom dataset
dataset = SupervisedDataset(
    data_path=data_path,
    tokenizer=tokenizer,
    model_max_length=4096,  # Adjust as needed
    user_tokens=[1786, 4194, 95388],  # Your custom tokens for user
    assistant_tokens=[1786, 10850, 95388]  # Your custom tokens for assistant
)
model = AutoModelForCausalLM.from_pretrained(path)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",  # Directory where the model predictions and checkpoints will be written.
    num_train_epochs=3,  # Total number of training epochs
    per_device_train_batch_size=1,  # Batch size per device during training
    gradient_accumulation_steps=4,  
    logging_dir='./logs',  # Directory for storing logs
    logging_steps=10,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,  # Your custom dataset
    tokenizer=tokenizer,
    
)

# Start training
trainer.train()
