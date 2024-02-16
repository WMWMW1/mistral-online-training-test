from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import Dataset
from transformers import TrainingArguments
from transformers import Trainer
torch.manual_seed(0)

path = 'openbmb/MiniCPM-2B-dpo-fp16'
tokenizer = AutoTokenizer.from_pretrained(path)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, device_map='cuda', trust_remote_code=True)

prompt="""write a pytorch code

you must return in json fromat 
you need to respond in the format in json:
{
"reasoning":"reasoning before coding(planning)"
"code":"all code in here"
}

"""


prompt="""
hello

"""
chat1 = [
    {"role": "user", "content": "who are you?"},
    {"role": "assistant", "content": "My name is."}
]
chat2 = [
    {"role": "user", "content": "Which is bigger, a virus or a bacterium?"},
    {"role": "assistant", "content": "A bacterium."}
]

dataset = Dataset.from_dict({"chat": [chat1, chat2]})
dataset = dataset.map(lambda x: {"formatted_chat": tokenizer.apply_chat_template(x["chat"], tokenize=False, add_generation_prompt=False)})
print(dataset['formatted_chat'][0])





training_args = TrainingArguments(
    output_dir="./my-model",
    learning_rate=2e-5,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False, # Set to True if you intend to push to the Hugging Face Hub
)

# Prepare dataset for training (mock preprocessing)
# In a real scenario, you'd convert 'formatted_chat' into model input IDs
train_dataset = dataset.map(lambda examples: tokenizer(examples['formatted_chat'], padding="max_length", truncation=True), batched=True)
print("train_dataset",train_dataset)
# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)
trainer.train()
# messages=[]
# modelinput= input("input from user: ")
# responds, history = model.chat(tokenizer,modelinput , temperature=0.8, top_p=0.8)

# messages.append(history)
# print(history)

# while True:
#     modelinput= input("input from user: ")
#     responds, history = model.chat(tokenizer,modelinput , history,temperature=0.8, top_p=0.8)
#     print(responds)
#     print(history)
#     messages.append(history)
# # trainer.save_model('my-model')

