import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer


#
# model_name = "Open-Orca/LlongOrca-7B-16k"
# model_name = "internlm/internlm-20b"
model_name = "HuggingFaceH4/zephyr-7b-alpha"
model_name ='mistralai/Mistral-7B-v0.1'
#model_name = "NousResearch/llama-2-7b-chat-hf" # use this if you have access to the official LLaMA 2 model "meta-llama/Llama-2-7b-chat-hf", though keep in mind you'll need to pass a Hugging Face key argument
dataset_name = "./train.jsonl"
new_model = "test_mistral"
lora_r = 64
lora_alpha = 16
lora_dropout = 0.1
use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False
output_dir = "./results"
num_train_epochs = 1
fp16 = False
bf16 = False
per_device_train_batch_size = 4
per_device_eval_batch_size = 4
gradient_accumulation_steps = 1
gradient_checkpointing = True
max_grad_norm = 0.3
learning_rate = 2e-4
weight_decay = 0.001
optim = "paged_adamw_32bit"
lr_scheduler_type = "constant"
max_steps = -1
warmup_ratio = 0.03
group_by_length = True
save_steps = 25
logging_steps = 5
max_seq_length = None
packing = False
device_map = {"": 0}



tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


# Load datasets
train_dataset = load_dataset('json', data_files='./train.jsonl', split="train")
valid_dataset = load_dataset('json', data_files='./test.jsonl', split="train")

# 新的格式模板
system_message="you are an ai scientist"
system_message_prefix = "<|system|>"
user_input_prefix = "<|user|>\n"
llm_output_prefix = "<|assistant|>\n"
end="</s>"
train_dataset_mapped = train_dataset.map(lambda examples: {'text': [f'{system_message_prefix}\n{system_message}{end}\n{user_input_prefix}' + prompt + f'{end}\n{llm_output_prefix}' + response for prompt, response in zip(examples['prompt'], examples['response'])]}, batched=True)
valid_dataset_mapped = valid_dataset.map(lambda examples: {'text': [f'{system_message_prefix}\n{system_message}{end}\n{user_input_prefix}' + prompt + f'{end}\n{llm_output_prefix}' + response for prompt, response in zip(examples['prompt'], examples['response'])]}, batched=True)


def preprocess_function(examples):
    # 对文本进行编码
    tokenized_inputs = tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)
    # 计算模板部分的长度
    length_of_system_message_prefix = len(tokenizer(system_message_prefix)['input_ids']) - 2
    length_of_system_message = len(tokenizer(system_message)['input_ids']) - 2  # 减去特殊符号的长度
    length_of_user_input_prefix = len(tokenizer(user_input_prefix)['input_ids']) - 2
    length_of_llm_output_prefix = len(tokenizer(llm_output_prefix)['input_ids']) - 2
    length_of_end = len(tokenizer(end)['input_ids']) - 2
    total=length_of_system_message_prefix+length_of_system_message+length_of_end+length_of_user_input_prefix+length_of_end+length_of_llm_output_prefix+2
    # 初始化标签
    labels = []
    for i in range(len(tokenized_inputs["input_ids"])):
        # 创建一个初始标签列表
        label = [0] * len(tokenized_inputs["input_ids"][i])
        total =33

        label[:total] = [-100] * total

        labels.append(label)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Apply preprocessing and print a sample for debugging
# train_dataset_mapped = train_dataset_mapped.map(preprocess_function, batched=True)
# print(train_dataset_mapped[0]["text"], train_dataset_mapped[0]["labels"])


valid_dataset_mapped = valid_dataset_mapped.map(preprocess_function, batched=True)
print(valid_dataset_mapped["text"][0])


















compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device_map
)
model.config.use_cache = False
model.config.pretraining_tp = 1

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                    "lm_head",
]
)
# Set training parameters
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to="all",
    evaluation_strategy="steps",
    eval_steps=5  # Evaluate every 20 steps
)
# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset_mapped,
    eval_dataset=valid_dataset_mapped,  
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing,
)
trainer.train()



# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# pipe = pipeline("text-generation", model=model, torch_dtype=torch.bfloat16, device_map="auto")
# while True:
#     # Ask the user for their input
#     user_input = input("Enter your prompt: ")

#     # Prepare the messages for the pipeline
#     messages = [
#         {
#             "role": "system",
#             "content": "You are a ai researcher who do research on ai",
#         },
#         {"role": "user", "content": user_input},
#     ]

#     # Format the messages using the tokenizer's chat template
#     # prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#     input =tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#     # Generate a response using the pipeline
#     outputs = pipe(input, max_new_tokens=6000, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    
#     # Print the generated response
#     print(outputs[0]["generated_text"])





model_path = "./test_mistral"  # change to your preferred path


# Reload model in FP16 and merge it with LoRA weights
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=device_map,
)
model = PeftModel.from_pretrained(base_model, new_model)
model = model.merge_and_unload()

# Reload tokenizer to save it
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Save the merged model
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)