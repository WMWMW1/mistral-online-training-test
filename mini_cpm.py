from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
torch.manual_seed(0)

path = 'openbmb/MiniCPM-2B-dpo-fp16'
tokenizer = AutoTokenizer.from_pretrained(path)
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
# print(responds)
# print(history)
messages=[]
modelinput= input("input from user: ")
responds, history = model.chat(tokenizer,modelinput , temperature=0.8, top_p=0.8)
# print(responds)
# print(history)
messages.append(history)
print(history)

while True:
    modelinput= input("input from user: ")
    responds, history = model.chat(tokenizer,modelinput , history,temperature=0.8, top_p=0.8)
    print(responds)
    print(history)
    messages.append(history)


# length = 0
# for response, history in model.stream_chat(tokenizer, "Hello", history=[]):
#     print(response[length:], flush=True, end="")
#     length = len(response)
        # print(tokenizer.decode(tokenized_chat[0]))
