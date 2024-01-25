from aibot import AIBot
import torch

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
system_prompt = """
You are an useful assistant.
"""

text_prompt = f"<|system|>{system_prompt}\s\n<|user|>Explain to me what is an Aspect in Fate?<|assistant|>"


# tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer.model_max_length = 4096
# model = AutoModelForCausalLM.from_pretrained(model_name)
chat = AIBot()
# model_name = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
model_name = 'stabilityai/stablelm-zephyr-3b'
chat.initialize(model_name,
                max_length=1024,
                load_in_8bit=False,
                load_in_4bit=True,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map='cuda')


ln = 1
print("---------------")
accumulator = ""
while ln > 0:
    chunk = chat.generate_text(text_prompt,
                             max_new_tokens=2,
                             repetition_penalty=2.,
                             stops_when_find_token=True)
    ln = len(chunk)
    accumulator += chunk
    print(accumulator)
    if chat.text_has_end(accumulator):
        break
    else:
        text_prompt += chunk
print("---------------")
print (text_prompt)
