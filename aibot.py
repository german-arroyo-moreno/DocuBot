import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

class AIBot:

    def __init__(self):
        self.system_token = "<|system|>"
        self.user_token = "<|user|>"
        self.assistant_token = "<|assistant|>"
        self.end_token = "\s"
        
        # trust_remote_code=True,
        # load_in_8bit=load_in_8bit,
        # load_in_4bit=load_in_4bit,
        # pad_token_id=0,
        # torch_dtype=torch_dtype,
        # device_map=device_map
        # device_map='auto', max_length=0,
        #            torch_dtype=None,
        #            load_in_8bit=False,
        #            load_in_4bit=False
                                                           
    def initialize(self, model_name, device_map='auto', max_length=0, **inputs):
        self._device_map = device_map
        self._tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                        **inputs)
        self._tokenizer.model_max_length = max_length
        self._model = AutoModelForCausalLM.from_pretrained(model_name, **inputs)
       
    def chat_to_str(self, chat_input, system_text=None,add_end_token=True):
        if system_text:
            txt = f"{self.system_token}{system_text}{self.end_token}\n"
        else:
            txt = ""
        if len(chat_input) == 0:
            return txt
        # Add the messages if any
        for chat in chat_input:
            role = chat['role']
            content = chat['content']
            if role == 'user':
                txt += f"{self.user_token}{content}"
            elif role == 'assistant':
                txt += f"{self.assistant_token}{content}"
            elif role == 'system':
                txt += f"{self.system_token}{content}"
            if add_end_token:
                txt += f"{self.end_token}\n"
        return txt

    def str_to_chat(text):
        """
        Split a dialogue into an array of [speaker, text] elements
        """
        pattern = r"\<\|(system|user|assistant|SYSTEM|USER|ASSISTANT)\|\>"
        data = re.split(pattern, text)[1:]
        return [ [data[i],data[i+1]] for i in range(0, len(data),2)]

    def remove_tokens(self, text, remove_end_token=True):
        text = text.replace(self.system_token, "")
        text = text.replace(self.assistant_token, "")
        text = text.replace(self.user_token, "")
        if remove_end_token:
            text = text.replace(self.end_token, "")
        return text

    def generate_text(self, prompt, max_new_tokens=20, min_length=0, max_iterations=1,
                      num_beams=5, num_beam_groups=5, diversity_penalty=1.0,
                      repetition_penalty=1.0,
                      stops_when_find_token=False):
        index = len(prompt)
        for i in range(max_iterations):
            if self._device_map == "cpu":
                inputs = self._tokenizer(prompt, return_tensors="pt").to('cpu')
            else:
                inputs = self._tokenizer(prompt, return_tensors="pt").to(0)
            # Generate the text
            generation_config = GenerationConfig(
                min_length=min_length,
                num_beams=num_beams,
                num_beam_groups=num_beam_groups,
                repetition_penalty=repetition_penalty,
                max_new_tokens=max_new_tokens,
                sliding_window=2048,
                diversity_penalty=diversity_penalty
            )
            outputs = self._model.generate(**inputs, generation_config=generation_config)
            text = self._tokenizer.batch_decode(outputs, skip_special_tokens=True)
            text_prompt = text[0]
            if stops_when_find_token:
                generated = text_prompt[index:]
                if (indexE := generated.find(self.end_token)) >= 0:
                    return generated[:indexE] + self.end_token
                if (indexE := generated.find(self.user_token)) >= 0:
                    return generated[:indexE]
                if (indexE := generated.find(self.assistant_token)) >= 0:
                    return generated[:indexE]
                if (indexE := generated.find(self.system_token)) >= 0:
                    return generated[:indexE]
        return text_prompt[index:]

    def text_has_end(self, text):
        return (text.find(self.end_token) >= 0)
