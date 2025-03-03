# 用于支持来自不同来源的模型，比如：
# 1. transformers
# 2. vllm
# 3. litellm
# 4. ollama

from typing import List, Dict

class TransformersModel():
    def __init__(self, model_name):
        
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def __call__(self, chat_history: List[Dict[str, str]]):
        prompt_tensor = self.tokenizer.apply_chat_template(
            chat_history,
            return_tensors='pt',
            return_dict=True
        )
        out = self.model.generate(**prompt_tensor)
        output = self.tokneizer.decode(out, skip_special_tokens=True)
        return output
    
if __name__ == "__main__":
    model = TransformersModel(
        model_name='../Qwen2.5-0.5B-Instruct'
    )
    chat_history = [
        {'role': 'user', 'text': '你好啊'}
    ]
    output = model(chat_history)
    print(output)