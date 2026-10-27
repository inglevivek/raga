# pyrefly: ignore [missing-import]
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model():
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    return model, tokenizer

    