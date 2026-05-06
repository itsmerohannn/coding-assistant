from functools import lru_cache
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.config import GENERATION_MODEL

@lru_cache(maxsize=1)
def load_generation_pipeline(model_name: str = GENERATION_MODEL):
    """Load and cache a Causal LM."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto"
    )
    return tokenizer, model

def generate_text(generator, prompt: str, max_new_tokens: int = 512) -> str:
    tokenizer, model = generator
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    return tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()