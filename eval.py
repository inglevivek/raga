# pyrefly: ignore [missing-import]
from datasets import load_dataset
from model_loader import load_model
# pyrefly: ignore [missing-import]
import torch
import time
import requests

MODEL_PATH = "Qwen/Qwen3-0.6B"
NUM_SAMPLES = 20
OLLAMA_MODEL = "qwen3.5:2b"
model, tokenizer = load_model()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# EXACT same 1000 examples used during training
dataset = load_dataset(
    "tatsu-lab/alpaca",
    split="train"
).select(range(1000))

# EXACT same split as finetune.py
dataset = dataset.train_test_split(test_size=0.2, seed=42)
test_dataset = dataset["test"]

print(f"Evaluation samples: {len(test_dataset)}")

eval_pairs=[]
for example in test_dataset.select(range(NUM_SAMPLES)):

    prompt = example["instruction"]

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=128
    ).to(device)

    start = time.perf_counter()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    elapsed = time.perf_counter() - start

    generated_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]

    answer = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    print("\n" + "=" * 60)
    print("INSTRUCTION:")
    print(prompt)

    print("\nEXPECTED:")
    print(example["output"])

    print("\nMODEL:")
    print(answer)

    print(f"\nLatency: {elapsed:.2f}s")
    print(f"Tokens/sec: {generated_tokens / elapsed:.2f}")
    temporary = {
        "prompt": prompt,
        "expected": example["output"],
        "answer": answer
    }
    eval_pairs.append(temporary)



LLM_HOST_URL= "http://localhost:11434/api/generate"

def judge(query):
    instruction="""
    You are an LLM Judge.
    Evaluate the following prompt, expected output, and model's answer.
    If the model's answer effectively matches the core meaning of the expected output, print 'Correct'.
    If it misses the point or is factually wrong compared to the expected output, print 'Wrong'.
    
    Provide a score (1-10) for each of these criteria:
    Relevance: x/10 (How well the answer stays on topic to the prompt)
    Semantic Similarity: x/10 (How closely the meaning matches the expected text)
    Completeness: x/10 (Whether it answers all parts of the instruction)
    Factual Alignment: x/10 (Whether it contradicts the expected answer)
    """
    query_parameters={
        "model":OLLAMA_MODEL,
        "stream": False,
        "prompt": instruction+query,
        "options":{
            "thinking": True
        }
    }
    response= requests.post(LLM_HOST_URL, json=query_parameters)
    response= response.json()
    
    print(response['response'])

    
    
for pairs in eval_pairs:
    query = (
        f"Prompt: {pairs['prompt']}\n"
        f"Expected: {pairs['expected']}\n"
        f"Answer: {pairs['answer']}\n"
    )
 
    judge(query)






    